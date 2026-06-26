from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class HighLevelActionDims:
    base_command_dim: int = 5
    ee_delta_dim: int = 6
    chunk_horizon: int = 3
    grasp_intent_dim: int = 1
    h_choice_dim: int = 3

    @property
    def total_action_dim(self) -> int:
        return (
            self.base_command_dim
            + self.chunk_horizon * self.ee_delta_dim
            + self.grasp_intent_dim
            + self.h_choice_dim
        )


@dataclass(frozen=True)
class HighLevelActionLimits:
    max_vx: float = 0.5
    max_vy: float = 0.3
    max_yaw_rate: float = 0.8
    max_pitch: float = 0.4
    min_base_height: float = 0.25
    max_base_height: float = 0.55
    max_ee_pos_delta: float = 0.05
    max_ee_rot_delta: float = 0.2


def flatten_high_level_action(action_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a structured high-level action into the flat PPO action tensor."""
    base_command = action_dict["base_command"]
    arm_action_chunk = action_dict["arm_action_chunk"]
    grasp_intent = action_dict.get("grasp_intent", action_dict.get("grasp_intent_logit"))
    h_logits = action_dict["h_logits"]

    if grasp_intent is None:
        raise KeyError("action_dict must contain grasp_intent or grasp_intent_logit")
    if arm_action_chunk.dim() != 3 or arm_action_chunk.shape[-1] != 6:
        raise ValueError("arm_action_chunk must have shape [B, H, 6]")

    batch = base_command.shape[0]
    _require_batch("arm_action_chunk", arm_action_chunk, batch)
    _require_batch("grasp_intent", grasp_intent, batch)
    _require_batch("h_logits", h_logits, batch)

    return torch.cat(
        [
            base_command,
            arm_action_chunk.reshape(batch, -1),
            grasp_intent,
            h_logits,
        ],
        dim=-1,
    )


def unflatten_high_level_action(
    action_tensor: torch.Tensor,
    chunk_horizon: int,
    base_command_dim: int = 5,
    ee_delta_dim: int = 6,
    grasp_intent_dim: int = 1,
    h_choice_dim: int = 3,
) -> Dict[str, torch.Tensor]:
    """Recover structured high-level action fields from a flat tensor."""
    if action_tensor.dim() != 2:
        raise ValueError("action_tensor must have shape [B, total_action_dim]")

    dims = HighLevelActionDims(
        base_command_dim=base_command_dim,
        ee_delta_dim=ee_delta_dim,
        chunk_horizon=chunk_horizon,
        grasp_intent_dim=grasp_intent_dim,
        h_choice_dim=h_choice_dim,
    )
    if action_tensor.shape[-1] != dims.total_action_dim:
        raise ValueError(
            f"Expected flat action dim {dims.total_action_dim}, got {action_tensor.shape[-1]}"
        )

    idx = 0
    base_command = action_tensor[:, idx : idx + base_command_dim]
    idx += base_command_dim
    arm_flat_dim = chunk_horizon * ee_delta_dim
    arm_action_chunk = action_tensor[:, idx : idx + arm_flat_dim].reshape(
        action_tensor.shape[0], chunk_horizon, ee_delta_dim
    )
    idx += arm_flat_dim
    grasp_intent = action_tensor[:, idx : idx + grasp_intent_dim]
    idx += grasp_intent_dim
    h_logits = action_tensor[:, idx : idx + h_choice_dim]

    return {
        "base_command": base_command,
        "arm_action_chunk": arm_action_chunk,
        "grasp_intent": grasp_intent,
        "h_logits": h_logits,
    }


def select_h_from_logits(h_logits: torch.Tensor) -> torch.Tensor:
    """Map h logits to h in {1, 2, 3} by argmax for deterministic execution."""
    if h_logits.shape[-1] != 3:
        raise ValueError("h_logits last dimension must be 3")
    return torch.argmax(h_logits, dim=-1) + 1


def clip_high_level_action(
    action_dict: Dict[str, torch.Tensor],
    limits: Optional[HighLevelActionLimits] = None,
) -> Dict[str, torch.Tensor]:
    """Clip safety-critical fields without changing logits."""
    limits = limits or HighLevelActionLimits()
    clipped = dict(action_dict)

    base = clipped["base_command"].clone()
    base[:, 0] = base[:, 0].clamp(-limits.max_vx, limits.max_vx)
    base[:, 1] = base[:, 1].clamp(-limits.max_vy, limits.max_vy)
    base[:, 2] = base[:, 2].clamp(-limits.max_yaw_rate, limits.max_yaw_rate)
    base[:, 3] = base[:, 3].clamp(-limits.max_pitch, limits.max_pitch)
    base[:, 4] = base[:, 4].clamp(limits.min_base_height, limits.max_base_height)
    clipped["base_command"] = base

    chunk = clipped["arm_action_chunk"].clone()
    chunk[..., :3] = chunk[..., :3].clamp(-limits.max_ee_pos_delta, limits.max_ee_pos_delta)
    chunk[..., 3:] = chunk[..., 3:].clamp(-limits.max_ee_rot_delta, limits.max_ee_rot_delta)
    clipped["arm_action_chunk"] = chunk

    if "grasp_intent" in clipped:
        clipped["grasp_intent"] = clipped["grasp_intent"].clamp(0.0, 1.0)
    return clipped


def clip_flat_high_level_action(
    action_tensor: torch.Tensor,
    chunk_horizon: int,
    limits: Optional[HighLevelActionLimits] = None,
) -> torch.Tensor:
    action_dict = unflatten_high_level_action(action_tensor, chunk_horizon)
    return flatten_high_level_action(clip_high_level_action(action_dict, limits))


def _require_batch(name: str, tensor: torch.Tensor, batch: int) -> None:
    if tensor.shape[0] != batch:
        raise ValueError(f"{name} batch size {tensor.shape[0]} does not match {batch}")

