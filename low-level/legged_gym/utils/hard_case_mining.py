"""GPU-first hard-case detection, buffering, ranking and snapshot support."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def directional_manipulability_xy(
    arm_jacobian: torch.Tensor,
    delta_p_bar_xyz: torch.Tensor,
    *,
    damping: float = 1.0e-6,
    request_eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched directional manipulability in the requested world XY direction.

    Returns ``(value, valid_mask)``.  Invalid near-zero requests are represented
    as NaN so they cannot silently become hard cases.
    """

    if arm_jacobian.dim() != 3 or arm_jacobian.shape[1] < 2:
        raise ValueError("arm_jacobian must have shape [B, >=2, arm_dofs]")
    if delta_p_bar_xyz.shape != (arm_jacobian.shape[0], 3):
        raise ValueError("delta_p_bar_xyz must have shape [B, 3]")
    request_xy = delta_p_bar_xyz[:, :2]
    request_norm = torch.linalg.norm(request_xy, dim=-1)
    valid = request_norm > float(request_eps)
    direction = request_xy / torch.clamp(request_norm.unsqueeze(-1), min=float(request_eps))
    jacobian_xy = arm_jacobian[:, :2, :]
    gram = torch.bmm(jacobian_xy, jacobian_xy.transpose(1, 2))
    eye = torch.eye(2, device=gram.device, dtype=gram.dtype).expand(gram.shape[0], 2, 2)
    solved = torch.linalg.solve(gram + float(damping) * eye, direction.unsqueeze(-1))
    quadratic = torch.bmm(direction.unsqueeze(1), solved).reshape(-1)
    value = torch.rsqrt(torch.clamp(quadratic, min=1.0e-12))
    value = torch.where(valid, value, torch.full_like(value, float("nan")))
    return value, valid


class HardCaseDetector:
    """Vectorized persistent-condition detector with independent env counters."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        *,
        eta_raw_threshold: float = 0.0,
        eta_ratio_threshold: float = 0.20,
        ee_request_threshold: float = 0.02,
        ee_error_threshold: float = 0.05,
        min_steps: int = 10,
    ) -> None:
        if min_steps <= 0:
            raise ValueError("min_steps must be positive")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.eta_raw_threshold = float(eta_raw_threshold)
        self.eta_ratio_threshold = float(eta_ratio_threshold)
        self.ee_request_threshold = float(ee_request_threshold)
        self.ee_error_threshold = float(ee_error_threshold)
        self.min_steps = int(min_steps)
        self.persistence = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.triggered = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.first_low_manip_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.persistence.zero_()
            self.triggered.zero_()
            self.first_low_manip_step.fill_(-1)
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.persistence[env_ids] = 0
        self.triggered[env_ids] = False
        self.first_low_manip_step[env_ids] = -1

    def update(
        self,
        *,
        eta_raw: torch.Tensor,
        eta_ratio: torch.Tensor,
        ee_request_amplitude: torch.Tensor,
        ee_pos_error: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        sim_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        tensors = (eta_raw, eta_ratio, ee_request_amplitude, ee_pos_error)
        if any(tensor.shape != (self.num_envs,) for tensor in tensors):
            raise ValueError("Hard-case detector inputs must all have shape [num_envs]")
        if active_mask is None:
            active_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            active_mask = active_mask.to(device=self.device, dtype=torch.bool)

        low_ratio = eta_ratio < self.eta_ratio_threshold
        low_raw = (
            eta_raw < self.eta_raw_threshold
            if self.eta_raw_threshold > 0.0
            else torch.zeros_like(low_ratio)
        )
        low_manip = low_ratio | low_raw
        large_request = ee_request_amplitude > self.ee_request_threshold
        large_error = ee_pos_error > self.ee_error_threshold
        candidate = low_manip & large_request & large_error & active_mask

        newly_low = low_manip & active_mask & (self.first_low_manip_step < 0)
        if sim_step is not None:
            self.first_low_manip_step[newly_low] = int(sim_step)
        self.persistence = torch.where(candidate, self.persistence + 1, torch.zeros_like(self.persistence))
        new_trigger = (
            candidate
            & (self.persistence >= self.min_steps)
            & ~self.triggered
        )
        self.triggered |= new_trigger
        return new_trigger, {
            "candidate": candidate,
            "low_manip_ratio": low_ratio,
            "low_manip_raw": low_raw,
            "large_ee_request": large_request,
            "large_tracking_error": large_error,
            "persistence": self.persistence.clone(),
        }


class GPUStateRingBuffer:
    """Per-environment ring buffer with no step-wise device-to-host transfer."""

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if capacity <= 0 or feature_dim <= 0:
            raise ValueError("capacity and feature_dim must be positive")
        self.num_envs = int(num_envs)
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.data = torch.zeros(
            self.num_envs,
            self.capacity,
            self.feature_dim,
            device=self.device,
            dtype=dtype,
        )
        self.write_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.write_index.zero_()
            self.count.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.write_index[env_ids] = 0
        self.count[env_ids] = 0

    def append(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        if features.shape != (self.num_envs, self.feature_dim):
            raise ValueError(
                f"Expected features {(self.num_envs, self.feature_dim)}, got {tuple(features.shape)}"
            )
        if mask is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = mask.to(device=self.device, dtype=torch.bool).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        slots = self.write_index[env_ids]
        self.data[env_ids, slots] = features[env_ids]
        self.write_index[env_ids] = (slots + 1) % self.capacity
        self.count[env_ids] = torch.clamp(self.count[env_ids] + 1, max=self.capacity)

    def ordered(self, env_id: int) -> torch.Tensor:
        env_id = int(env_id)
        count = int(self.count[env_id].item())
        if count == 0:
            return self.data[env_id, :0]
        start = (int(self.write_index[env_id].item()) - count) % self.capacity
        indices = (torch.arange(count, device=self.device) + start) % self.capacity
        return self.data[env_id].index_select(0, indices)


class PeriodicSnapshotBuffer:
    """GPU ring of replay state captured at a configurable coarse interval."""

    def __init__(
        self,
        num_envs: int,
        slots: int,
        device: torch.device,
    ) -> None:
        self.num_envs = int(num_envs)
        self.slots = max(2, int(slots))
        self.device = torch.device(device)
        self.buffers: Dict[str, torch.Tensor] = {}
        self.step_buffer = torch.full(
            (self.num_envs, self.slots), -1, device=self.device, dtype=torch.long
        )
        self.write_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _initialize(self, state: Mapping[str, torch.Tensor]) -> None:
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != self.num_envs:
                raise ValueError(f"Snapshot field {name!r} must be a batched tensor")
            self.buffers[name] = torch.zeros(
                (self.num_envs, self.slots) + tuple(value.shape[1:]),
                device=self.device,
                dtype=value.dtype,
            )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.write_index.zero_()
            self.count.zero_()
            self.step_buffer.fill_(-1)
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.write_index[env_ids] = 0
        self.count[env_ids] = 0
        self.step_buffer[env_ids] = -1

    def capture(
        self,
        state: Mapping[str, torch.Tensor],
        sim_step: int,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.buffers:
            self._initialize(state)
        if tuple(state) != tuple(self.buffers):
            raise ValueError("Snapshot state fields changed after initialization")
        if mask is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = mask.to(device=self.device, dtype=torch.bool).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        slots = self.write_index[env_ids]
        for name, value in state.items():
            self.buffers[name][env_ids, slots] = value[env_ids]
        self.step_buffer[env_ids, slots] = int(sim_step)
        self.write_index[env_ids] = (slots + 1) % self.slots
        self.count[env_ids] = torch.clamp(self.count[env_ids] + 1, max=self.slots)

    def extract(self, env_id: int, desired_step: int) -> Dict[str, Any]:
        env_id = int(env_id)
        count = int(self.count[env_id].item())
        if count == 0:
            raise RuntimeError(f"No replay snapshot is available for env {env_id}")
        start = (int(self.write_index[env_id].item()) - count) % self.slots
        ordered_slots = (torch.arange(count, device=self.device) + start) % self.slots
        steps = self.step_buffer[env_id].index_select(0, ordered_slots)
        eligible = (steps <= int(desired_step)).nonzero(as_tuple=False).flatten()
        ordered_index = int(eligible[-1].item()) if eligible.numel() else 0
        slot = int(ordered_slots[ordered_index].item())
        return {
            "snapshot_sim_step": int(self.step_buffer[env_id, slot].item()),
            "state": {
                name: value[env_id, slot].detach().cpu().clone()
                for name, value in self.buffers.items()
            },
        }


def pack_feature_dict(
    features: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int, Tuple[int, ...]]]]:
    """Flatten a stable mapping of batched tensors into one GPU feature matrix."""

    if not features:
        raise ValueError("features cannot be empty")
    batch_size = None
    flat_values = []
    layout: Dict[str, Tuple[int, int, Tuple[int, ...]]] = {}
    offset = 0
    for name, value in features.items():
        if not isinstance(value, torch.Tensor) or value.dim() == 0:
            raise ValueError(f"Feature {name!r} must be a batched tensor")
        if batch_size is None:
            batch_size = value.shape[0]
        elif value.shape[0] != batch_size:
            raise ValueError(f"Feature {name!r} has a mismatched batch size")
        shape = tuple(value.shape[1:])
        flat = value.reshape(value.shape[0], -1).to(dtype=torch.float32)
        width = flat.shape[1]
        layout[name] = (offset, offset + width, shape)
        offset += width
        flat_values.append(flat)
    return torch.cat(flat_values, dim=-1), layout


def unpack_feature_trajectory(
    trajectory: torch.Tensor,
    layout: Mapping[str, Tuple[int, int, Tuple[int, ...]]],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    trajectory = trajectory.detach().cpu()
    for name, (start, end, shape) in layout.items():
        value = trajectory[:, start:end]
        value = value.reshape((value.shape[0],) + tuple(shape))
        arrays[name] = value.numpy()
    return arrays


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


@dataclass
class _CaseRecord:
    case_id: int
    env_id: int
    sample_id: int
    batch_id: int
    episode_id: int
    trigger_step: int
    trigger_index: int
    first_low_manip_step: int
    snapshot_payload: Dict[str, Any]
    metadata: Dict[str, Any]
    trajectory_index: int
    trajectory_length: int = 0
    finalized: bool = False
    arrays: Optional[Dict[str, np.ndarray]] = None
    hardness_score: float = float("-inf")
    snapshot_path: str = ""


class HardCaseMiner:
    """Coordinate detector, GPU buffers, replay snapshots and case outputs."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device,
        dt: float,
        out_dir: str,
        pre_seconds: float = 1.0,
        post_seconds: float = 3.0,
        min_steps: int = 10,
        max_cases: int = 200,
        max_video_candidates: int = 20,
        eta_raw_threshold: float = 0.0,
        eta_ratio_threshold: float = 0.20,
        ee_request_threshold: float = 0.02,
        ee_error_threshold: float = 0.05,
        snapshot_interval_seconds: float = 0.25,
        snapshot_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.out_dir = out_dir
        self.cases_dir = os.path.join(out_dir, "cases")
        os.makedirs(self.cases_dir, exist_ok=True)
        self.pre_steps = max(1, int(round(float(pre_seconds) / self.dt)))
        self.post_steps = max(0, int(round(float(post_seconds) / self.dt)))
        self.snapshot_interval_steps = max(
            1, int(round(float(snapshot_interval_seconds) / self.dt))
        )
        snapshot_slots = math.ceil(self.pre_steps / self.snapshot_interval_steps) + 3
        self.max_cases = max(0, int(max_cases))
        self.max_video_candidates = max(0, int(max_video_candidates))
        self.detector = HardCaseDetector(
            self.num_envs,
            self.device,
            eta_raw_threshold=eta_raw_threshold,
            eta_ratio_threshold=eta_ratio_threshold,
            ee_request_threshold=ee_request_threshold,
            ee_error_threshold=ee_error_threshold,
            min_steps=min_steps,
        )
        self.snapshot_buffer = PeriodicSnapshotBuffer(
            self.num_envs, snapshot_slots, self.device
        )
        self.ring: Optional[GPUStateRingBuffer] = None
        self.feature_layout: Optional[Dict[str, Tuple[int, int, Tuple[int, ...]]]] = None
        self.case_trajectories: Optional[torch.Tensor] = None
        self.case_write_index: Optional[torch.Tensor] = None
        self.slot_in_use: Optional[torch.Tensor] = None
        self.env_to_case = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.active_records: Dict[int, _CaseRecord] = {}
        self.records: List[_CaseRecord] = []
        self.detected_cases = 0
        self.completed_candidates = 0
        self.next_candidate_id = 1
        self.snapshot_save_time = 0.0
        self.trajectory_save_time = 0.0
        self.video_save_time = 0.0
        self.snapshot_metadata = dict(snapshot_metadata or {})

    @property
    def saved_cases(self) -> int:
        return len(self.records)

    def _initialize_features(self, packed: torch.Tensor, layout: Dict[str, Any]) -> None:
        self.feature_layout = layout
        self.ring = GPUStateRingBuffer(
            self.num_envs,
            self.pre_steps + 1,
            packed.shape[1],
            self.device,
        )
        max_length = self.pre_steps + 1 + self.post_steps
        self.case_trajectories = torch.zeros(
            max(1, self.num_envs),
            max_length,
            packed.shape[1],
            device=self.device,
            dtype=packed.dtype,
        )
        self.case_write_index = torch.zeros(
            max(1, self.num_envs), device=self.device, dtype=torch.long
        )
        self.slot_in_use = torch.zeros(
            max(1, self.num_envs), device=self.device, dtype=torch.bool
        )

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.finish_envs(env_ids)
        self.detector.reset(env_ids)
        if self.ring is not None:
            self.ring.reset(env_ids)
        self.snapshot_buffer.reset(env_ids)

    def capture_snapshot(
        self,
        snapshot_state: Mapping[str, torch.Tensor],
        sim_step: int,
        active_mask: Optional[torch.Tensor] = None,
        *,
        force: bool = False,
    ) -> None:
        if force or int(sim_step) % self.snapshot_interval_steps == 0:
            self.snapshot_buffer.capture(snapshot_state, int(sim_step), active_mask)

    def update(
        self,
        *,
        features: Mapping[str, torch.Tensor],
        snapshot_state: Mapping[str, torch.Tensor],
        active_mask: torch.Tensor,
        sim_step: int,
        sample_ids: torch.Tensor,
        batch_id: int,
        episode_ids: torch.Tensor,
        scheduled_base_commands: torch.Tensor,
        scheduled_ee_commands: torch.Tensor,
    ) -> torch.Tensor:
        packed, layout = pack_feature_dict(features)
        if self.ring is None:
            self._initialize_features(packed, layout)
        elif layout != self.feature_layout:
            raise ValueError("Hard-case numerical feature layout changed during rollout")
        assert self.ring is not None
        assert self.case_trajectories is not None
        assert self.case_write_index is not None

        active_mask = active_mask.to(device=self.device, dtype=torch.bool)
        self.capture_snapshot(snapshot_state, sim_step, active_mask)

        # Append this step to already-open candidates before detecting new ones.
        recording_envs = (self.env_to_case >= 0) & active_mask
        recording_ids = recording_envs.nonzero(as_tuple=False).flatten()
        if recording_ids.numel() > 0:
            case_ids = self.env_to_case[recording_ids]
            write_positions = self.case_write_index[case_ids]
            within = write_positions < self.case_trajectories.shape[1]
            if within.any():
                self.case_trajectories[
                    case_ids[within], write_positions[within]
                ] = packed[recording_ids[within]]
                self.case_write_index[case_ids[within]] += 1
            complete_case_ids = case_ids[
                self.case_write_index[case_ids] >= self.case_trajectories.shape[1]
            ].unique()
            for trajectory_index in complete_case_ids.tolist():
                self._finalize_trajectory_index(int(trajectory_index))

        self.ring.append(packed, active_mask)
        low_before_eligible = (
            features["eta_ratio"].reshape(-1) < self.detector.eta_ratio_threshold
        )
        if self.detector.eta_raw_threshold > 0.0:
            low_before_eligible |= (
                features["eta_raw"].reshape(-1) < self.detector.eta_raw_threshold
            )
        first_low = (
            low_before_eligible
            & active_mask
            & (self.detector.first_low_manip_step < 0)
        )
        self.detector.first_low_manip_step[first_low] = int(sim_step)
        detection_mask = active_mask & (self.ring.count >= self.pre_steps)
        new_trigger, conditions = self.detector.update(
            eta_raw=features["eta_raw"].reshape(-1),
            eta_ratio=features["eta_ratio"].reshape(-1),
            ee_request_amplitude=features["ma2b_ee_request_amplitude_current"].reshape(-1),
            ee_pos_error=features["ee_pos_error"].reshape(-1),
            active_mask=detection_mask,
            sim_step=sim_step,
        )
        trigger_env_ids = new_trigger.nonzero(as_tuple=False).flatten()
        self.detected_cases += int(trigger_env_ids.numel())
        for env_id in trigger_env_ids.tolist():
            if self.max_cases <= 0:
                continue
            self._open_case(
                env_id=env_id,
                sim_step=sim_step,
                sample_id=int(sample_ids[env_id].item()),
                batch_id=int(batch_id),
                episode_id=int(episode_ids[env_id].item()),
                scheduled_base=scheduled_base_commands[env_id],
                scheduled_ee=scheduled_ee_commands[env_id],
                features=features,
                persistence=int(conditions["persistence"][env_id].item()),
            )
        return new_trigger

    def _open_case(
        self,
        *,
        env_id: int,
        sim_step: int,
        sample_id: int,
        batch_id: int,
        episode_id: int,
        scheduled_base: torch.Tensor,
        scheduled_ee: torch.Tensor,
        features: Mapping[str, torch.Tensor],
        persistence: int,
    ) -> None:
        assert self.ring is not None
        assert self.case_trajectories is not None
        assert self.case_write_index is not None
        assert self.slot_in_use is not None
        free_slots = (~self.slot_in_use).nonzero(as_tuple=False).flatten()
        if free_slots.numel() == 0:
            raise RuntimeError(
                "No free hard-case trajectory slot; each environment should have "
                "at most one active candidate"
            )
        trajectory_index = int(free_slots[0].item())
        self.slot_in_use[trajectory_index] = True
        candidate_id = self.next_candidate_id
        self.next_candidate_id += 1
        pre = self.ring.ordered(env_id)
        length = min(pre.shape[0], self.case_trajectories.shape[1])
        self.case_trajectories[trajectory_index, :length] = pre[-length:]
        self.case_write_index[trajectory_index] = length
        self.env_to_case[env_id] = trajectory_index

        desired_snapshot_step = int(sim_step) - self.pre_steps
        snapshot = self.snapshot_buffer.extract(env_id, desired_snapshot_step)
        snapshot_payload = {
            **snapshot,
            "evaluation_config": self.snapshot_metadata,
            "source_env_id": int(env_id),
            "sample_id": int(sample_id),
            "candidate_id": int(candidate_id),
            "batch_id": int(batch_id),
            "trigger_sim_step": int(sim_step),
            "desired_pre_trigger_step": desired_snapshot_step,
            "rng_state_at_trigger": capture_rng_state(),
            "determinism_note": (
                "Actor/DOF and Python-visible environment buffers are restored. "
                "PhysX contact caches and GPU kernel scheduling are not serialized, "
                "so replay is state-matched but not guaranteed bitwise deterministic. "
                "For non-deterministic Stage-A runs, per-env actor mass/friction "
                "properties set during creation also cannot be fully rewritten from "
                "tensor snapshots; use --eval_deterministic for matched case replay."
            ),
        }
        def scalar(name: str) -> float:
            value = features[name][env_id].reshape(-1)[0]
            return float(value.detach().item())

        metadata: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "sample_id": sample_id,
            "batch_id": batch_id,
            "env_id": env_id,
            "episode_id": episode_id,
            "trigger_step": int(sim_step),
            "snapshot_step": snapshot["snapshot_sim_step"],
            "first_low_manip_step": int(self.detector.first_low_manip_step[env_id].item()),
            "base_command": scheduled_base.detach().cpu().tolist(),
            "ee_goal_sphere": scheduled_ee[:3].detach().cpu().tolist(),
            "ee_goal_orientation_delta": scheduled_ee[3:].detach().cpu().tolist(),
            "eta_raw_at_trigger": scalar("eta_raw"),
            "eta_ratio_at_trigger": scalar("eta_ratio"),
            "manip_log_eta_at_trigger": scalar("manip_log_eta"),
            "manip_log_eta_ref": scalar("manip_log_eta_ref"),
            "directional_manip_at_trigger": scalar("directional_manipulability_xy"),
            "ee_request_amplitude": scalar("ma2b_ee_request_amplitude_current"),
            "ee_error_at_trigger": scalar("ee_pos_error"),
            "base_assist_alignment_raw_at_trigger": scalar("base_assist_alignment_raw"),
            "persistence_steps_at_trigger": persistence,
            "pre_trigger_seconds_requested": self.pre_steps * self.dt,
            "post_trigger_seconds_requested": self.post_steps * self.dt,
        }
        self.active_records[trajectory_index] = _CaseRecord(
                case_id=candidate_id,
                env_id=env_id,
                sample_id=sample_id,
                batch_id=batch_id,
                episode_id=episode_id,
                trigger_step=int(sim_step),
                trigger_index=max(0, length - 1),
                first_low_manip_step=int(self.detector.first_low_manip_step[env_id].item()),
                snapshot_payload=snapshot_payload,
                metadata=metadata,
                trajectory_index=trajectory_index,
                trajectory_length=length,
        )

    def _finalize_trajectory_index(self, trajectory_index: int) -> None:
        record = self.active_records.get(int(trajectory_index))
        if record is None:
            return
        assert self.case_write_index is not None
        assert self.case_trajectories is not None
        assert self.feature_layout is not None
        assert self.slot_in_use is not None
        record.trajectory_length = int(self.case_write_index[trajectory_index].item())
        trajectory = self.case_trajectories[
            trajectory_index, : record.trajectory_length
        ]
        record.arrays = unpack_feature_trajectory(trajectory, self.feature_layout)
        record.hardness_score = self._rank_case(record.arrays, record.metadata)
        record.metadata["trajectory_steps"] = record.trajectory_length
        record.metadata["trigger_trajectory_index"] = record.trigger_index
        record.finalized = True
        if self.env_to_case[record.env_id] == trajectory_index:
            self.env_to_case[record.env_id] = -1
        self.case_write_index[trajectory_index] = 0
        self.slot_in_use[trajectory_index] = False
        del self.active_records[trajectory_index]
        self.completed_candidates += 1

        self.records.append(record)
        self.records.sort(key=lambda item: (-item.hardness_score, item.case_id))
        if len(self.records) > self.max_cases:
            self.records.pop()

    def finish_envs(self, env_ids: torch.Tensor) -> None:
        if not self.active_records:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        trajectory_indices = self.env_to_case[env_ids]
        for trajectory_index in trajectory_indices[trajectory_indices >= 0].unique().tolist():
            self._finalize_trajectory_index(int(trajectory_index))

    def _rank_case(self, arrays: Mapping[str, np.ndarray], metadata: Dict[str, Any]) -> float:
        eta_raw = np.asarray(arrays["eta_raw"]).reshape(-1)
        eta_ratio = np.asarray(arrays["eta_ratio"]).reshape(-1)
        directional = np.asarray(arrays["directional_manipulability_xy"]).reshape(-1)
        request = np.asarray(arrays["ma2b_ee_request_amplitude_current"]).reshape(-1)
        error = np.asarray(arrays["ee_pos_error"]).reshape(-1)
        alignment = np.asarray(arrays["base_assist_alignment_raw"]).reshape(-1)
        low = eta_ratio < self.detector.eta_ratio_threshold
        if self.detector.eta_raw_threshold > 0.0:
            low |= eta_raw < self.detector.eta_raw_threshold
        low_duration = float(np.sum(low) * self.dt)
        raw_severity = (
            max(
                0.0,
                1.0
                - float(np.nanmin(eta_raw))
                / self.detector.eta_raw_threshold,
            )
            if self.detector.eta_raw_threshold > 0.0
            else 0.0
        )
        ratio_severity = max(0.0, 1.0 - float(np.nanmin(eta_ratio)) / max(self.detector.eta_ratio_threshold, 1e-9))
        finite_directional = directional[np.isfinite(directional)]
        directional_severity = 0.0 if finite_directional.size == 0 else 1.0 / (1.0 + float(np.nanmin(finite_directional)))
        request_severity = float(np.nanmax(request)) / max(self.detector.ee_request_threshold, 1e-9)
        error_severity = float(np.nanmax(error)) / max(self.detector.ee_error_threshold, 1e-9)
        poor_alignment = max(0.0, -float(np.nanmean(alignment)))
        score = (
            raw_severity
            + ratio_severity
            + directional_severity
            + min(request_severity, 5.0)
            + min(error_severity, 5.0)
            + min(low_duration / max(self.dt * self.detector.min_steps, 1e-9), 5.0)
            + poor_alignment
        )
        metadata.update(
            hardness_score=float(score),
            low_manip_duration=low_duration,
            minimum_eta_raw=float(np.nanmin(eta_raw)),
            minimum_eta_ratio=float(np.nanmin(eta_ratio)),
            peak_ee_error=float(np.nanmax(error)),
            peak_ee_request=float(np.nanmax(request)),
            mean_base_assist_alignment_raw=float(np.nanmean(alignment)),
            peak_error_step=int(np.nanargmax(error)),
            minimum_eta_step=int(np.nanargmin(eta_ratio)),
        )
        return float(score)

    def save(
        self,
        sample_records: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if self.active_records:
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            self.finish_envs(all_env_ids)
        outcome_by_sample = {
            int(row["sample_id"]): row
            for row in (sample_records or [])
            if row.get("status") == "executed"
        }
        case_rows: List[Dict[str, Any]] = []
        self.records.sort(key=lambda item: (-item.hardness_score, item.case_id))
        for rank, record in enumerate(self.records, start=1):
            assert record.arrays is not None
            arrays = record.arrays
            record.case_id = rank
            record.metadata["case_id"] = rank
            record.metadata["rank"] = rank
            record.metadata["video_candidate"] = rank <= self.max_video_candidates
            case_dir = os.path.join(self.cases_dir, f"case_{rank:06d}")
            os.makedirs(case_dir, exist_ok=True)
            record.snapshot_path = os.path.join(case_dir, "snapshot.pt")
            trajectory_path = os.path.join(case_dir, "trajectory.npz")
            metadata_path = os.path.join(case_dir, "metadata.json")
            save_start = time.perf_counter()
            torch.save(record.snapshot_payload, record.snapshot_path)
            self.snapshot_save_time += time.perf_counter() - save_start
            save_start = time.perf_counter()
            np.savez_compressed(trajectory_path, **arrays)
            self.trajectory_save_time += time.perf_counter() - save_start
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(record.metadata, file, indent=2, sort_keys=True, allow_nan=True)
            case_row = dict(
                record.metadata,
                snapshot=record.snapshot_path,
                hardness_score=record.hardness_score,
            )
            sample_outcome = outcome_by_sample.get(record.sample_id)
            if sample_outcome is not None:
                for key in (
                    "success", "fall", "collision", "timeout", "ee_rmse",
                    "termination_reason", "evaluation_steps",
                ):
                    if key in sample_outcome:
                        case_row[key] = sample_outcome[key]
                record.metadata["sample_outcome"] = {
                    key: sample_outcome[key]
                    for key in (
                        "success", "fall", "collision", "timeout", "ee_rmse",
                        "termination_reason", "evaluation_steps",
                    )
                    if key in sample_outcome
                }
                with open(metadata_path, "w", encoding="utf-8") as file:
                    json.dump(record.metadata, file, indent=2, sort_keys=True, allow_nan=True)
            case_rows.append(case_row)

        hard_cases_path = os.path.join(self.out_dir, "hard_cases.csv")
        if case_rows:
            fieldnames = sorted({key for row in case_rows for key in row})
            with open(hard_cases_path, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(case_rows)
        else:
            with open(hard_cases_path, "w", encoding="utf-8") as file:
                file.write("case_id,rank,hardness_score\n")

        replay_queue = [
            {
                "rank": row["rank"],
                "case_id": row["case_id"],
                "snapshot": row["snapshot"],
                "hardness_score": row["hardness_score"],
            }
            for row in case_rows[: self.max_video_candidates]
        ]
        replay_queue_path = os.path.join(self.out_dir, "replay_candidates.json")
        with open(replay_queue_path, "w", encoding="utf-8") as file:
            json.dump(replay_queue, file, indent=2, sort_keys=True)

        subset_stats = self._subset_statistics(case_rows)
        subset_path = os.path.join(self.out_dir, "hard_case_subset_stats.json")
        with open(subset_path, "w", encoding="utf-8") as file:
            json.dump(subset_stats, file, indent=2, sort_keys=True, allow_nan=True)
        return {
            "hard_cases_csv": hard_cases_path,
            "replay_candidates_json": replay_queue_path,
            "subset_stats_json": subset_path,
            "detected_cases": self.detected_cases,
            "completed_candidates": self.completed_candidates,
            "saved_cases": len(case_rows),
            "snapshot_save_time_sec": self.snapshot_save_time,
            "trajectory_save_time_sec": self.trajectory_save_time,
            "video_save_time_sec": self.video_save_time,
        }

    def _subset_statistics(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {
                "number_of_hard_cases": 0,
                "hard_case_rate_denominator": None,
                "hard_case_rate": None,
            }

        def mean(name: str) -> float:
            return float(np.mean([float(row[name]) for row in rows]))

        return {
            "number_of_hard_cases": len(rows),
            "mean_min_eta_ratio": mean("minimum_eta_ratio"),
            "minimum_eta_ratio": min(float(row["minimum_eta_ratio"]) for row in rows),
            "mean_low_manip_duration": mean("low_manip_duration"),
            "mean_peak_ee_error": mean("peak_ee_error"),
            "mean_base_assist_alignment_raw": mean("mean_base_assist_alignment_raw"),
            "ee_rmse": (
                float(np.mean([float(row["ee_rmse"]) for row in rows if "ee_rmse" in row]))
                if any("ee_rmse" in row for row in rows) else None
            ),
            "success_rate": (
                float(np.mean([bool(row["success"]) for row in rows if "success" in row]))
                if any("success" in row for row in rows) else None
            ),
            "fall_rate": (
                float(np.mean([bool(row["fall"]) for row in rows if "fall" in row]))
                if any("fall" in row for row in rows) else None
            ),
        }


__all__ = [
    "GPUStateRingBuffer",
    "HardCaseDetector",
    "HardCaseMiner",
    "PeriodicSnapshotBuffer",
    "capture_rng_state",
    "directional_manipulability_xy",
    "pack_feature_dict",
    "unpack_feature_trajectory",
]
