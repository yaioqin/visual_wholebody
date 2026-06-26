from dataclasses import dataclass
from typing import Optional, Union

import torch


class KinematicsInterface:
    """Interface for robot-specific FK/Jacobian/IK backends.

    TODO: Wire this to the real B1+Z1 kinematics backend if a project-level
    implementation is added. The controller intentionally does not hard-code a
    robot model.
    """

    def solve_ik(self, current_joint_pos, current_ee_pose, target_ee_pose, q_prev=None):
        raise NotImplementedError

    def jacobian(self, current_joint_pos):
        raise NotImplementedError


@dataclass(frozen=True)
class ArmIKControllerConfig:
    damping: float = 0.05
    max_ee_pos_delta: float = 0.05
    max_ee_rot_delta: float = 0.2
    max_joint_delta: float = 0.25


class ArmIKController:
    """IK-based arm controller for action chunks.

    A real kinematics backend can be passed through KinematicsInterface. When a
    batched arm_jacobian is supplied, the controller runs damped least-squares IK
    directly and clips the resulting joint targets.
    """

    def __init__(
        self,
        num_arm_joints: int,
        joint_lower_limits: Optional[torch.Tensor] = None,
        joint_upper_limits: Optional[torch.Tensor] = None,
        config: Optional[ArmIKControllerConfig] = None,
        kinematics: Optional[KinematicsInterface] = None,
    ):
        self.num_arm_joints = num_arm_joints
        self.config = config or ArmIKControllerConfig()
        self.kinematics = kinematics
        self.joint_lower_limits = joint_lower_limits
        self.joint_upper_limits = joint_upper_limits

    def solve(
        self,
        current_ee_pose: torch.Tensor,
        action_chunk: torch.Tensor,
        h: Union[int, torch.Tensor],
        current_joint_pos: torch.Tensor,
        arm_jacobian: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute the first h chunk steps and return arm joint targets."""
        self._validate_inputs(current_ee_pose, action_chunk, current_joint_pos, arm_jacobian)

        batch, horizon, _ = action_chunk.shape
        q_target = current_joint_pos.clone()
        target_ee_pose = current_ee_pose[..., :6].clone()
        h_tensor = self._normalize_h(h, batch, action_chunk.device)
        max_steps = int(torch.clamp(h_tensor.max(), min=0, max=horizon).item())

        for step_idx in range(max_steps):
            active = h_tensor > step_idx
            if not torch.any(active):
                continue
            delta = self._clip_delta(action_chunk[:, step_idx])
            target_ee_pose = torch.where(active[:, None], target_ee_pose + delta, target_ee_pose)

            if self.kinematics is not None:
                q_next = self.kinematics.solve_ik(
                    current_joint_pos=q_target,
                    current_ee_pose=current_ee_pose[..., :6],
                    target_ee_pose=target_ee_pose,
                    q_prev=q_target,
                )
            elif arm_jacobian is not None:
                dpose = torch.zeros_like(target_ee_pose)
                dpose[active] = delta[active]
                dq = self._damped_least_squares(arm_jacobian, dpose.unsqueeze(-1)).squeeze(-1)
                q_next = q_target + dq.clamp(-self.config.max_joint_delta, self.config.max_joint_delta)
            else:
                # No robot-specific IK backend is available yet. Keep the last
                # safe target instead of inventing a robot model.
                q_next = q_target

            q_target = torch.where(active[:, None], q_next, q_target)
            q_target = self._clip_joint_limits(q_target)

        return torch.nan_to_num(q_target, nan=0.0, posinf=0.0, neginf=0.0)

    def step(
        self,
        current_ee_pos: torch.Tensor,
        current_ee_rot: torch.Tensor,
        current_arm_q: torch.Tensor,
        arm_delta_action: torch.Tensor,
        jacobian: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one end-effector delta action and return arm joint targets.

        Args:
            current_ee_pos: Tensor [B, 3].
            current_ee_rot: Tensor [B, 3], [B, 4] quat or [B, 3, 3]. The DLS
                path only needs the delta pose and accepts any of these shapes.
            current_arm_q: Tensor [B, 6].
            arm_delta_action: Tensor [B, 6] = [dx, dy, dz, droll, dpitch, dyaw].
            jacobian: Optional arm-only Jacobian [B, 6, 6].

        Returns:
            Tensor [B, 6] clipped to configured joint limits when available.
        """
        if current_ee_pos.dim() != 2 or current_ee_pos.shape[-1] != 3:
            raise ValueError("current_ee_pos must have shape [B, 3]")
        if arm_delta_action.dim() != 2 or arm_delta_action.shape[-1] != 6:
            raise ValueError("arm_delta_action must have shape [B, 6]")

        batch = current_ee_pos.shape[0]
        current_ee_pose = torch.zeros(batch, 6, device=current_ee_pos.device, dtype=current_ee_pos.dtype)
        current_ee_pose[:, :3] = current_ee_pos
        if current_ee_rot.dim() == 2 and current_ee_rot.shape[-1] == 3:
            current_ee_pose[:, 3:] = current_ee_rot

        return self.solve(
            current_ee_pose=current_ee_pose,
            action_chunk=arm_delta_action.view(batch, 1, 6),
            h=1,
            current_joint_pos=current_arm_q,
            arm_jacobian=jacobian,
        )

    def _damped_least_squares(self, jacobian: torch.Tensor, dpose: torch.Tensor) -> torch.Tensor:
        rows = jacobian.shape[1]
        j_t = jacobian.transpose(1, 2)
        eye = torch.eye(rows, device=jacobian.device, dtype=jacobian.dtype).expand(jacobian.shape[0], rows, rows)
        lhs = jacobian @ j_t + (self.config.damping ** 2) * eye
        return j_t @ torch.linalg.solve(lhs, dpose)

    def _clip_delta(self, delta: torch.Tensor) -> torch.Tensor:
        clipped = delta.clone()
        clipped[..., :3] = clipped[..., :3].clamp(
            -self.config.max_ee_pos_delta, self.config.max_ee_pos_delta
        )
        clipped[..., 3:] = clipped[..., 3:].clamp(
            -self.config.max_ee_rot_delta, self.config.max_ee_rot_delta
        )
        return clipped

    def _clip_joint_limits(self, q: torch.Tensor) -> torch.Tensor:
        if self.joint_lower_limits is not None:
            q = torch.maximum(q, self.joint_lower_limits.to(device=q.device, dtype=q.dtype))
        if self.joint_upper_limits is not None:
            q = torch.minimum(q, self.joint_upper_limits.to(device=q.device, dtype=q.dtype))
        return q

    def _normalize_h(self, h: Union[int, torch.Tensor], batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(h, int):
            return torch.full((batch,), h, dtype=torch.long, device=device).clamp(0, 3)
        if h.dim() == 2 and h.shape[-1] == 1:
            h = h.squeeze(-1)
        if h.shape != (batch,):
            raise ValueError("h must be an int or tensor with shape [B] / [B, 1]")
        return h.to(device=device, dtype=torch.long).clamp(0, 3)

    def _validate_inputs(self, current_ee_pose, action_chunk, current_joint_pos, arm_jacobian):
        if current_ee_pose.dim() != 2 or current_ee_pose.shape[-1] < 6:
            raise ValueError("current_ee_pose must have shape [B, >=6]")
        if action_chunk.dim() != 3 or action_chunk.shape[-1] != 6:
            raise ValueError("action_chunk must have shape [B, H, 6]")
        if current_joint_pos.shape != (action_chunk.shape[0], self.num_arm_joints):
            raise ValueError("current_joint_pos must have shape [B, num_arm_joints]")
        if arm_jacobian is not None and arm_jacobian.shape[:2] != (action_chunk.shape[0], 6):
            raise ValueError("arm_jacobian must have shape [B, 6, num_arm_joints]")
