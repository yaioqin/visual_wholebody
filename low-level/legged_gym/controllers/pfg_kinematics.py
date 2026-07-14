"""Self-contained batched kinematics and IK for the PFG reward.

The existing Isaac Gym Jacobian tensor is evaluated at the simulated joint
configuration only. PFG instead performs several *virtual* IK iterations, so
FK and the Jacobian must be recomputed at every virtual q. This module parses
the robot URDF once and implements those operations directly in PyTorch; it
has no dependency beyond PyTorch and the Python standard library.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import torch


@dataclass(frozen=True)
class PFGKinematicsConfig:
    """Numerical parameters corresponding to the paper's IK feasibility test."""

    max_iterations: int = 10
    error_tolerance: float = 1.0e-3
    damping_delta: float = 1.0e-4
    joint_limit_margin: float = 1.0e-4
    max_joint_step: float = 0.25
    pose_error_weights: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )


@dataclass
class _ChainJoint:
    name: str
    joint_type: str
    origin: torch.Tensor
    axis: torch.Tensor
    arm_index: Optional[int]


class PFGKinematics:
    """GPU-batched FK, geometric Jacobian, and damped least-squares IK."""

    def __init__(
        self,
        urdf_path: str,
        end_link_name: str,
        arm_joint_names: Sequence[str],
        joint_lower_limits: torch.Tensor,
        joint_upper_limits: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        config: Optional[PFGKinematicsConfig] = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config or PFGKinematicsConfig()
        self.arm_joint_names = list(arm_joint_names)
        self.num_arm_joints = len(self.arm_joint_names)

        if self.num_arm_joints != 6:
            raise ValueError(
                f"PFG expects six Z1 joints, got {self.num_arm_joints}: "
                f"{self.arm_joint_names}"
            )

        self.chain_joints, self.root_link_name = self._load_chain(
            urdf_path=urdf_path,
            end_link_name=end_link_name,
        )
        self.chain_joint_names = [
            joint.name for joint in self.chain_joints
            if joint.joint_type in ("revolute", "continuous")
        ]

        missing = [
            name for name in self.arm_joint_names
            if name not in self.chain_joint_names
        ]
        extra = [
            name for name in self.chain_joint_names
            if name not in self.arm_joint_names
        ]
        if missing or extra:
            raise RuntimeError(
                "The URDF path from its root to the configured EE link must "
                "contain exactly the six configured arm joints. "
                f"missing={missing}, extra={extra}, "
                f"moving_chain={self.chain_joint_names}"
            )

        lower = joint_lower_limits.to(device=self.device, dtype=self.dtype).flatten()
        upper = joint_upper_limits.to(device=self.device, dtype=self.dtype).flatten()
        if lower.numel() != self.num_arm_joints or upper.numel() != self.num_arm_joints:
            raise ValueError(
                "Joint-limit tensors must each contain six values; got "
                f"{tuple(lower.shape)} and {tuple(upper.shape)}"
            )
        if not bool(torch.all(lower < upper)):
            raise ValueError("Every PFG lower joint limit must be below its upper limit")

        self.lower = lower.unsqueeze(0)
        self.upper = upper.unsqueeze(0)
        self.error_weights = torch.tensor(
            self.config.pose_error_weights,
            dtype=self.dtype,
            device=self.device,
        )
        if self.error_weights.shape != (6,) or not bool(torch.all(self.error_weights > 0.0)):
            raise ValueError("pose_error_weights must contain six positive values")

    # ------------------------------------------------------------------
    # URDF parsing
    # ------------------------------------------------------------------
    def _load_chain(
        self,
        urdf_path: str,
        end_link_name: str,
    ) -> Tuple[List[_ChainJoint], str]:
        with open(urdf_path, "r", encoding="utf-8") as urdf_file:
            urdf_text = urdf_file.read()

        # The branch currently contains this typo in base_static_joint. Isaac
        # Gym accepts it, while strict XML/URDF parsing does not. Repair only
        # the in-memory text used by PFG; the simulation asset is untouched.
        urdf_text = urdf_text.replace(
            'xyz="0.3 0 0.09>>"',
            'xyz="0.3 0 0.09"',
        )
        root = ET.fromstring(urdf_text)

        child_to_joint: Dict[str, ET.Element] = {}
        for joint_element in root.findall("joint"):
            child_element = joint_element.find("child")
            if child_element is None or "link" not in child_element.attrib:
                raise RuntimeError("URDF joint without a child link")
            child_name = child_element.attrib["link"]
            if child_name in child_to_joint:
                raise RuntimeError(f"URDF link has multiple parent joints: {child_name}")
            child_to_joint[child_name] = joint_element

        if end_link_name not in {link.attrib.get("name") for link in root.findall("link")}:
            raise RuntimeError(f"PFG end link is absent from URDF: {end_link_name}")

        reversed_path: List[ET.Element] = []
        current_link = end_link_name
        visited = set()
        while current_link in child_to_joint:
            if current_link in visited:
                raise RuntimeError("Cycle detected while tracing the URDF kinematic chain")
            visited.add(current_link)
            joint_element = child_to_joint[current_link]
            reversed_path.append(joint_element)
            parent_element = joint_element.find("parent")
            if parent_element is None or "link" not in parent_element.attrib:
                raise RuntimeError("URDF joint without a parent link")
            current_link = parent_element.attrib["link"]

        root_link_name = current_link
        arm_name_to_index = {
            name: index for index, name in enumerate(self.arm_joint_names)
        }
        chain: List[_ChainJoint] = []
        for joint_element in reversed(reversed_path):
            name = joint_element.attrib.get("name", "")
            joint_type = joint_element.attrib.get("type", "fixed")
            if joint_type not in ("fixed", "revolute", "continuous"):
                raise NotImplementedError(
                    f"PFG URDF parser does not support joint type '{joint_type}' "
                    f"on chain joint '{name}'"
                )

            origin_element = joint_element.find("origin")
            xyz = self._parse_vector(
                None if origin_element is None else origin_element.attrib.get("xyz"),
                default=(0.0, 0.0, 0.0),
            )
            rpy = self._parse_vector(
                None if origin_element is None else origin_element.attrib.get("rpy"),
                default=(0.0, 0.0, 0.0),
            )
            origin = self._make_transform(xyz, rpy)

            axis_element = joint_element.find("axis")
            axis_values = self._parse_vector(
                None if axis_element is None else axis_element.attrib.get("xyz"),
                default=(1.0, 0.0, 0.0),
            )
            axis = torch.tensor(axis_values, dtype=self.dtype, device=self.device)
            axis_norm = torch.norm(axis, p=2)
            if joint_type != "fixed" and float(axis_norm) <= 1.0e-12:
                raise RuntimeError(f"Moving URDF joint has a zero axis: {name}")
            if joint_type != "fixed":
                axis = axis / axis_norm

            chain.append(
                _ChainJoint(
                    name=name,
                    joint_type=joint_type,
                    origin=origin,
                    axis=axis,
                    arm_index=arm_name_to_index.get(name),
                )
            )

        return chain, root_link_name

    @staticmethod
    def _parse_vector(
        text: Optional[str],
        default: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        if text is None:
            return default
        values = tuple(float(value) for value in text.strip().split())
        if len(values) != 3:
            raise RuntimeError(f"Expected a 3-vector in URDF, got: {text!r}")
        return values  # type: ignore[return-value]

    def _make_transform(
        self,
        xyz: Tuple[float, float, float],
        rpy: Tuple[float, float, float],
    ) -> torch.Tensor:
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # URDF fixed-axis RPY: R = Rz(yaw) Ry(pitch) Rx(roll).
        rotation = torch.tensor(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        transform = torch.eye(4, dtype=self.dtype, device=self.device)
        transform[:3, :3] = rotation
        transform[:3, 3] = torch.tensor(xyz, dtype=self.dtype, device=self.device)
        return transform

    # ------------------------------------------------------------------
    # Rotation utilities
    # ------------------------------------------------------------------
    @staticmethod
    def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        quaternion = quaternion / torch.clamp(
            torch.norm(quaternion, p=2, dim=-1, keepdim=True),
            min=1.0e-8,
        )
        x, y, z, w = quaternion.unbind(dim=-1)
        two = 2.0
        matrix = torch.stack(
            (
                1.0 - two * (y * y + z * z),
                two * (x * y - z * w),
                two * (x * z + y * w),
                two * (x * y + z * w),
                1.0 - two * (x * x + z * z),
                two * (y * z - x * w),
                two * (x * z - y * w),
                two * (y * z + x * w),
                1.0 - two * (x * x + y * y),
            ),
            dim=-1,
        )
        return matrix.reshape(quaternion.shape[:-1] + (3, 3))

    @staticmethod
    def _matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
        """Stable batched rotation-matrix to unit quaternion conversion."""

        m00 = matrix[..., 0, 0]
        m01 = matrix[..., 0, 1]
        m02 = matrix[..., 0, 2]
        m10 = matrix[..., 1, 0]
        m11 = matrix[..., 1, 1]
        m12 = matrix[..., 1, 2]
        m20 = matrix[..., 2, 0]
        m21 = matrix[..., 2, 1]
        m22 = matrix[..., 2, 2]

        q_abs = torch.sqrt(
            torch.clamp(
                torch.stack(
                    (
                        1.0 + m00 + m11 + m22,
                        1.0 + m00 - m11 - m22,
                        1.0 - m00 + m11 - m22,
                        1.0 - m00 - m11 + m22,
                    ),
                    dim=-1,
                ),
                min=0.0,
            )
        )

        candidates = torch.stack(
            (
                torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
                torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
                torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
                torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1),
            ),
            dim=-2,
        )
        candidates = candidates / (
            2.0 * torch.clamp(q_abs, min=0.1).unsqueeze(-1)
        )
        best = torch.argmax(q_abs, dim=-1)
        gather_index = best[..., None, None].expand(
            best.shape + (1, 4)
        )
        quaternion = torch.gather(candidates, -2, gather_index).squeeze(-2)
        quaternion = quaternion / torch.clamp(
            torch.norm(quaternion, p=2, dim=-1, keepdim=True),
            min=1.0e-8,
        )
        return torch.where(
            quaternion[..., :1] < 0.0,
            -quaternion,
            quaternion,
        )

    @classmethod
    def _rotation_error_axis_angle(
        cls,
        target_rotation: torch.Tensor,
        current_rotation: torch.Tensor,
    ) -> torch.Tensor:
        error_rotation = torch.bmm(
            target_rotation,
            current_rotation.transpose(1, 2),
        )
        quaternion = cls._matrix_to_quaternion_wxyz(error_rotation)
        vector = quaternion[..., 1:]
        vector_norm = torch.norm(vector, p=2, dim=-1, keepdim=True)
        half_angle = torch.atan2(vector_norm, quaternion[..., :1])
        scale = torch.where(
            vector_norm > 1.0e-8,
            2.0 * half_angle / torch.clamp(vector_norm, min=1.0e-8),
            torch.full_like(vector_norm, 2.0),
        )
        return vector * scale

    @staticmethod
    def _axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rodrigues rotation for a fixed local axis and batched angles."""

        batch_size = angle.shape[0]
        axis = axis.view(1, 3).expand(batch_size, -1)
        x, y, z = axis.unbind(dim=-1)
        zeros = torch.zeros_like(x)
        skew = torch.stack(
            (
                zeros, -z, y,
                z, zeros, -x,
                -y, x, zeros,
            ),
            dim=-1,
        ).reshape(batch_size, 3, 3)
        outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
        identity = torch.eye(
            3,
            dtype=angle.dtype,
            device=angle.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        cosine = torch.cos(angle).view(-1, 1, 1)
        sine = torch.sin(angle).view(-1, 1, 1)
        return cosine * identity + (1.0 - cosine) * outer + sine * skew

    # ------------------------------------------------------------------
    # Batched FK and Jacobian
    # ------------------------------------------------------------------
    def _forward_and_jacobian(
        self,
        arm_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = arm_q.shape[0]
        transform = torch.eye(
            4,
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(0).repeat(batch_size, 1, 1)

        joint_positions: List[Optional[torch.Tensor]] = [None] * self.num_arm_joints
        joint_axes: List[Optional[torch.Tensor]] = [None] * self.num_arm_joints

        for joint in self.chain_joints:
            origin = joint.origin.unsqueeze(0).expand(batch_size, -1, -1)
            transform = torch.bmm(transform, origin)

            if joint.joint_type in ("revolute", "continuous"):
                if joint.arm_index is None:
                    raise RuntimeError(
                        f"Moving chain joint is not configured as an arm joint: {joint.name}"
                    )
                rotation_world = transform[:, :3, :3]
                axis_world = torch.bmm(
                    rotation_world,
                    joint.axis.view(1, 3, 1).expand(batch_size, -1, -1),
                ).squeeze(-1)
                joint_positions[joint.arm_index] = transform[:, :3, 3]
                joint_axes[joint.arm_index] = axis_world

                local_rotation = self._axis_angle_rotation(
                    joint.axis,
                    arm_q[:, joint.arm_index],
                )
                local_transform = torch.eye(
                    4,
                    dtype=self.dtype,
                    device=self.device,
                ).unsqueeze(0).repeat(batch_size, 1, 1)
                local_transform[:, :3, :3] = local_rotation
                transform = torch.bmm(transform, local_transform)

        if any(value is None for value in joint_positions) or any(
            value is None for value in joint_axes
        ):
            raise RuntimeError("Failed to construct every arm Jacobian column")

        end_position = transform[:, :3, 3]
        positions = torch.stack(
            [value for value in joint_positions if value is not None],
            dim=1,
        )
        axes = torch.stack(
            [value for value in joint_axes if value is not None],
            dim=1,
        )
        linear = torch.cross(
            axes,
            end_position.unsqueeze(1) - positions,
            dim=-1,
        ).transpose(1, 2)
        angular = axes.transpose(1, 2)
        jacobian = torch.cat((linear, angular), dim=1)
        return transform, jacobian

    def _pose_error(
        self,
        arm_q: torch.Tensor,
        target_position: torch.Tensor,
        target_rotation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_transform, jacobian = self._forward_and_jacobian(arm_q)
        position_error = target_position - current_transform[:, :3, 3]
        rotation_error = self._rotation_error_axis_angle(
            target_rotation,
            current_transform[:, :3, :3],
        )
        pose_error = torch.cat((position_error, rotation_error), dim=-1)
        return pose_error, position_error, rotation_error, jacobian

    # ------------------------------------------------------------------
    # Paper-style feasibility solve
    # ------------------------------------------------------------------
    @torch.no_grad()
    def solve(
        self,
        target_position_body: torch.Tensor,
        target_quaternion_body_xyzw: torch.Tensor,
        current_arm_q: torch.Tensor,
        previous_solution: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return IK feasibility and the safe paper-style desired q.

        On failure, the returned solution is exactly ``current_arm_q``. This
        matters because the reward must be masked to zero for infeasible torso
        states, not shaped from an unconverged numerical iterate.
        """

        target_position = target_position_body.to(self.device, self.dtype)
        target_quaternion = target_quaternion_body_xyzw.to(self.device, self.dtype)
        target_quaternion = target_quaternion / torch.clamp(
            torch.norm(target_quaternion, p=2, dim=-1, keepdim=True),
            min=1.0e-8,
        )
        target_rotation = self.quaternion_xyzw_to_matrix(target_quaternion)
        current_q = current_arm_q.to(self.device, self.dtype)

        if previous_solution is None:
            q = current_q.clone()
        else:
            seed = previous_solution.to(self.device, self.dtype)
            valid_seed = torch.isfinite(seed).all(dim=-1, keepdim=True)
            q = torch.where(valid_seed, seed, current_q).clone()
        q = torch.maximum(torch.minimum(q, self.upper), self.lower)

        batch_size = q.shape[0]
        identity = torch.eye(
            self.num_arm_joints,
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        weights = self.error_weights.view(1, 6)

        for _ in range(self.config.max_iterations):
            pose_error, _, _, jacobian = self._pose_error(
                q,
                target_position,
                target_rotation,
            )
            quadratic_error = torch.sum(weights * pose_error.square(), dim=-1)
            energy = 0.5 * quadratic_error
            converged = energy <= self.config.error_tolerance

            # Paper Algorithm 1: a joint at its limit contributes no IK column.
            at_limit = (
                (q <= self.lower + self.config.joint_limit_margin)
                | (q >= self.upper - self.config.joint_limit_margin)
            )
            jacobian = torch.where(
                at_limit.unsqueeze(1),
                torch.zeros_like(jacobian),
                jacobian,
            )

            weighted_jacobian = jacobian * weights.unsqueeze(-1)
            jacobian_transpose = jacobian.transpose(1, 2)
            system_matrix = torch.bmm(jacobian_transpose, weighted_jacobian)
            adaptive_damping = 0.5 * (
                quadratic_error + self.config.damping_delta
            )
            system_matrix = system_matrix + adaptive_damping[:, None, None] * identity
            gradient = torch.bmm(
                jacobian_transpose,
                (weights * pose_error).unsqueeze(-1),
            )
            delta_q = torch.linalg.solve(system_matrix, gradient).squeeze(-1)
            delta_q = torch.clamp(
                delta_q,
                min=-self.config.max_joint_step,
                max=self.config.max_joint_step,
            )
            candidate = torch.maximum(
                torch.minimum(q + delta_q, self.upper),
                self.lower,
            )
            q = torch.where(converged.unsqueeze(-1), q, candidate)

        pose_error, position_error, rotation_error, _ = self._pose_error(
            q,
            target_position,
            target_rotation,
        )
        quadratic_error = torch.sum(weights * pose_error.square(), dim=-1)
        energy = 0.5 * quadratic_error
        finite = torch.isfinite(q).all(dim=-1) & torch.isfinite(energy)
        success = (energy <= self.config.error_tolerance) & finite

        safe_solution = torch.where(success.unsqueeze(-1), q, current_q)
        position_error_norm = torch.norm(position_error, p=2, dim=-1)
        rotation_error_norm = torch.norm(rotation_error, p=2, dim=-1)
        return (
            safe_solution,
            success,
            energy,
            position_error_norm,
            rotation_error_norm,
        )
