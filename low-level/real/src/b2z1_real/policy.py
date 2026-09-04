"""Exact deployment observation path for ``b2z1 --observe_gait_commands``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


NUM_PROPRIO = 71
HISTORY_LEN = 10
POLICY_INPUT_DIM = NUM_PROPRIO * (HISTORY_LEN + 1)
NUM_ACTIONS = 18


def _array(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def cartesian_to_sphere(vector: Sequence[float]) -> np.ndarray:
    values = _array(vector, 3, "Cartesian goal")
    radius = float(np.linalg.norm(values))
    horizontal = float(np.linalg.norm(values[:2]))
    return np.asarray(
        [radius, np.arctan2(values[2], horizontal), np.arctan2(values[1], values[0])],
        dtype=np.float64,
    )


def sphere_to_cartesian(sphere: Sequence[float]) -> np.ndarray:
    radius, pitch, yaw = _array(sphere, 3, "spherical goal")
    horizontal = radius * np.cos(pitch)
    return np.asarray(
        [horizontal * np.cos(yaw), horizontal * np.sin(yaw), radius * np.sin(pitch)],
        dtype=np.float64,
    )


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = _array(quaternion, 4, "orientation")
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-8:
        raise ValueError("orientation quaternion has zero norm")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Return fixed-axis roll, pitch, yaw, matching Isaac Gym's xyz Euler helper."""
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-7:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class GoalTransform:
    arm_position: np.ndarray
    arm_orientation: np.ndarray
    sphere: np.ndarray


class ObservationBuilder:
    """Build the 71-D current observation and the 10-frame history input.

    All leg arrays use the Unitree low-level order:
    ``FR, FL, RR, RL``, with ``hip, thigh, calf`` inside each leg.  That is
    also the policy order after ``ManipLoco._reindex_all``.
    """

    def __init__(self, config: Mapping[str, object]):
        robot = config["robot"]
        observation = config["observation"]
        goal = config["goal"]

        self.dt = float(config.get("control_dt", 0.02))
        self.default_leg = _array(robot["default_leg_position"], 12, "default_leg_position")
        self.default_arm = _array(robot["default_arm_position"], 6, "default_arm_position")
        self.arm_mount_offset = _array(robot["arm_mount_offset"], 3, "arm_mount_offset")
        self.center_offset = _array(goal["sphere_center_offset"], 3, "sphere_center_offset")
        self.base_height = float(goal["assumed_base_height"])
        self.arm_induced_pitch = float(goal.get("arm_induced_pitch", 0.38))
        self.dof_pos_scale = float(observation.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(observation.get("dof_vel_scale", 0.05))
        self.ang_vel_scale = float(observation.get("ang_vel_scale", 1.0))
        self.contact_threshold = float(observation.get("contact_threshold", 1.5))
        self.observation_clip = float(observation.get("clip", 100.0))
        self.gait_frequency = float(observation.get("gait_frequency", 2.0))
        self.walking_linear_threshold = float(observation.get("walking_linear_threshold", 0.2))
        self.walking_yaw_threshold = float(observation.get("walking_yaw_threshold", 0.5))

        self._history: Optional[np.ndarray] = None
        self.gait_phase = 0.0

    def reset(self) -> None:
        self._history = None
        self.gait_phase = 0.0

    def transform_goal(
        self,
        goal_vector: Sequence[float],
        body_rotation_world: np.ndarray,
        base_height: Optional[float] = None,
    ) -> GoalTransform:
        """Transform the training spherical-center vector into the Z1 mount frame.

        ``goal_vector`` is the Cartesian form of ``curr_ee_goal_sphere`` and is
        expressed in the gravity-aligned base-yaw frame.  This reproduces
        ``_get_ee_goal_spherical_center`` and ``compute_observations`` without
        requiring global x/y odometry.
        """
        vector = _array(goal_vector, 3, "goal_vector")
        sphere = cartesian_to_sphere(vector)

        rotation_world_body = np.asarray(body_rotation_world, dtype=np.float64)
        if rotation_world_body.shape != (3, 3):
            raise ValueError("body_rotation_world must have shape (3, 3)")
        yaw = matrix_to_rpy(rotation_world_body)[2]
        rotation_world_yaw = rpy_to_matrix(0.0, 0.0, yaw)
        rotation_body_yaw = rotation_world_body.T @ rotation_world_yaw

        height = self.base_height if base_height is None else float(base_height)
        target_from_root_yaw = self.center_offset + vector
        target_from_root_yaw[2] -= height
        # Both the target and the Z1 mount are defined in the gravity-aligned
        # base-yaw frame in ManipLoco.  Apply body tilt only after subtracting
        # the mount offset; subtracting it afterwards is wrong when roll/pitch
        # are non-zero.
        target_from_arm_body = rotation_body_yaw @ (target_from_root_yaw - self.arm_mount_offset)

        goal_rotation_yaw = rpy_to_matrix(
            np.pi / 2.0,
            -sphere[1] + self.arm_induced_pitch,
            sphere[2],
        )
        goal_rotation_arm = rotation_body_yaw @ goal_rotation_yaw
        return GoalTransform(target_from_arm_body, goal_rotation_arm, sphere)

    def _step_gait(self, command: np.ndarray) -> np.ndarray:
        walking = (
            abs(command[0]) > self.walking_linear_threshold
            or abs(command[1]) > self.walking_linear_threshold
            or abs(command[2]) > self.walking_yaw_threshold
        )
        if walking:
            self.gait_phase = (self.gait_phase + self.dt * self.gait_frequency) % 1.0
        else:
            self.gait_phase = 0.0
        phases = np.asarray(
            [self.gait_phase + 0.5, self.gait_phase, self.gait_phase, self.gait_phase + 0.5],
            dtype=np.float64,
        )
        return np.sin(2.0 * np.pi * phases)

    def build_current(
        self,
        orientation_xyzw: Sequence[float],
        angular_velocity: Sequence[float],
        leg_position: Sequence[float],
        leg_velocity: Sequence[float],
        arm_position: Sequence[float],
        arm_velocity: Sequence[float],
        last_leg_action: Sequence[float],
        foot_force: Sequence[float],
        command: Sequence[float],
        goal_vector: Sequence[float],
        base_height: Optional[float] = None,
    ) -> Tuple[np.ndarray, GoalTransform]:
        rotation = quaternion_xyzw_to_matrix(orientation_xyzw)
        rpy = matrix_to_rpy(rotation)
        angular_velocity = _array(angular_velocity, 3, "angular_velocity")
        leg_position = _array(leg_position, 12, "leg_position")
        leg_velocity = _array(leg_velocity, 12, "leg_velocity")
        arm_position = _array(arm_position, 6, "arm_position")
        arm_velocity = _array(arm_velocity, 6, "arm_velocity")
        last_leg_action = _array(last_leg_action, 12, "last_leg_action")
        foot_force = _array(foot_force, 4, "foot_force")
        command = _array(command, 3, "command")
        goal = self.transform_goal(goal_vector, rotation, base_height)
        clock = self._step_gait(command)

        joint_position = np.concatenate((leg_position - self.default_leg, arm_position - self.default_arm))
        joint_velocity = np.concatenate((leg_velocity, arm_velocity))
        current = np.concatenate(
            (
                rpy[:2],
                angular_velocity * self.ang_vel_scale,
                joint_position * self.dof_pos_scale,
                joint_velocity * self.dof_vel_scale,
                last_leg_action,
                (foot_force > self.contact_threshold).astype(np.float64),
                command,
                goal.arm_position,
                np.zeros(3, dtype=np.float64),
                np.asarray([self.gait_phase], dtype=np.float64),
                clock,
            )
        ).astype(np.float32)
        if current.shape != (NUM_PROPRIO,):
            raise RuntimeError(f"internal observation shape error: {current.shape}")
        return current, goal

    def policy_input(self, current: Sequence[float]) -> np.ndarray:
        current = np.asarray(current, dtype=np.float32)
        if current.shape != (NUM_PROPRIO,):
            raise ValueError(f"current observation must have shape ({NUM_PROPRIO},)")
        if self._history is None:
            self._history = np.repeat(current[None, :], HISTORY_LEN, axis=0)
        result = np.concatenate((current, self._history.reshape(-1))).astype(np.float32)
        result = np.clip(result, -self.observation_clip, self.observation_clip)
        self._history = np.concatenate((self._history[1:], current[None, :]), axis=0)
        if result.shape != (POLICY_INPUT_DIM,):
            raise RuntimeError(f"internal policy input shape error: {result.shape}")
        return result


class PolicyRunner:
    """Small TorchScript runner with strict shape and finite-value checks."""

    def __init__(self, path: str, device: str = "cpu"):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.module = torch.jit.load(path, map_location=self.device)
        self.module.eval()
        with torch.inference_mode():
            output = self.module(torch.zeros(1, POLICY_INPUT_DIM, device=self.device))
        if tuple(output.shape) != (1, NUM_ACTIONS):
            raise ValueError(
                f"policy must map (1, {POLICY_INPUT_DIM}) to (1, {NUM_ACTIONS}), got {tuple(output.shape)}"
            )

    def infer(self, policy_input: Sequence[float]) -> np.ndarray:
        values = np.asarray(policy_input, dtype=np.float32)
        if values.shape != (POLICY_INPUT_DIM,):
            raise ValueError(f"policy input must have shape ({POLICY_INPUT_DIM},)")
        tensor = self._torch.from_numpy(values).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            output = self.module(tensor).squeeze(0).cpu().numpy()
        if output.shape != (NUM_ACTIONS,) or not np.all(np.isfinite(output)):
            raise RuntimeError("policy returned an invalid action")
        return output.astype(np.float64)
