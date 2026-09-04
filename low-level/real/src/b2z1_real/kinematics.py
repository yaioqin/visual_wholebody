"""Pure NumPy forward kinematics and damped least-squares IK for Unitree Z1."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return xyzw quaternion with a numerically stable branch selection."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        result = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            result = np.asarray(
                [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            result = np.asarray(
                [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            result = np.asarray(
                [(matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                 (matrix[1, 0] - matrix[0, 1]) / scale]
            )
    return result / np.linalg.norm(result)


class Z1Kinematics:
    """Kinematic chain copied from the B2-Z1 training URDF.

    The output frame is the ``gripperMover`` link origin used by Isaac Gym.
    """

    JOINT_ORIGINS = np.asarray(
        [
            [0.0, 0.0, 0.0585],
            [0.0, 0.0, 0.0450],
            [-0.3500, 0.0, 0.0],
            [0.2180, 0.0, 0.0570],
            [0.0700, 0.0, 0.0],
            [0.0492, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    JOINT_AXES = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    END_OFFSET = np.asarray([0.1000, 0.0, 0.0], dtype=np.float64)

    def forward_and_jacobian(self, joint_position: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(joint_position, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("joint_position must be a finite 6-vector")

        rotation = np.eye(3, dtype=np.float64)
        position = np.zeros(3, dtype=np.float64)
        origins = []
        axes = []
        for origin, axis, angle in zip(self.JOINT_ORIGINS, self.JOINT_AXES, q):
            position = position + rotation @ origin
            origins.append(position.copy())
            axes.append(rotation @ axis)
            rotation = rotation @ _rotation(axis, float(angle))

        end_position = position + rotation @ self.END_OFFSET
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for index, (origin, axis) in enumerate(zip(origins, axes)):
            jacobian[:3, index] = np.cross(axis, end_position - origin)
            jacobian[3:, index] = axis
        return end_position, rotation, jacobian

    @staticmethod
    def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        error_rotation = np.asarray(target, dtype=np.float64) @ np.asarray(current, dtype=np.float64).T
        quaternion = _quaternion_from_matrix(error_rotation)
        sign = 1.0 if quaternion[3] >= 0.0 else -1.0
        return quaternion[:3] * sign

    def ik_step(
        self,
        joint_position: Sequence[float],
        target_position: Sequence[float],
        target_orientation: np.ndarray,
        damping: float = 0.05,
        max_delta: float = 0.08,
    ) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(joint_position, dtype=np.float64)
        target_position = np.asarray(target_position, dtype=np.float64)
        target_orientation = np.asarray(target_orientation, dtype=np.float64)
        if target_position.shape != (3,) or target_orientation.shape != (3, 3):
            raise ValueError("invalid IK target shape")
        current_position, current_orientation, jacobian = self.forward_and_jacobian(q)
        error = np.concatenate(
            (target_position - current_position, self.orientation_error(target_orientation, current_orientation))
        )
        regularizer = np.eye(6, dtype=np.float64) * float(damping) ** 2
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, error)
        largest = float(np.max(np.abs(delta)))
        if largest > max_delta:
            delta *= float(max_delta) / largest
        target_q = q + delta
        if not np.all(np.isfinite(target_q)):
            raise RuntimeError("IK returned NaN or infinity")
        return target_q, error
