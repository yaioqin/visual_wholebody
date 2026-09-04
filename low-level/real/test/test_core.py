#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

REAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL_ROOT / "src"))

from b2z1_real.kinematics import Z1Kinematics
from b2z1_real.policy import (
    NUM_PROPRIO,
    POLICY_INPUT_DIM,
    ObservationBuilder,
    cartesian_to_sphere,
    rpy_to_matrix,
    sphere_to_cartesian,
)
from b2z1_real.safety import SafetyGate


def load_config():
    with (REAL_ROOT / "config" / "b2z1_real.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["control_dt"] = 1.0 / config["control_rate"]
    return config


class ObservationTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.builder = ObservationBuilder(self.config)

    def test_dimensions_and_level_goal_transform(self):
        current, goal = self.builder.build_current(
            orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            angular_velocity=[0.1, 0.2, 0.3],
            leg_position=self.config["robot"]["default_leg_position"],
            leg_velocity=np.zeros(12),
            arm_position=np.zeros(6),
            arm_velocity=np.zeros(6),
            last_leg_action=np.zeros(12),
            foot_force=[0.0, 10.0, 0.0, 10.0],
            command=[0.5, 0.0, 0.0],
            goal_vector=self.config["goal"]["default_vector"],
        )
        self.assertEqual(current.shape, (NUM_PROPRIO,))
        np.testing.assert_allclose(
            goal.arm_position,
            [0.3197597663, 0.0, 0.2028317162],
            atol=1e-9,
        )
        self.assertAlmostEqual(current[-5], 0.04, places=7)
        self.assertEqual(self.builder.policy_input(current).shape, (POLICY_INPUT_DIM,))

    def test_standing_command_resets_gait(self):
        self.builder.gait_phase = 0.7
        clock = self.builder._step_gait(np.zeros(3))
        self.assertEqual(self.builder.gait_phase, 0.0)
        np.testing.assert_allclose(clock, np.zeros(4), atol=1e-12)

    def test_tilted_goal_transform_rotates_mount_offset(self):
        body_rotation = rpy_to_matrix(0.20, -0.10, 0.40)
        yaw_rotation = rpy_to_matrix(0.0, 0.0, 0.40)
        vector = np.asarray(self.config["goal"]["default_vector"])
        center = np.asarray(self.config["goal"]["sphere_center_offset"])
        mount = np.asarray(self.config["robot"]["arm_mount_offset"])
        base_height = float(self.config["goal"]["assumed_base_height"])
        expected = body_rotation.T @ yaw_rotation @ (
            center + vector - mount - np.asarray([0.0, 0.0, base_height])
        )
        transformed = self.builder.transform_goal(vector, body_rotation)
        np.testing.assert_allclose(transformed.arm_position, expected, atol=1e-12)

    def test_policy_input_uses_training_observation_clip(self):
        current = np.full(NUM_PROPRIO, 200.0, dtype=np.float32)
        policy_input = self.builder.policy_input(current)
        np.testing.assert_array_equal(policy_input, np.full(POLICY_INPUT_DIM, 100.0))

    def test_goal_coordinate_round_trip(self):
        vector = np.asarray([0.55, -0.12, 0.17])
        np.testing.assert_allclose(
            sphere_to_cartesian(cartesian_to_sphere(vector)), vector, atol=1e-12
        )


class KinematicsTest(unittest.TestCase):
    def test_linear_jacobian_matches_finite_difference(self):
        model = Z1Kinematics()
        q = np.asarray([0.2, 1.1, -1.0, -0.2, 0.1, 0.3])
        position, _, jacobian = model.forward_and_jacobian(q)
        epsilon = 1e-7
        numerical = np.zeros((3, 6))
        for index in range(6):
            perturbed = q.copy()
            perturbed[index] += epsilon
            next_position, _, _ = model.forward_and_jacobian(perturbed)
            numerical[:, index] = (next_position - position) / epsilon
        np.testing.assert_allclose(jacobian[:3], numerical, atol=2e-7)

    def test_ik_step_reduces_small_position_error(self):
        model = Z1Kinematics()
        q = np.asarray([0.2, 1.1, -1.0, -0.2, 0.1, 0.3])
        position, orientation, _ = model.forward_and_jacobian(q)
        target = position + np.asarray([0.002, -0.001, 0.001])
        next_q, _ = model.ik_step(q, target, orientation, max_delta=0.08)
        next_position, _, _ = model.forward_and_jacobian(next_q)
        self.assertLess(np.linalg.norm(target - next_position), np.linalg.norm(target - position))


class SafetyTest(unittest.TestCase):
    def test_valid_nominal_state(self):
        config = load_config()
        gate = SafetyGate(config["safety"])
        reason = gate.check(
            now=10.0,
            b2_received_at=9.95,
            z1_received_at=9.95,
            rpy=[0.0, 0.0, 0.0],
            leg_position=config["robot"]["default_leg_position"],
            arm_position=np.zeros(6),
            leg_temperature=np.full(12, 30),
            arm_temperature=np.full(7, 30),
            b2_valid=True,
            z1_valid=True,
        )
        self.assertIsNone(reason)

    def test_timeout_is_rejected(self):
        gate = SafetyGate(load_config()["safety"])
        reason = gate.check(
            now=10.0,
            b2_received_at=9.0,
            z1_received_at=10.0,
            rpy=[0.0, 0.0, 0.0],
            leg_position=np.zeros(12),
            arm_position=np.zeros(6),
            leg_temperature=[],
            arm_temperature=[],
            b2_valid=True,
            z1_valid=True,
        )
        self.assertIn("timeout", reason)


if __name__ == "__main__":
    unittest.main()
