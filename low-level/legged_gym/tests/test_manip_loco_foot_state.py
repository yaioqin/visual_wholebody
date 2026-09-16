"""Exercise live foot-state rewards without loading Isaac Gym native bindings."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


class ManipLocoFootStateTest(unittest.TestCase):
    def test_contact_change_uses_previous_frame_and_masks_recent_resets(self):
        feet_jerk = load_method(
            "envs/rewards/maniploco_rewards.py", "ManipLoco_rewards",
            "_reward_feet_jerk", {"torch": torch},
        )
        env = SimpleNamespace(
            num_envs=3, device="cpu", force_sensor_tensor=torch.zeros(3, 4, 6),
            episode_length_buf=torch.tensor([100, 49, 50]),
        )
        reward_container = SimpleNamespace(env=env)
        initial, _ = feet_jerk(reward_container)
        torch.testing.assert_close(initial, torch.zeros(3))

        # Gym updates its tensor in place; the previous forces must be a snapshot.
        env.force_sensor_tensor[0, 0, :2] = torch.tensor([3.0, 4.0])
        env.force_sensor_tensor[0, 1, 3] = 12.0
        env.force_sensor_tensor[1, 0, 0] = 99.0
        env.force_sensor_tensor[2, 0, :2] = torch.tensor([6.0, 8.0])
        changed, metric = feet_jerk(reward_container)
        torch.testing.assert_close(changed, torch.tensor([17.0, 0.0, 10.0]))
        torch.testing.assert_close(metric, changed)

        # Masked frames still update history, avoiding a spike when the mask ends.
        env.episode_length_buf[1] = 50
        env.force_sensor_tensor[1, 0, 0] += 2.0
        next_frame, _ = feet_jerk(reward_container)
        torch.testing.assert_close(next_frame, torch.tensor([0.0, 2.0, 0.0]))

    def test_physics_updates_reach_contact_velocity_reward_each_step(self):
        post_step = load_method("envs/manip_loco/manip_loco.py", "ManipLoco", "post_physics_step", {
            "torch": torch,
            "quat_rotate_inverse": lambda _quat, vector: vector,
            "euler_from_quat": lambda quat: tuple(torch.zeros(quat.shape[0]) for _ in range(3)),
            "quat_from_euler_xyz": lambda _roll, _pitch, yaw: torch.zeros(yaw.shape[0], 4),
        })
        velocity_reward = load_method(
            "envs/rewards/maniploco_rewards.py", "ManipLoco_rewards",
            "_reward_tracking_contacts_shaped_vel", {"torch": torch},
        )
        env = SimpleNamespace(
            gym=Mock(), sim=object(), num_envs=2, viewer=None, device="cpu",
            cfg=SimpleNamespace(env=SimpleNamespace(observe_gait_commands=True),
                                rewards=SimpleNamespace(gait_vel_sigma=0.5)),
            rigid_body_state=torch.zeros(2, 7, 13), feet_indices=torch.tensor([1, 3, 4, 6]),
            foot_positions=torch.zeros(2, 4, 3), foot_velocities=torch.zeros(2, 4, 3),
            desired_contact_states=torch.ones(2, 4), episode_length_buf=torch.zeros(2),
            common_step_counter=0, root_states=torch.zeros(2, 13),
            base_quat=torch.zeros(2, 4), base_lin_vel=torch.zeros(2, 3),
            base_ang_vel=torch.zeros(2, 3), base_yaw_euler=torch.zeros(2, 3),
            base_yaw_quat=torch.zeros(2, 4), projected_gravity=torch.zeros(2, 3),
            gravity_vec=torch.zeros(2, 3), contact_forces=torch.zeros(2, 7, 3),
            last_contacts=torch.zeros(2, 4, dtype=torch.bool),
            force_sensor_tensor=torch.zeros(2, 4, 6),
            reset_buf=torch.zeros(2, dtype=torch.bool),
            last_actions=torch.zeros(2, 18), actions=torch.zeros(2, 18),
            last_dof_vel=torch.zeros(2, 19), dof_vel=torch.zeros(2, 19),
            last_root_vel=torch.zeros(2, 6), last_torques=torch.zeros(2, 19),
            torques=torch.zeros(2, 19),
            _update_curr_ee_goal=Mock(), check_termination=Mock(),
            reset_idx=Mock(), compute_observations=Mock(),
        )
        rewards, callback_positions = [], []
        env.compute_reward = lambda: rewards.append(velocity_reward(SimpleNamespace(env=env))[0])
        env._post_physics_step_callback = lambda: callback_positions.append(env.foot_positions.clone())

        moving = torch.zeros_like(env.rigid_body_state)
        moving[:, :, :3] = torch.arange(42).reshape(2, 7, 3)
        moving[:, 0, 7:10] = 999.0  # Non-foot bodies must not contribute.
        moving[0, 1, 7] = 1.0
        moving[1, 6, 8] = 2.0
        stopped = torch.zeros_like(moving)
        stopped[:, :, :3] = moving[:, :, :3] + 100.0
        for frame in (moving, stopped):
            env.gym.refresh_rigid_body_state_tensor.side_effect = lambda _sim, frame=frame: (
                env.rigid_body_state.copy_(frame)
            )
            post_step(env)
            torch.testing.assert_close(env.foot_velocities, frame[:, env.feet_indices, 7:10])
            torch.testing.assert_close(callback_positions[-1], frame[:, env.feet_indices, :3])

        torch.testing.assert_close(rewards[0], -(1 - torch.exp(-torch.tensor([1.0, 4.0]) / 0.5)) / 4)
        torch.testing.assert_close(rewards[1], torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
