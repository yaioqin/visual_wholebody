"""Source-level contract checks for the B1+Z1 DWBC implementation."""

from types import SimpleNamespace

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import torch

from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.rewards.maniploco_rewards import ManipLoco_rewards
from legged_gym.utils.helpers import class_to_dict


def check_config():
    cfg = B1Z1RoughCfg()
    train_cfg = B1Z1RoughCfgPPO()

    assert cfg.env.num_actions == 18
    assert cfg.env.num_proprio == 72
    assert cfg.env.num_priv == 24
    assert cfg.env.num_observations == 816
    assert len(cfg.control.action_scale) == 18
    assert cfg.control.stiffness == {"joint": 50.0, "z1": 5.0}
    assert cfg.control.damping == {"joint": 1.0, "z1": 0.5}
    assert class_to_dict(cfg.rewards.scales) == {
        "energy_square": -6.0e-5,
        "foot_contacts_z": -1.0e-4,
        "hip_action_l2": -0.01,
        "survive": 0.2,
        "tracking_ang_vel_yaw_exp": 0.15,
        "tracking_lin_vel_x_l1": 0.5,
    }
    assert class_to_dict(cfg.rewards.arm_scales) == {
        "arm_energy_abs_sum": -0.004,
        "tracking_ee_sphere": 0.55,
    }
    assert train_cfg.algorithm.mixing_schedule == [1.0, 0, 3000]
    assert train_cfg.algorithm.priv_reg_coef_schedual == [0.0, 0.1, 3000, 7000]
    assert train_cfg.policy.priv_encoder_dims == [64, 20]
    assert train_cfg.runner.num_steps_per_env == 40


def check_whole_body_pd():
    batch = 3
    fake_env = SimpleNamespace(
        motor_strength=torch.ones(batch, 18),
        action_scale=torch.ones(18),
        p_gains=torch.cat([torch.full((12,), 50.0), torch.full((6,), 5.0)]),
        d_gains=torch.cat([torch.full((12,), 1.0), torch.full((6,), 0.5)]),
        default_dof_pos_wo_gripper=torch.zeros(18),
        dof_pos_wo_gripper=torch.zeros(batch, 18),
        dof_pos_wo_gripper_wrapped=torch.zeros(batch, 18),
        dof_vel_wo_gripper=torch.zeros(batch, 18),
        gripper_torques_zero=torch.zeros(batch, 1),
        torque_limits=torch.full((19,), 1000.0),
    )
    actions = torch.ones(batch, 18) * 0.1
    torques = ManipLoco._compute_torques(fake_env, actions)

    assert torques.shape == (batch, 19)
    assert torch.allclose(torques[:, :12], torch.full((batch, 12), 5.0))
    assert torch.allclose(torques[:, 12:18], torch.full((batch, 6), 0.5))
    assert torch.all(torques[:, 18] == 0.0)


def check_reference_reward_formulas():
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, -0.5]])
    base_lin_vel = torch.tensor([[0.2, 0.0, 0.0], [0.5, 0.0, 0.0]])
    base_ang_vel = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, -0.1]])
    fake_env = SimpleNamespace(
        commands=commands,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        cfg=SimpleNamespace(rewards=SimpleNamespace(tracking_sigma=1.0)),
    )
    rewards = ManipLoco_rewards(fake_env)

    lin_reward, lin_error = rewards._reward_tracking_lin_vel_x_l1()
    assert torch.allclose(lin_error, torch.tensor([0.2, 0.3]))
    assert torch.allclose(lin_reward, torch.tensor([-0.2, 0.5]))

    yaw_reward, yaw_error = rewards._reward_tracking_ang_vel_yaw_exp()
    assert torch.allclose(yaw_error, torch.tensor([0.2, 0.4]))
    assert torch.allclose(yaw_reward, torch.exp(-yaw_error))


def main():
    check_config()
    check_whole_body_pd()
    check_reference_reward_formulas()
    print("DWBC config and whole-body PD contract passed")


if __name__ == "__main__":
    main()
