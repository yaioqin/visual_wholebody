# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# This task config is intentionally a thin wrapper around the original 3D
# command base config. Keep experiment changes here limited to the arm-to-base
# message observation so the trained policy differs from 3D only by m_a2b.

from .b1z1_config_3D import (
    B1Z1RoughCfg as B1Z1RoughCfg3D,
    B1Z1RoughCfgPPO as B1Z1RoughCfgPPO3D,
)


class B1Z1RoughCfg(B1Z1RoughCfg3D):
    class env(B1Z1RoughCfg3D.env):
        base_num_proprio = B1Z1RoughCfg3D.env.num_proprio
        num_proprio = base_num_proprio + 5
        num_observations = (
            num_proprio * (B1Z1RoughCfg3D.env.history_len + 1)
            + B1Z1RoughCfg3D.env.num_priv
        )

    class multi_agent:
        use_arm_base_message = True

        use_arm_delta_action = False
        allow_arm_policy_action = False
        arm_chunk_horizon = 1

        max_ee_pos_delta = 0.05
        max_ee_rot_delta = 0.2
        max_joint_delta = 0.25

        use_assist_reward = False
        debug_print_kinematics_names = False

        arm_dof_names = [
            "z1_waist",
            "z1_shoulder",
            "z1_elbow",
            "z1_wrist_angle",
            "z1_forearm_roll",
            "z1_wrist_rotate",
        ]
        ee_body_name = "ee_gripper_link"



    # >>> PFG REWARD PATCH (3d_m_a2b) >>>
    class rewards(B1Z1RoughCfg3D.rewards):
        class pfg:
            enabled = True # 可关闭该奖励函数计算

            # Algorithm 1 in the paper uses at most ten virtual IK steps.
            max_iterations = 10
            error_tolerance = 1.0e-3
            damping_delta = 1.0e-4
            joint_limit_margin = 1.0e-4
            max_joint_step = 0.25

            # W_e. The paper does not publish a non-identity matrix, so the
            # reproducible default is I_6; change this only as an ablation.
            pose_error_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

            # Compute every policy step. Raise to 2-5 only if profiling shows
            # that batched IK is the training bottleneck.
            update_interval = 1

            # max(exp(-sum(abs(q_ideal-q))), 0.2), only when IK succeeds.
            minimum_feasible_reward = 0.2
            joint_error_scale = 1.0

        class manipulability:
            enabled = True
            # 防止奇异值为 0 时出现 log(0)
            eps = 1.0e-6
            # 低于默认姿态 manipulability 的 20% 时开始惩罚
            soft_ratio = 0.20
            # 低于默认姿态 manipulability 的 10% 时开始强制惩罚
            critical_ratio = 0.10
            # 限制 exp() 的最小指数，防止数值下溢
            min_log_ratio = -20.0

        class scales(B1Z1RoughCfg3D.rewards.scales):
            # PFG is a torso/base feasibility reward. Keep it in base scales,
            # especially while allow_arm_policy_action=False.
            pfg_feasible = 0.16  # 总奖励中pfg奖励函数的权值
            low_manipulability = -0.15
    # <<< PFG REWARD PATCH (3d_m_a2b) <<<

class B1Z1RoughCfgPPO(B1Z1RoughCfgPPO3D):
    pass
