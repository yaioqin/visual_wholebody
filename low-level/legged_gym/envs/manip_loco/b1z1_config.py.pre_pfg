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


class B1Z1RoughCfgPPO(B1Z1RoughCfgPPO3D):
    pass
