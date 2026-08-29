"""Deep Whole-Body Control configuration for the B1 + Z1 platform.

The learning parameters in this file follow the public DWBC ``widowGo1``
implementation. Robot-specific geometry (URDF, safe Z1 rest pose and the
arm-base offset) remains B1/Z1-specific.
"""

import numpy as np

from .b1z1_config_3D import (
    B1Z1RoughCfg as B1Z1RoughCfg3D,
    B1Z1RoughCfgPPO as B1Z1RoughCfgPPO3D,
)


class B1Z1RoughCfg(B1Z1RoughCfg3D):
    class goal_ee(B1Z1RoughCfg3D.goal_ee):
        traj_time = [1.0, 3.0]
        hold_time = [0.5, 2.0]
        command_mode = "sphere"

        # The source controller keeps the spherical command origin invariant
        # to torso height/roll/pitch. These offsets place that origin at the
        # Z1 mount on B1.
        class sphere_center:
            x_offset = 0.30
            y_offset = 0.0
            z_invariant_offset = 0.70

        class ranges:
            init_pos_start = [0.30, np.pi / 4.0, 0.0]
            init_pos_end = [0.50, np.pi / 4.0, 0.0]
            pos_l = [0.20, 0.70]
            pos_p = [-2.0 * np.pi / 5.0, np.pi / 5.0]
            pos_y = [-3.0 * np.pi / 5.0, 3.0 * np.pi / 5.0]
            delta_orn_r = [0.0, 0.0]
            delta_orn_p = [0.0, 0.0]
            delta_orn_y = [0.0, 0.0]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [
            1.0 / (ranges.pos_l[1] - ranges.pos_l[0]),
            1.0 / (ranges.pos_p[1] - ranges.pos_p[0]),
            1.0 / (ranges.pos_y[1] - ranges.pos_y[0]),
        ]
        orn_error_scale = [2.0 / np.pi, 2.0 / np.pi, 2.0 / np.pi]

    class commands(B1Z1RoughCfg3D.commands):
        curriculum = False
        num_commands = 3
        resampling_time = 3.0
        use_5d_base_command = False
        lin_vel_x_clip = 0.30
        ang_vel_yaw_clip = 0.60

        class ranges:
            lin_vel_x = [0.0, 0.90]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]
            # Retained so legacy 5-D evaluation flags fail gracefully.
            base_pitch = [-0.30, 0.30]
            base_height = [0.32, 0.45]

    class env(B1Z1RoughCfg3D.env):
        num_envs = 5000
        num_actions = 18
        num_torques = 18
        action_delay = 2
        num_gripper_joints = 1

        # roll/pitch + angular velocity + 18 q + 18 qdot + 18 previous
        # actions + 4 contacts + 3 base commands + 3 EE position commands +
        # 3 EE orientation commands.
        base_num_proprio = 72
        num_proprio = base_num_proprio
        # base payload/COM (5), friction (1), all 18 motor strengths.
        num_priv = 24
        history_len = 10
        num_observations = num_proprio * (history_len + 1) + num_priv
        num_privileged_obs = None
        send_timeouts = True
        episode_length_s = 10
        observe_gait_commands = False

    class init_state(B1Z1RoughCfg3D.init_state):
        # Keep the B1/Z1 joint rest pose, but match DWBC's translational
        # perturbation without the unrelated random-yaw curriculum.
        rand_yaw_range = 0.0
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control(B1Z1RoughCfg3D.control):
        stiffness = {"joint": 50.0, "z1": 5.0}
        damping = {"joint": 1.0, "z1": 0.5}
        adaptive_arm_gains = False
        action_scale = (
            [0.40, 0.45, 0.45] * 4
            + [2.10, 0.60, 0.60, 0.0, 0.0, 0.0]
        )
        decimation = 4
        torque_supervision = False

    class asset(B1Z1RoughCfg3D.asset):
        penalize_contacts_on = ["thigh", "trunk"]
        terminate_after_contacts_on = []

    class domain_rand(B1Z1RoughCfg3D.domain_rand):
        observe_priv = True
        randomize_friction = True
        friction_range = [-0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 2.5]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0.0, 0.1]
        push_robots = True
        push_interval_s = 3.0
        max_push_vel_xy = 0.5

    class noise(B1Z1RoughCfg3D.noise):
        add_noise = False

    class rewards(B1Z1RoughCfg3D.rewards):
        reward_container_name = "maniploco_rewards"
        only_positive_rewards = False
        tracking_sigma = 1.0
        tracking_ee_sigma = 1.0
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.25
        max_contact_force = 100.0

        class scales:
            # Exact non-zero locomotion terms from the public DWBC config.
            energy_square = -6.0e-5
            survive = 0.2
            tracking_lin_vel_x_l1 = 0.5
            tracking_ang_vel_yaw_exp = 0.15
            hip_action_l2 = -0.01
            foot_contacts_z = -1.0e-4

        class arm_scales:
            # Exact non-zero manipulation terms from the public DWBC config.
            tracking_ee_sphere = 0.55
            arm_energy_abs_sum = -0.0040

    class termination(B1Z1RoughCfg3D.termination):
        r_threshold = 0.20
        p_threshold = 0.20
        z_threshold = 0.325

    class terrain(B1Z1RoughCfg3D.terrain):
        # DWBC uses fractal Perlin terrain instead of hand-shaped gait rewards.
        dwbc_perlin = True
        mesh_type = "trimesh"
        horizontal_scale = 0.025
        vertical_scale = 1.0 / 100000.0
        border_size = 0.0
        tot_cols = 600
        tot_rows = 10000
        zScale = 0.15
        transform_x = -tot_cols * horizontal_scale / 2.0
        transform_y = -tot_rows * horizontal_scale / 2.0
        transform_z = 0.0
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = False
        slope_treshold = 100000000
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class multi_agent:
        # Compatibility metadata used by the evaluation utilities. DWBC does
        # not gate or convert the arm action: all 18 dimensions are learned.
        use_arm_base_message = False
        use_arm_delta_action = False
        allow_arm_policy_action = True
        use_assist_reward = False
        debug_print_kinematics_names = False
        arm_chunk_horizon = 1
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
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        continue_from_last_std = True
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = "elu"
        output_tanh = True
        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]
        priv_encoder_dims = [64, 20]
        num_leg_actions = 12
        num_arm_actions = 6
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2.0e-4
        schedule = "fixed"
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.0
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]
        mixing_schedule = [1.0, 0, 3000]
        torque_supervision = B1Z1RoughCfg.control.torque_supervision
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0.0, 0.1, 3000, 7000]
        only_train_leg = False

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 40
        max_iterations = 40000
        save_interval = 500
        experiment_name = "dwbc_b1z1"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
