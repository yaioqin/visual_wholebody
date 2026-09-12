# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

# codexchange: 2026-09-12，按当前URDF和实际控制路径复核；以下为轻载、低速全身协调训练基线，未经训练证明最优。
# [S1] 宇树B2规格（约60kg含电池、外形站高645mm、最大关节力矩360N·m）：
# https://www.unitree.com/b2/
# [S2] 宇树Z1规格（标称reach 740mm，AIR/PRO及工具版本不同）：
# https://www.unitree.com/z1/
# [S3] 宇树官方B2仿真PD起点160/5（本次核查main，不是B2-Z1任务最优值）：
# https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
# 几何/质量以asset.file为准：B2=60kg、Z1含夹爪=5.22196983kg，总计65.22196983kg；不重复加臂质量。
# 默认腿角FK：接地base_z=0.50957593m；reset随机角最深需0.58283815m，出生高度与站高目标分别设置。
# 当前base视觉包络含雷达：x[-0.42337,0.43099], y约±0.14893, z[-0.10498,0.24250]m；禁区不能只按简化碰撞箱设置。
# 实现限制：臂由IK+硬编码400/40位置驱动，非18维策略直接力矩控制；配置z1增益/臂动作缩放不改变该驱动。
# 球心、base_height及feet_height奖励实际使用世界高度；此基线针对现有以z=0为中心的rough flat，非任意地形自适应。
# 目标禁区仅检查EE离散轨迹；姿态约束、全臂自碰撞和边缘目标的机身补偿仍需仿真验证。
# max_contact_force当前作用于含力/力矩的六维wrench范数，不能视作严格的单足牛顿硬限幅。
# “保留”表示已复核无需因本次模型迁移修改；注明未生效的字段不能靠调数值修正其代码路径。

class B2Z1RoughCfg( LeggedRobotCfg ):
    class goal_ee:
        num_commands = 3
        traj_time = [1, 3] 
        hold_time = [0.5, 2]
        collision_upper_limits = [0.15, 0.20, 0.05] # codexchange: 按当前base视觉外形含雷达包络加约5cm余量，坐标相对目标球心
        collision_lower_limits = [-0.82, -0.20, -0.74851] # codexchange: 与球心世界z=0.74851联动，下界延伸至世界z=0；仅EE目标禁区
        underground_limit = -0.56851 # codexchange: 世界最低目标z=0.18=当前粗糙地形上界0.10+0.08m工具余量，再减球心0.74851
        num_collision_check_samples = 20 # codexchange: 10→20，增加球坐标插值轨迹检查密度；仍非全臂碰撞检测
        command_mode = 'sphere'
        arm_induced_pitch = 0.621807 # codexchange: 0.38→0.621807 rad，默认臂FK的球坐标俯仰0.611807+工具俯仰0.01；不是安装旋转角

        class sphere_center:
            x_offset = 0.34218
            y_offset = 0
            z_invariant_offset = 0.74851 # codexchange: 0.84851→0.74851=标称base高度0.51+安装高度0.23851；当前实现固定世界z

        class ranges:
            init_pos_start = [0.562109, 0.611807, 0]
            init_pos_end = [0.7, 0, 0]
            pos_l = [0.4, 0.95]
            pos_p = [-1 * np.pi / 4, 1 * np.pi / 3]
            pos_y = [-1.2, 1.2]
            
            delta_orn_r = [-0.5, 0.5] 
            delta_orn_p = [-0.5, 0.5] 
            delta_orn_y = [-0.5, 0.5] 
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1   

    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3. # time before command are changed[s]

        lin_vel_x_schedule = [0, 0.5]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]

        ang_vel_yaw_clip = 0.5
        lin_vel_x_clip = 0.2

        class ranges:
            lin_vel_x = [-0.8, 0.8]
            ang_vel_yaw = [-1.0, 1.0]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel =  1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 2.0 # codexchange: 100→2，与腿action_scale及motor_strength上限1.1联动，使腿位置目标处于软限位内

    class env:
        num_envs = 6144
        num_actions = 12 + 6
        num_torques = 12 + 6
        action_delay = 3
        num_gripper_joints = 1
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3
        num_priv = 5 + 1 + 12
        history_len = 10
        num_observations = num_proprio * (history_len+1) + num_priv
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds
        reorder_dofs = True
        teleop_mode = False # Overriden in teleop.py. When true, commands come from keyboard
        record_video = False
        stand_by = False
        observe_gait_commands = False
        frequencies = 2

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.60] # codexchange: 出生z：0.52→0.60m；reset关节角乘[0.8,1.2]时最深足底需0.58284m，另留约17mm；非站高目标
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.2,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,

            'RL_hip_joint': 0.2,
            'RL_thigh_joint': 0.8,
            'RL_calf_joint': -1.5,

            'FR_hip_joint': -0.2 ,
            'FR_thigh_joint': 0.8,
            'FR_calf_joint': -1.5,

            'RR_hip_joint': -0.2,
            'RR_thigh_joint': 0.8,
            'RR_calf_joint': -1.5,

            'z1_waist': 0.0,
            'z1_shoulder': 1.48,
            'z1_elbow': -0.63,
            'z1_wrist_angle': -0.84,
            'z1_forearm_roll': 0.0,
            'z1_wrist_rotate': np.pi / 2, # codexchange: 1.57→pi/2，与当前IK目标默认roll=pi/2一致
            'z1_jointGripper': -0.785, 
        }
        rand_yaw_range = np.pi/2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control:
        stiffness = {'joint': 160.0, 'z1': 5} # codexchange: 腿80→160 N·m/rad，参考官方B2 RL配置[S3]；z1保留但当前臂力矩被清零
        damping = {'joint': 5.0, 'z1': 0.5} # codexchange: 腿2→5 N·m·s/rad，参考[S3]；实际Z1位置驱动为manip_loco.py硬编码400/40

        adaptive_arm_gains = False
        # action scale: target angle = actionScale * clipped_action * motor_strength + defaultAngle
        action_scale = [0.25, 0.40, 0.40] * 4 + [2.1, 0.6, 0.6, 0, 0, 0] # codexchange: 腿缩放联动clip=2、motor≤1.1：hip绝对目标≤0.75，thigh[-0.08,1.68]，calf[-2.38,-0.62]rad；臂6项保留且不控制IK
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b2_z1_lidar_mount_fast/urdf/b2_z1_lidar_mount_fast.urdf'
        base_body_name = "base_link"
        foot_name = "foot"
        gripper_name = "ee_gripper_link"
        penalize_contacts_on = ["thigh", "base_link", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 0
        flip_visual_attachments = False
        collapse_fixed_joints = True
        fix_base_link = False
    
    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16
    
    class arm:
        base_offset = [0.34218, 0.0, 0.23851]
        init_target_ee_base = [0.460149, 0.0, 0.322846] # codexchange: 同步默认臂FK相对安装基座位置；当前仅初始化缓存，实际目标由goal_ee生成
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0] # [0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [-3.0, 5.0] # codexchange: 0–15→-3–5kg，轻载/附件及质量辨识误差初值；不重复增加已建模的Z1质量
        randomize_base_com = True
        added_com_range_x = [-0.05, 0.05] # codexchange: ±0.15→±0.05m，直接作用于约29.24kg基座COM的前后误差初值
        added_com_range_y = [-0.03, 0.03] # codexchange: ±0.15→±0.03m，居中安装条件下的横向COM误差初值
        added_com_range_z = [-0.02, 0.02] # codexchange: ±0.15→±0.02m，基座垂向COM误差初值
        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1] # codexchange: 0.7–1.3→0.9–1.1，与clip/action_scale联动保留关节目标余量，需实机辨识
        arm_motor_strength_range = [0.7, 1.3]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0.0, 0.5] # codexchange: 0–0.1→0–0.5kg额外负载初值，未指定载荷时不直接用Z1额定载荷上限
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.2 # codexchange: 0.5→0.2m/s；当前静止条件再乘2.5，最大每轴0.5m/s，属于速度重置而非固定外力
  
    class rewards:
        reward_container_name = "maniploco_rewards"   # select the reward container to use

        # -------Common Para. ---------
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 0.9 # codexchange: 1→0.9，基类_process_dof_props确实缩小限位，留总区间10%软余量
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.4
        base_height_target = 0.51 # codexchange: 0.45→0.51m，默认腿角FK足底接地所需0.509576m；当前奖励按世界z，非相对地形
        max_contact_force = 500.0 # codexchange: 40→500，65.222kg四足静载约160N/足，最大随机质量下双足约347N/足再留动态余量；当前奖励混用六维wrench范数
        # -------Gait control Para. ---------
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        kappa_gait_probs = 0.07
        feet_height_target = 0.18 # codexchange: 0.30→0.18m，减小操作步态抬足幅度；当前是前两足世界高度范数阈值，非单足净空

        feet_aritime_allfeet = False
        feet_height_allfeet = False

        # Scales set to 0 will still be logged (as zero reward and non-zero metric)
        # To not compute and log a given metric, set the scale to None
        class scales:
            # -------Gait control rewards ---------
            tracking_contacts_shaped_force = 2.0 # codexchange: 奖励函数已返回负值，使用正权重惩罚摆动期触地；仅启用步态观测时生效
            tracking_contacts_shaped_vel = 2.0 # codexchange: 奖励函数已返回负值，使用正权重惩罚支撑期足端滑动；仅启用步态观测时生效
            feet_air_time = 2.0
            feet_height = 1.0

            # -------Tracking rewards ----------
            tracking_lin_vel_max = 2.0 
            tracking_lin_vel_x_l1 = 0.
            tracking_lin_vel_x_exp = 0
            tracking_ang_vel = 0.5

            # codexchange: 删除后续同名赋值覆盖的旧 delta_torques，有效值及依据见下方。
            # codexchange: 删除后续同名赋值覆盖的旧 work，有效值及依据见下方。
            energy_square = 0.0
            torques = -2.0e-5 # codexchange: -2.5e-5×(57.987/65.222)^2≈-1.98e-5，按URDF质量近似归一的训练起点
            stand_still = 1.0
            walking_dof = 1.5
            dof_default_pos = 0.0
            dof_error = 0.0 
            alive = 1.0
            lin_vel_z = -1.5
            roll = -2

            # common rewards
            ang_vel_xy = -0.2
            dof_acc = -7.5e-7
            collision = -10.
            action_rate = -0.015
            dof_pos_limits = -10.0
            delta_torques = -8.0e-8 # codexchange: 有效-1e-7→-8e-8，按(57.987/65.222)^2缩放腿力矩差平方项
            hip_pos = -0.3
            work = -0.0027 # codexchange: 有效-0.003×57.987/65.222≈-0.00267，腿净功率绝对值的质量尺度起点
            feet_jerk = -0.0002
            feet_drag = -0.08
            feet_contact_forces = -0.0009 # codexchange: -0.001×57.987/65.222≈-0.000889，并将阈值联动至500；需修正六维wrench混合量纲
            orientation = 0.0
            orientation_walking = 0.0
            orientation_standing = 0.0
            base_height = -5.0
            torques_walking = 0.0
            torques_standing = 0.0
            energy_square = 0.0
            energy_square_walking = 0.0
            energy_square_standing = 0.0
            base_height_walking = 0.0
            base_height_standing = 0.0
            penalty_lin_vel_y = 0.

        class arm_scales:
            arm_termination = None
            tracking_ee_sphere = 0.
            tracking_ee_world = 0.8
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = None
        
    class viewer:
        pos = [-20, 0, 20]  # [m]
        lookat = [0, 0, -2]  # [m]

    class termination:
        r_threshold = 0.8
        p_threshold = 0.8
        z_threshold = 0.1

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.01 # codexchange: 0.1→0.01m，fast网格简化误差降至足端球体半径0.032m以内，保留粗糙地形细节
        horizontal_scale = 0.05
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        height = [0.00, 0.1]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 20 # number of terrain cols (types)

        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = None # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False


class B2Z1RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        continue_from_last_std = True
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_tanh = False

        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 6

        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2e-4 
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]

        mixing_schedule=[1.0, 0, 3000] #if not RESUME else [1.0, 0, 1]
        torque_supervision = B2Z1RoughCfg.control.torque_supervision  #alert: also appears above
        torque_supervision_schedule=[0.0, 1000, 1000]
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000] #if not RESUME else [0, 1, 1000, 1000]

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 37000 # number of policy updates
        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'b2z1_v3' # codexchange: v2→v3，区分本次PD/目标空间/奖励基线，避免自动混用旧训练
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
