import copy
import os
import time

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import euler_from_quat
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR


FOOT_VISUAL_RADIUS = 0.0165
FOOT_COLLISION_RADIUS = 0.0265

DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.2,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "RL_hip_joint": 0.2,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
    "FR_hip_joint": -0.2,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RR_hip_joint": -0.2,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
    "z1_waist": 0.0,
    "z1_shoulder": 1.48,
    "z1_elbow": -0.63,
    "z1_wrist_angle": -0.84,
    "z1_forearm_roll": 0.0,
    "z1_wrist_rotate": 1.57,
    "z1_jointGripper": -0.785,
}


def parse_args():
    return gymutil.parse_arguments(
        description="Visualize Aliengo + Z1 URDF in its standby pose.",
        custom_parameters=[
            {
                "name": "--duration",
                "type": float,
                "default": 0.0,
                "help": "Seconds to run. 0 means run until the viewer is closed.",
            },
            {
                "name": "--fix_base",
                "action": "store_true",
                "default": False,
                "help": "Fix the base link in the air for easier mesh inspection.",
            },
            {
                "name": "--headless",
                "action": "store_true",
                "default": False,
                "help": "Run without creating an Isaac Gym viewer.",
            },
            {
                "name": "--simulate",
                "action": "store_true",
                "default": False,
                "help": "Advance physics. By default the script only renders the static standby pose.",
            },
            {
                "name": "--collapse_fixed_joints",
                "action": "store_true",
                "default": False,
                "help": "Match the training asset option by collapsing fixed joints. Disabled by default for URDF assembly inspection.",
            },
            {
                "name": "--no_flip_visual_attachments",
                "action": "store_true",
                "default": False,
                "help": "Disable Isaac Gym visual attachment frame conversion.",
            },
            {
                "name": "--arm_zero",
                "action": "store_true",
                "default": False,
                "help": "Set all Z1 joints to zero for simple kinematic assembly inspection.",
            },
            {
                "name": "--base_height",
                "type": float,
                "default": 0.45,
                "help": "Initial base height in meters before optional foot-to-ground alignment.",
            },
            {
                "name": "--no_auto_ground",
                "action": "store_true",
                "default": False,
                "help": "Keep the requested base height instead of aligning the feet to the ground.",
            },
            {
                "name": "--foot_radius",
                "type": float,
                "default": -1.0,
                "help": "Foot radius used by auto-ground alignment. Negative selects visual radius for static view and collision radius for --simulate.",
            },
            {
                "name": "--zero_action_pd",
                "action": "store_true",
                "default": False,
                "help": "Run the registered aliengo_z1 task with zero policy actions to test whether the default stance PD can support the robot.",
            },
            {
                "name": "--zero_action_steps",
                "type": int,
                "default": 500,
                "help": "Number of policy steps for --zero_action_pd when --duration is 0.",
            },
            {
                "name": "--zero_action_log_interval",
                "type": int,
                "default": 10,
                "help": "Print state every N policy steps in --zero_action_pd mode.",
            },
            {
                "name": "--zero_action_record_video",
                "action": "store_true",
                "default": False,
                "help": "Record a video during --zero_action_pd mode.",
            },
            {
                "name": "--zero_action_video_path",
                "type": str,
                "default": "",
                "help": "Optional mp4 path for --zero_action_record_video. Defaults to logs/videos/zero_action_pd/aliengo_z1_zero_action_pd.mp4.",
            },
            {
                "name": "--zero_action_continue_after_done",
                "action": "store_true",
                "default": False,
                "help": "Keep stepping after the first reset in --zero_action_pd mode.",
            },
            {
                "name": "--task",
                "type": str,
                "default": "aliengo_z1",
                "help": "Task name used to load the training terrain and by --zero_action_pd.",
            },
            {
                "name": "--observe_gait_commands",
                "action": "store_true",
                "default": False,
                "help": "Enable gait-command observations in --zero_action_pd mode.",
            },
        ],
    )


def prepare_task_args(args, record_video):
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    args.num_envs = 1
    args.stop_update_goal = False
    args.seed = 1
    args.rows = None
    args.cols = None
    args.record_video = record_video
    args.stand_by = True
    args.vel_obs = False
    args.pitch_control = False
    return args


def configure_zero_action_cfg(env_cfg, record_video):
    env_cfg.env.num_envs = 1
    env_cfg.env.record_video = record_video
    env_cfg.env.stand_by = True
    env_cfg.env.teleop_mode = True

    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_gripper_mass = False

    env_cfg.init_state.rand_yaw_range = 0.0
    env_cfg.init_state.origin_perturb_range = 0.0
    env_cfg.init_state.init_vel_perturb_range = 0.0
    return env_cfg


def lock_arm_target_to_current_pose(env):
    if not all(hasattr(env, name) for name in ("curr_ee_goal_cart_world", "ee_goal_orn_quat")):
        return
    env.commands[:] = 0.0
    env.curr_ee_goal_cart_world[:] = env.ee_pos
    env.ee_goal_orn_quat[:] = env.ee_orn / torch.norm(
        env.ee_orn, dim=-1, keepdim=True
    ).clamp(min=1e-6)


def zero_action_video_path(args):
    if args.zero_action_video_path:
        return args.zero_action_video_path
    return os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "videos",
        "zero_action_pd",
        "aliengo_z1_zero_action_pd.mp4",
    )


def run_zero_action_pd(args):
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import task_registry

    record_video = args.zero_action_record_video
    args = prepare_task_args(args, record_video)
    if record_video:
        args.headless = True

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = configure_zero_action_cfg(env_cfg, record_video)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs, _ = env.reset()
    del obs
    lock_arm_target_to_current_pose(env)
    env.compute_observations()

    if args.duration > 0.0:
        num_steps = int(args.duration / env.dt)
    else:
        num_steps = args.zero_action_steps

    writer = None
    if record_video:
        import imageio

        video_path = zero_action_video_path(args)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=25)
        print("Recording zero-action PD video to:", video_path)

    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    print("Running zero-action PD diagnostic")
    print("task:", args.task)
    print("dt:", env.dt)
    print("steps:", num_steps)
    print("record_video:", record_video)

    first_done_step = None
    for step in range(num_steps):
        lock_arm_target_to_current_pose(env)
        _, _, rews, arm_rews, dones, _ = env.step(zero_actions)

        if record_video:
            imgs = env.render_record(mode="rgb_array")
            if imgs is not None:
                writer.append_data(imgs[0])

        should_log = (
            step == 0
            or step % args.zero_action_log_interval == 0
            or bool(dones[0].item())
            or step == num_steps - 1
        )
        if should_log:
            roll, pitch, yaw = euler_from_quat(env.base_quat)
            print(
                "step={:04d} z={:.3f} roll={:.3f} pitch={:.3f} yaw={:.3f} "
                "rew={:.4f} arm_rew={:.4f} done={}".format(
                    step,
                    env.root_states[0, 2].item(),
                    roll[0].item(),
                    pitch[0].item(),
                    yaw[0].item(),
                    rews[0].item(),
                    arm_rews[0].item(),
                    bool(dones[0].item()),
                )
            )

        if bool(dones[0].item()) and first_done_step is None:
            first_done_step = step
            if not args.zero_action_continue_after_done:
                break

    if writer is not None:
        writer.close()

    if first_done_step is None:
        print("Result: zero-action PD did not trigger reset.")
    else:
        print(f"Result: zero-action PD triggered reset at step {first_done_step}.")


def configure_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81 if args.simulate else 0.0)

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    return sim_params


def get_training_env_cfg(task_name):
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.task_registry import task_registry

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    return copy.deepcopy(env_cfg)


def select_training_spawn_origin(terrain, terrain_cfg):
    if terrain is None or not hasattr(terrain, "env_origins"):
        return np.zeros(3, dtype=np.float32), None, None

    max_init_level = terrain_cfg.max_init_terrain_level
    if not terrain_cfg.curriculum:
        max_init_level = terrain_cfg.num_rows - 1
    max_init_level = min(max_init_level, terrain_cfg.num_rows - 1)

    terrain_level = int(torch.randint(0, max_init_level + 1, (1,)).item())
    terrain_type = 0
    return (
        terrain.env_origins[terrain_level, terrain_type].astype(np.float32),
        terrain_level,
        terrain_type,
    )


def add_training_terrain(gym, sim, args):
    env_cfg = get_training_env_cfg(args.task)

    from legged_gym.utils.helpers import set_seed
    from legged_gym.utils.terrain import Terrain

    set_seed(env_cfg.seed)

    terrain_cfg = env_cfg.terrain
    mesh_type = terrain_cfg.mesh_type
    terrain = None

    if mesh_type == "trimesh":
        terrain = Terrain(terrain_cfg)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = terrain.vertices.shape[0]
        tm_params.nb_triangles = terrain.triangles.shape[0]
        tm_params.transform.p.x = -terrain.cfg.border_size
        tm_params.transform.p.y = -terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = terrain_cfg.static_friction
        tm_params.dynamic_friction = terrain_cfg.dynamic_friction
        tm_params.restitution = terrain_cfg.restitution
        gym.add_triangle_mesh(
            sim,
            terrain.vertices.flatten(order="C"),
            terrain.triangles.flatten(order="C"),
            tm_params,
        )
    elif mesh_type == "heightfield":
        terrain = Terrain(terrain_cfg)
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = terrain.cfg.horizontal_scale
        hf_params.row_scale = terrain.cfg.horizontal_scale
        hf_params.vertical_scale = terrain.cfg.vertical_scale
        hf_params.nbRows = terrain.tot_cols
        hf_params.nbColumns = terrain.tot_rows
        hf_params.transform.p.x = -terrain.cfg.border_size
        hf_params.transform.p.y = -terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = terrain_cfg.static_friction
        hf_params.dynamic_friction = terrain_cfg.dynamic_friction
        hf_params.restitution = terrain_cfg.restitution
        gym.add_heightfield(sim, terrain.heightsamples, hf_params)
    elif mesh_type == "plane":
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = terrain_cfg.static_friction
        plane_params.dynamic_friction = terrain_cfg.dynamic_friction
        plane_params.restitution = terrain_cfg.restitution
        gym.add_ground(sim, plane_params)
    elif mesh_type not in ("none", None):
        raise ValueError(
            "Terrain mesh type not recognised. Allowed types are "
            "[None, none, plane, heightfield, trimesh]"
        )

    spawn_origin, terrain_level, terrain_type = select_training_spawn_origin(
        terrain,
        terrain_cfg,
    )
    return env_cfg, terrain, spawn_origin, terrain_level, terrain_type


def load_robot(gym, sim, args):
    asset_path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "resources",
        "robots",
        "aliengo_z1",
        "urdf",
        "aliengo_z1.urdf",
    )
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.collapse_fixed_joints = args.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = False
    asset_options.flip_visual_attachments = not args.no_flip_visual_attachments
    asset_options.fix_base_link = args.fix_base
    asset_options.disable_gravity = False
    asset_options.use_mesh_materials = True
    asset_options.armature = 0.0

    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if robot_asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_path}")

    return robot_asset, asset_path


def set_standby_pose(gym, env, actor, robot_asset, arm_zero=False):
    dof_names = gym.get_asset_dof_names(robot_asset)
    dof_props = gym.get_actor_dof_properties(env, actor)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(0.0)
    dof_props["damping"].fill(0.0)

    for i, name in enumerate(dof_names):
        if name.startswith(("FL_", "FR_", "RL_", "RR_")):
            dof_props["stiffness"][i] = 80.0
            dof_props["damping"][i] = 2.0
        elif name.startswith("z1_"):
            dof_props["stiffness"][i] = 20.0
            dof_props["damping"][i] = 1.0

    gym.set_actor_dof_properties(env, actor, dof_props)

    dof_states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    targets = np.zeros(len(dof_names), dtype=np.float32)
    for i, name in enumerate(dof_names):
        angle = 0.0 if arm_zero and name.startswith("z1_") else DEFAULT_JOINT_ANGLES.get(name, 0.0)
        dof_states["pos"][i] = angle
        targets[i] = angle

    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, targets)
    return dof_names


def step_sim_once(gym, sim):
    gym.simulate(sim)
    gym.fetch_results(sim, True)


def shift_actor_root_z(gym, sim, env, actor, z_shift):
    actor_index = gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    gym.refresh_actor_root_state_tensor(sim)
    root_states[actor_index, 2] += z_shift
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))


def align_feet_to_ground(
    gym,
    sim,
    env,
    actor,
    body_names,
    foot_radius,
    settle_after_shift,
    ground_z=0.0,
):
    step_sim_once(gym, sim)

    rigid_body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    foot_indices = [
        i for i, name in enumerate(body_names)
        if name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    ]
    if not foot_indices:
        return 0.0, []

    foot_center_z = rigid_body_states["pose"]["p"]["z"][foot_indices]
    lowest_foot_center = float(np.min(foot_center_z))
    z_shift = ground_z + foot_radius - lowest_foot_center

    if abs(z_shift) > 1e-6:
        shift_actor_root_z(gym, sim, env, actor, z_shift)
        if settle_after_shift:
            step_sim_once(gym, sim)
            rigid_body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
            foot_center_z = rigid_body_states["pose"]["p"]["z"][foot_indices]
        else:
            foot_center_z = foot_center_z + z_shift

    return z_shift, list(zip((body_names[i] for i in foot_indices), foot_center_z.tolist()))


def main():
    args = parse_args()
    if args.zero_action_pd:
        run_zero_action_pd(args)
        return

    gym = gymapi.acquire_gym()
    sim_params = configure_sim(gym, args)

    graphics_device_id = -1 if args.headless else args.graphics_device_id
    sim = gym.create_sim(
        args.compute_device_id,
        graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")

    env_cfg, terrain, spawn_origin, terrain_level, terrain_type = add_training_terrain(
        gym,
        sim,
        args,
    )

    robot_asset, asset_path = load_robot(gym, sim, args)
    env = gym.create_env(
        sim,
        gymapi.Vec3(0.0, 0.0, 0.0),
        gymapi.Vec3(0.0, 0.0, 0.0),
        1,
    )
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(
        float(spawn_origin[0]),
        float(spawn_origin[1]),
        float(spawn_origin[2] + args.base_height),
    )
    actor = gym.create_actor(env, robot_asset, pose, "aliengo_z1", 0, 0, 0)

    dof_names = set_standby_pose(gym, env, actor, robot_asset, args.arm_zero)
    body_names = gym.get_asset_rigid_body_names(robot_asset)
    gym.prepare_sim(sim)
    foot_radius = (
        args.foot_radius
        if args.foot_radius > 0.0
        else (FOOT_COLLISION_RADIUS if args.simulate else FOOT_VISUAL_RADIUS)
    )

    if args.no_auto_ground:
        step_sim_once(gym, sim)
        ground_shift = 0.0
        foot_heights = []
    else:
        ground_shift, foot_heights = align_feet_to_ground(
            gym,
            sim,
            env,
            actor,
            body_names,
            foot_radius,
            args.simulate,
            float(spawn_origin[2]),
        )

    print("Loaded asset:", asset_path)
    print("terrain_task:", args.task)
    print("terrain_mesh_type:", env_cfg.terrain.mesh_type)
    print("terrain_spawn_origin:", spawn_origin.tolist())
    if terrain_level is not None:
        print("terrain_level:", terrain_level)
        print("terrain_type:", terrain_type)
    if terrain is not None and hasattr(terrain, "env_origins"):
        print("terrain_rows:", env_cfg.terrain.num_rows)
        print("terrain_cols:", env_cfg.terrain.num_cols)
    print("num_dofs:", len(dof_names))
    print("dof_names:", dof_names)
    print("num_bodies:", len(body_names))
    print("body_names:", body_names)
    print("collapse_fixed_joints:", args.collapse_fixed_joints)
    print("flip_visual_attachments:", not args.no_flip_visual_attachments)
    print("arm_zero:", args.arm_zero)
    print("auto_ground:", not args.no_auto_ground)
    print("auto_ground_foot_radius:", foot_radius)
    if foot_heights:
        print("ground_z_shift:", ground_shift)
        print("foot_center_z_after_alignment:", foot_heights)
        print("foot_bottom_z_after_alignment:", [z - foot_radius for _, z in foot_heights])
    print("GPU pipeline: disabled for viewer stability")
    if not args.simulate:
        print("Physics simulation: one initialization step only; rendering static standby pose")
    print("Press ESC or close the viewer to quit. Use mouse controls to inspect the robot.")

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create viewer")
        gym.viewer_camera_look_at(
            viewer,
            None,
            gymapi.Vec3(
                float(spawn_origin[0] + 1.6),
                float(spawn_origin[1] + 1.2),
                float(spawn_origin[2] + 0.9),
            ),
            gymapi.Vec3(
                float(spawn_origin[0]),
                float(spawn_origin[1]),
                float(spawn_origin[2] + 0.35),
            ),
        )

    start_time = time.time()
    if args.headless and args.duration <= 0.0:
        args.duration = 1.0

    while True:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        if args.duration > 0.0 and time.time() - start_time > args.duration:
            break

        if args.simulate:
            gym.simulate(sim)
            gym.fetch_results(sim, True)

        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
