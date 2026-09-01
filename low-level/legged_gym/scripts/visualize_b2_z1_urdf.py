# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone B2-Z1 asset viewer.

This viewer defaults to the local Isaac Gym URDF.  --asset-file can also point
at an MJCF/XML file because Isaac Gym loads both formats through gym.load_asset().
"""

import argparse
import os
from pathlib import Path

import numpy as np
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR


B2_Z1_DEFAULT_JOINTS = {
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
}


ARM_GAINS = {
    "joint1": (50.0, 3.0),
    "joint2": (50.0, 2.0),
    "joint3": (80.0, 3.0),
    "joint4": (30.0, 3.0),
    "joint5": (30.0, 2.5),
    "joint6": (20.0, 1.0),
}

DOF_MODE_POS = int(gymapi.DOF_MODE_POS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the B2-Z1 URDF/MJCF asset in Isaac Gym."
    )
    parser.add_argument(
        "--asset-file",
        default="{LEGGED_GYM_ROOT_DIR}/resources/robots/b2_z1/urdf/b2_z1.urdf",
        help="URDF or MJCF/XML file to load. Defaults to the local B2-Z1 URDF.",
    )
    parser.add_argument(
        "--sim-device",
        "--sim_device",
        dest="sim_device",
        default="cpu",
        help="Simulation device, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--graphics-device-id",
        "--graphics_device_id",
        dest="graphics_device_id",
        type=int,
        default=0,
        help="Graphics device id used by the viewer.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not create a viewer; useful for smoke-testing asset loading.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of simulation steps to run. 0 keeps the viewer open until closed.",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Fix the floating base so the robot stays suspended at the root pose.",
    )
    parser.add_argument(
        "--root-pos",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.55],
        metavar=("X", "Y", "Z"),
        help="Actor root position.",
    )
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=[2.2, -2.0, 1.2],
        metavar=("X", "Y", "Z"),
        help="Viewer camera position.",
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.45],
        metavar=("X", "Y", "Z"),
        help="Viewer camera target.",
    )
    parser.add_argument(
        "--print-names",
        action="store_true",
        help="Print rigid-body and DOF names after loading.",
    )
    return parser.parse_args()


def resolve_asset_path(asset_file):
    expanded = asset_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    path = Path(os.path.expanduser(expanded)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Asset file does not exist: {path}")
    return path


def make_sim(gym, args):
    sim_device_type, sim_device_id = gymutil.parse_device_str(args.sim_device)
    graphics_device_id = -1 if args.headless else args.graphics_device_id

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = sim_device_type == "cuda"
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(
        sim_device_id,
        graphics_device_id,
        gymapi.SIM_PHYSX,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")
    return sim


def set_joint_targets(gym, env, actor, asset, dof_props):
    dof_names = gym.get_asset_dof_names(asset)
    dof_states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    targets = np.zeros(len(dof_names), dtype=np.float32)

    for i, name in enumerate(dof_names):
        default_pos = B2_Z1_DEFAULT_JOINTS.get(name, 0.0)
        dof_states["pos"][i] = default_pos
        targets[i] = default_pos

        if name in ARM_GAINS:
            stiffness, damping = ARM_GAINS[name]
        else:
            stiffness, damping = 250.0, 5.0

        dof_props["driveMode"][i] = DOF_MODE_POS
        dof_props["stiffness"][i] = stiffness
        dof_props["damping"][i] = damping

    gym.set_actor_dof_properties(env, actor, dof_props)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, targets)


def main():
    args = parse_args()
    asset_path = resolve_asset_path(args.asset_file)

    gym = gymapi.acquire_gym()
    sim = make_sim(gym, args)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    gym.add_ground(sim, plane_params)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = DOF_MODE_POS
    asset_options.collapse_fixed_joints = False
    asset_options.fix_base_link = args.fix_base
    asset_options.replace_cylinder_with_capsule = False
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.disable_gravity = False
    asset_options.density = 0.001
    asset_options.armature = 0.01

    asset_root = str(asset_path.parent)
    asset_file = asset_path.name
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_path}")

    env = gym.create_env(
        sim,
        gymapi.Vec3(-1.5, -1.5, 0.0),
        gymapi.Vec3(1.5, 1.5, 1.5),
        1,
    )
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*args.root_pos)
    actor = gym.create_actor(env, asset, pose, "b2_z1", 0, 0, 0)

    dof_props = gym.get_asset_dof_properties(asset)
    set_joint_targets(gym, env, actor, asset, dof_props)

    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    print(f"Loaded asset: {asset_path}")
    print(f"Rigid bodies: {len(body_names)}")
    print(f"DOFs: {len(dof_names)}")
    if args.print_names:
        print("Body names:", body_names)
        print("DOF names:", dof_names)

    gym.prepare_sim(sim)

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create viewer")
        gym.viewer_camera_look_at(
            viewer,
            None,
            gymapi.Vec3(*args.camera_pos),
            gymapi.Vec3(*args.camera_lookat),
        )

    step = 0
    while True:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        if args.steps > 0 and step >= args.steps:
            break

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        step += 1

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
