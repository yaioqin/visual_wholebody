# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone B2-Z1 asset viewer.

This viewer defaults to the local Isaac Gym URDF.  --asset-file can also point
at an MJCF/XML file because Isaac Gym loads both formats through gym.load_asset().
"""

import argparse
import copy
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.manip_loco.b2z1_config import B2Z1RoughCfg


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
ARM_JOINT_NAMES = tuple(ARM_GAINS)
IK_ARM_SEED = {
    # The all-zero training pose is singular for Cartesian IK. These values are
    # the repository's standard non-singular Z1 arm pose from B1Z1RoughCfg.
    "joint1": 0.0,
    "joint2": 1.48,
    "joint3": -0.63,
    "joint4": -0.84,
    "joint5": 0.0,
    "joint6": 1.57,
}

DOF_MODE_POS = int(gymapi.DOF_MODE_POS)

EE_RANGE_COLOR = (0.0, 0.8, 1.0)
EE_SAMPLE_COLOR = (0.2, 1.0, 0.2)
EE_CENTER_COLOR = (1.0, 1.0, 0.0)
EE_COLLISION_COLOR = (1.0, 0.1, 0.1)
EE_UNDERGROUND_COLOR = (1.0, 0.55, 0.0)


class LineSetGeometry(gymutil.LineGeometry):
    """A collection of colored line segments understood by Isaac Gym."""

    def __init__(self, vertices, colors):
        vertices = np.asarray(vertices, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        if vertices.ndim != 3 or vertices.shape[1:] != (2, 3):
            raise ValueError("vertices must have shape (num_lines, 2, 3)")
        if colors.shape != (vertices.shape[0], 3):
            raise ValueError("colors must have shape (num_lines, 3)")

        self._vertices = np.empty(vertices.shape[:2], dtype=gymapi.Vec3.dtype)
        self._colors = np.empty(colors.shape[0], dtype=gymapi.Vec3.dtype)
        for index, field in enumerate(("x", "y", "z")):
            self._vertices[field] = vertices[..., index]
            self._colors[field] = colors[..., index]

    def vertices(self):
        return self._vertices

    def colors(self):
        return self._colors


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
    parser.add_argument(
        "--show-collisions",
        "--show_collisions",
        dest="show_collisions",
        action="store_true",
        help="Render collision geometry instead of the visual meshes.",
    )
    parser.add_argument(
        "--hide-ee-range",
        "--hide_ee_range",
        dest="show_ee_range",
        action="store_false",
        help="Hide the b2z1 training EE-goal sampling range.",
    )
    parser.add_argument(
        "--ee-samples",
        "--ee_samples",
        dest="ee_samples",
        type=int,
        default=400,
        help=(
            "Number of accepted training-style EE samples to show as green "
            "crosses. Use 0 to show only the range boundary."
        ),
    )
    parser.add_argument(
        "--disable-mouse-ik",
        "--disable_mouse_ik",
        dest="mouse_ik",
        action="store_false",
        help="Disable left-drag EE target selection and arm IK control.",
    )
    parser.set_defaults(show_ee_range=True, mouse_ik=True)
    return parser.parse_args()


def resolve_asset_path(asset_file):
    expanded = asset_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    path = Path(os.path.expanduser(expanded)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Asset file does not exist: {path}")
    return path


def make_collision_visual_urdf(asset_path):
    """Create a temporary URDF whose visual geometry mirrors its collisions.

    Isaac Gym Preview 4 does not reliably honor the viewer collision-render
    flag for URDF assets.  Copying collision elements to visual elements gives
    the viewer an exact collision-only representation without changing the
    physics collision definitions.
    """
    tree = ET.parse(asset_path)
    collision_count = 0

    for link in tree.getroot().findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)

        collisions = list(link.findall("collision"))
        if not collisions:
            continue

        first_collision_index = list(link).index(collisions[0])
        for offset, collision in enumerate(collisions):
            visual = copy.deepcopy(collision)
            visual.tag = "visual"
            link.insert(first_collision_index + offset, visual)
            collision_count += 1

    with tempfile.NamedTemporaryFile(
        mode="wb",
        # Isaac Gym treats a leading dot as the start of the file extension.
        prefix=f"{asset_path.stem}_collision_visual_",
        suffix=".urdf",
        dir=asset_path.parent,
        delete=False,
    ) as temp_file:
        tree.write(temp_file, encoding="utf-8", xml_declaration=True)
        temp_path = Path(temp_file.name)

    return temp_path, collision_count


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


def sphere_to_cartesian(sphere_coords):
    """NumPy equivalent of the training environment's sphere2cart()."""
    sphere_coords = np.asarray(sphere_coords, dtype=np.float64)
    radius = sphere_coords[..., 0]
    pitch = sphere_coords[..., 1]
    yaw = sphere_coords[..., 2]
    return np.stack(
        (
            radius * np.cos(pitch) * np.cos(yaw),
            radius * np.cos(pitch) * np.sin(yaw),
            radius * np.sin(pitch),
        ),
        axis=-1,
    )


def sample_training_ee_goals(goal_cfg, num_samples, seed=0):
    """Reproduce b2z1 EE sampling, including trajectory collision checks."""
    if num_samples < 0:
        raise ValueError("--ee-samples must be non-negative")

    ranges = goal_cfg.ranges
    lower = np.array(
        [ranges.pos_l[0], ranges.pos_p[0], ranges.pos_y[0]], dtype=np.float64
    )
    upper = np.array(
        [ranges.pos_l[1], ranges.pos_p[1], ranges.pos_y[1]], dtype=np.float64
    )
    collision_lower = np.asarray(goal_cfg.collision_lower_limits, dtype=np.float64)
    collision_upper = np.asarray(goal_cfg.collision_upper_limits, dtype=np.float64)
    check_t = np.linspace(0.0, 1.0, goal_cfg.num_collision_check_samples)
    previous = np.asarray(ranges.init_pos_end, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty((num_samples, 3), dtype=np.float64)

    for sample_index in range(num_samples):
        # This retry count and the spherical interpolation match
        # ManipLoco._resample_ee_goal() and _collision_check().
        for _ in range(10):
            candidate = rng.uniform(lower, upper)
            trajectory_sphere = (
                previous[None, :] * (1.0 - check_t[:, None])
                + candidate[None, :] * check_t[:, None]
            )
            trajectory_cart = sphere_to_cartesian(trajectory_sphere)
            inside_collision_box = np.all(
                (trajectory_cart < collision_upper)
                & (trajectory_cart > collision_lower),
                axis=1,
            )
            below_limit = trajectory_cart[:, 2] < goal_cfg.underground_limit
            if not np.any(inside_collision_box | below_limit):
                break

        samples[sample_index] = candidate
        previous = candidate

    return samples


def make_ee_sampling_geometry(goal_cfg, num_samples):
    """Build the wireframe sampling sector and its training filters."""
    vertices = []
    colors = []
    center = np.array(
        [
            goal_cfg.sphere_center.x_offset,
            goal_cfg.sphere_center.y_offset,
            goal_cfg.sphere_center.z_invariant_offset,
        ],
        dtype=np.float64,
    )
    radius_min, radius_max = goal_cfg.ranges.pos_l
    pitch_min, pitch_max = goal_cfg.ranges.pos_p
    yaw_min, yaw_max = goal_cfg.ranges.pos_y

    def add_segments(points, color):
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 2:
            return
        vertices.extend(np.stack((points[:-1], points[1:]), axis=1))
        colors.extend([color] * (len(points) - 1))

    def add_line(point_a, point_b, color):
        vertices.append(np.stack((point_a, point_b)))
        colors.append(color)

    # Inner and outer spherical-sector shells. Uniform parameter-space guides
    # also make the training distribution (uniform in l/pitch/yaw) apparent.
    yaw_curve = np.linspace(yaw_min, yaw_max, 49)
    pitch_curve = np.linspace(pitch_min, pitch_max, 49)
    pitch_guides = np.linspace(pitch_min, pitch_max, 7)
    yaw_guides = np.linspace(yaw_min, yaw_max, 7)
    for radius in (radius_min, radius_max):
        for pitch in pitch_guides:
            spherical = np.column_stack(
                (
                    np.full_like(yaw_curve, radius),
                    np.full_like(yaw_curve, pitch),
                    yaw_curve,
                )
            )
            add_segments(center + sphere_to_cartesian(spherical), EE_RANGE_COLOR)
        for yaw in yaw_guides:
            spherical = np.column_stack(
                (
                    np.full_like(pitch_curve, radius),
                    pitch_curve,
                    np.full_like(pitch_curve, yaw),
                )
            )
            add_segments(center + sphere_to_cartesian(spherical), EE_RANGE_COLOR)

    for pitch in (pitch_min, pitch_max):
        for yaw in (yaw_min, yaw_max):
            endpoints = np.array(
                [[radius_min, pitch, yaw], [radius_max, pitch, yaw]],
                dtype=np.float64,
            )
            add_segments(center + sphere_to_cartesian(endpoints), EE_RANGE_COLOR)

    # Accepted samples use the same sequential trajectory rejection logic as
    # training. Each target is a small three-axis cross for good visibility.
    sample_sphere = sample_training_ee_goals(goal_cfg, num_samples)
    sample_cart = center + sphere_to_cartesian(sample_sphere)
    marker_half_size = 0.006
    for point in sample_cart:
        for axis in np.eye(3) * marker_half_size:
            add_line(point - axis, point + axis, EE_SAMPLE_COLOR)

    # The spherical center used by _get_ee_goal_spherical_center().
    center_half_size = 0.04
    for axis in np.eye(3) * center_half_size:
        add_line(center - axis, center + axis, EE_CENTER_COLOR)

    # Training rejects any sampled trajectory entering this local Cartesian
    # box. Draw all twelve box edges in red.
    box_lower = center + np.asarray(goal_cfg.collision_lower_limits)
    box_upper = center + np.asarray(goal_cfg.collision_upper_limits)
    corners = np.array(
        [
            [x, y, z]
            for x in (box_lower[0], box_upper[0])
            for y in (box_lower[1], box_upper[1])
            for z in (box_lower[2], box_upper[2])
        ]
    )
    for first in range(len(corners)):
        for second in range(first + 1, len(corners)):
            if np.count_nonzero(corners[first] != corners[second]) == 1:
                add_line(corners[first], corners[second], EE_COLLISION_COLOR)

    # Orange rectangle: trajectories below this center-relative z are rejected.
    angle_grid = np.stack(
        np.meshgrid(pitch_curve, yaw_curve, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    boundary_sphere = []
    for radius in (radius_min, radius_max):
        boundary_sphere.append(
            np.column_stack(
                (
                    np.full(len(angle_grid), radius),
                    angle_grid[:, 0],
                    angle_grid[:, 1],
                )
            )
        )
    boundary_cart = sphere_to_cartesian(np.concatenate(boundary_sphere))
    plane_x = center[0] + np.array(
        [boundary_cart[:, 0].min(), boundary_cart[:, 0].max()]
    )
    plane_y = center[1] + np.array(
        [boundary_cart[:, 1].min(), boundary_cart[:, 1].max()]
    )
    plane_z = center[2] + goal_cfg.underground_limit
    plane_corners = np.array(
        [
            [plane_x[0], plane_y[0], plane_z],
            [plane_x[1], plane_y[0], plane_z],
            [plane_x[1], plane_y[1], plane_z],
            [plane_x[0], plane_y[1], plane_z],
            [plane_x[0], plane_y[0], plane_z],
        ]
    )
    add_segments(plane_corners, EE_UNDERGROUND_COLOR)

    return LineSetGeometry(vertices, colors)


def get_ee_range_anchor_pose(gym, env, actor, base_body_index):
    """Return the training goal frame: base x/y and yaw, with world z=0."""
    poses = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)["pose"]
    base_pose = gymapi.Transform.from_buffer(poses[base_body_index])
    quat = base_pose.r
    sin_yaw = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cos_yaw = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    half_yaw = 0.5 * np.arctan2(sin_yaw, cos_yaw)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(base_pose.p.x, base_pose.p.y, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw))
    return pose


def rotate_vector(quat, vector):
    """Rotate a NumPy vector by an Isaac Gym xyzw quaternion."""
    vector = np.asarray(vector, dtype=np.float64)
    quat_vector = np.array([quat.x, quat.y, quat.z], dtype=np.float64)
    return (
        2.0 * np.dot(quat_vector, vector) * quat_vector
        + (quat.w * quat.w - np.dot(quat_vector, quat_vector)) * vector
        + 2.0 * quat.w * np.cross(quat_vector, vector)
    )


def transform_point(pose, point):
    return np.array([pose.p.x, pose.p.y, pose.p.z]) + rotate_vector(
        pose.r, point
    )


def inverse_transform_point(pose, point):
    conjugate = gymapi.Quat(-pose.r.x, -pose.r.y, -pose.r.z, pose.r.w)
    translation = np.asarray(point) - np.array([pose.p.x, pose.p.y, pose.p.z])
    return rotate_vector(conjugate, translation)


def cartesian_to_sphere(cart_coords):
    cart_coords = np.asarray(cart_coords, dtype=np.float64)
    xy_length = np.linalg.norm(cart_coords[:2])
    return np.array(
        [
            np.linalg.norm(cart_coords),
            np.arctan2(cart_coords[2], xy_length),
            np.arctan2(cart_coords[1], cart_coords[0]),
        ]
    )


def clamp_ee_goal_sphere(goal_cfg, sphere_goal, previous_goal=None):
    """Clamp an interactive goal to b2z1's r/p/y and static filters."""
    ranges = goal_cfg.ranges
    lower = np.array([ranges.pos_l[0], ranges.pos_p[0], ranges.pos_y[0]])
    upper = np.array([ranges.pos_l[1], ranges.pos_p[1], ranges.pos_y[1]])
    sphere_goal = np.clip(np.asarray(sphere_goal, dtype=np.float64), lower, upper)

    # Match the training underground filter by lifting the pitch just enough.
    radius = sphere_goal[0]
    if radius > abs(goal_cfg.underground_limit):
        minimum_pitch = np.arcsin(goal_cfg.underground_limit / radius)
        sphere_goal[1] = max(sphere_goal[1], minimum_pitch)

    cart_goal = sphere_to_cartesian(sphere_goal)
    inside_collision_box = np.all(
        (cart_goal < np.asarray(goal_cfg.collision_upper_limits))
        & (cart_goal > np.asarray(goal_cfg.collision_lower_limits))
    )
    if inside_collision_box and previous_goal is not None:
        return np.asarray(previous_goal, dtype=np.float64).copy()
    return sphere_goal


class MouseIKController:
    """Map a viewer left-drag to a bounded EE target in a camera plane."""

    def __init__(self, goal_cfg, horizontal_fov):
        self.goal_cfg = goal_cfg
        self.horizontal_fov = horizontal_fov
        self.target_sphere = clamp_ee_goal_sphere(
            goal_cfg,
            np.array([goal_cfg.ranges.pos_l[0], 0.0, 0.0]),
        )
        self.dragging = False
        self.drag_plane_point = None
        self.drag_plane_normal = None

    def subscribe(self, gym, viewer):
        gym.subscribe_viewer_mouse_event(
            viewer, gymapi.MOUSE_LEFT_BUTTON, "ee_target_drag"
        )
        gym.subscribe_viewer_mouse_event(
            viewer, gymapi.MOUSE_SCROLL_UP, "ee_radius_increase"
        )
        gym.subscribe_viewer_mouse_event(
            viewer, gymapi.MOUSE_SCROLL_DOWN, "ee_radius_decrease"
        )

    def world_target(self, anchor_pose):
        center = np.array(
            [
                self.goal_cfg.sphere_center.x_offset,
                self.goal_cfg.sphere_center.y_offset,
                self.goal_cfg.sphere_center.z_invariant_offset,
            ]
        )
        local_target = center + sphere_to_cartesian(self.target_sphere)
        return transform_point(anchor_pose, local_target)

    def _screen_ray(self, gym, viewer, env):
        mouse = gym.get_viewer_mouse_position(viewer)
        window = gym.get_viewer_size(viewer)
        if window.x <= 0 or window.y <= 0:
            return None, None

        tangent_x = np.tan(np.deg2rad(self.horizontal_fov) * 0.5)
        tangent_y = tangent_x * window.y / window.x
        ray_camera = np.array(
            [
                (2.0 * mouse.x - 1.0) * tangent_x,
                (1.0 - 2.0 * mouse.y) * tangent_y,
                1.0,
            ]
        )
        ray_camera /= np.linalg.norm(ray_camera)
        camera_pose = gym.get_viewer_camera_transform(viewer, env)
        ray_world = rotate_vector(camera_pose.r, ray_camera)
        ray_world /= np.linalg.norm(ray_world)
        origin = np.array(
            [camera_pose.p.x, camera_pose.p.y, camera_pose.p.z]
        )
        return origin, ray_world

    def _update_drag_target(self, gym, viewer, env, anchor_pose):
        origin, ray = self._screen_ray(gym, viewer, env)
        if origin is None:
            return
        denominator = np.dot(ray, self.drag_plane_normal)
        if abs(denominator) < 1.0e-6:
            return
        distance = np.dot(
            self.drag_plane_point - origin, self.drag_plane_normal
        ) / denominator
        if distance <= 0.0:
            return

        world_point = origin + distance * ray
        center = np.array(
            [
                self.goal_cfg.sphere_center.x_offset,
                self.goal_cfg.sphere_center.y_offset,
                self.goal_cfg.sphere_center.z_invariant_offset,
            ]
        )
        local_point = inverse_transform_point(anchor_pose, world_point) - center
        candidate = cartesian_to_sphere(local_point)
        self.target_sphere = clamp_ee_goal_sphere(
            self.goal_cfg, candidate, self.target_sphere
        )

    def process_events(self, gym, viewer, env, anchor_pose):
        for event in gym.query_viewer_action_events(viewer):
            if event.action == "ee_target_drag":
                self.dragging = event.value > 0.0
                if self.dragging:
                    camera_pose = gym.get_viewer_camera_transform(viewer, env)
                    self.drag_plane_point = self.world_target(anchor_pose)
                    self.drag_plane_normal = rotate_vector(
                        camera_pose.r, np.array([0.0, 0.0, 1.0])
                    )
            elif event.action == "ee_radius_increase" and event.value > 0.0:
                candidate = self.target_sphere.copy()
                candidate[0] += 0.03
                self.target_sphere = clamp_ee_goal_sphere(
                    self.goal_cfg, candidate, self.target_sphere
                )
            elif event.action == "ee_radius_decrease" and event.value > 0.0:
                candidate = self.target_sphere.copy()
                candidate[0] -= 0.03
                self.target_sphere = clamp_ee_goal_sphere(
                    self.goal_cfg, candidate, self.target_sphere
                )

        if self.dragging:
            self._update_drag_target(gym, viewer, env, anchor_pose)


def update_arm_position_ik(
    gym,
    sim,
    env,
    actor,
    gripper_handle,
    jacobian,
    jacobian_body_index,
    jacobian_dof_columns,
    arm_dof_indices,
    dof_targets,
    dof_lower,
    dof_upper,
    target_world,
):
    """Take one damped-least-squares position IK step."""
    gym.refresh_jacobian_tensors(sim)
    gripper_pose = gym.get_rigid_transform(env, gripper_handle)
    current_world = np.array(
        [gripper_pose.p.x, gripper_pose.p.y, gripper_pose.p.z]
    )
    position_error = np.asarray(target_world) - current_world

    if np.linalg.norm(position_error) > 0.003:
        ee_jacobian = (
            jacobian[0, jacobian_body_index, :3, :][
                :, jacobian_dof_columns
            ]
            .detach()
            .cpu()
            .numpy()
        )
        damping = 0.05
        task_matrix = ee_jacobian @ ee_jacobian.T
        joint_delta = ee_jacobian.T @ np.linalg.solve(
            task_matrix + np.eye(3) * damping * damping,
            position_error,
        )
        joint_delta = np.clip(joint_delta, -0.07, 0.07)
        # ManipLoco applies each IK delta relative to the measured DOF state,
        # rather than repeatedly integrating it into the previous PD target.
        current_dof_pos = gym.get_actor_dof_states(
            env, actor, gymapi.STATE_POS
        )["pos"]
        dof_targets[arm_dof_indices] = (
            current_dof_pos[arm_dof_indices] + joint_delta
        )
        dof_targets[arm_dof_indices] = np.clip(
            dof_targets[arm_dof_indices], dof_lower, dof_upper
        )
        gym.set_actor_dof_position_targets(env, actor, dof_targets)

    return current_world, np.linalg.norm(position_error)


def set_joint_targets(gym, env, actor, asset, dof_props, use_ik_seed=False):
    dof_names = gym.get_asset_dof_names(asset)
    dof_states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    targets = np.zeros(len(dof_names), dtype=np.float32)

    for i, name in enumerate(dof_names):
        if use_ik_seed and name in IK_ARM_SEED:
            default_pos = IK_ARM_SEED[name]
        else:
            default_pos = B2_Z1_DEFAULT_JOINTS.get(name, 0.0)
        dof_states["pos"][i] = default_pos
        targets[i] = default_pos

        if name in ARM_GAINS:
            if use_ik_seed:
                # Responsive and well damped for the 60 Hz interactive servo.
                stiffness, damping = 400.0, 20.0
            else:
                stiffness, damping = ARM_GAINS[name]
        else:
            stiffness, damping = 250.0, 5.0

        dof_props["driveMode"][i] = DOF_MODE_POS
        dof_props["stiffness"][i] = stiffness
        dof_props["damping"][i] = damping

    gym.set_actor_dof_properties(env, actor, dof_props)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, targets)
    return targets


def main():
    args = parse_args()
    asset_path = resolve_asset_path(args.asset_file)
    ee_goal_cfg = B2Z1RoughCfg.goal_ee
    ee_range_geometry = None
    if args.show_ee_range:
        ee_range_geometry = make_ee_sampling_geometry(ee_goal_cfg, args.ee_samples)

    load_path = asset_path
    temporary_asset_path = None
    collision_visual_count = None
    use_native_collision_render = False

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

    if args.show_collisions:
        if asset_path.suffix.lower() == ".urdf":
            temporary_asset_path, collision_visual_count = make_collision_visual_urdf(
                asset_path
            )
            load_path = temporary_asset_path
        else:
            # MJCF/XML assets use Isaac Gym's native collision-render path.
            use_native_collision_render = True

    asset_root = str(load_path.parent)
    asset_file = load_path.name
    try:
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    finally:
        if temporary_asset_path is not None:
            temporary_asset_path.unlink(missing_ok=True)
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
    dof_targets = set_joint_targets(
        gym,
        env,
        actor,
        asset,
        dof_props,
        use_ik_seed=args.mouse_ik and not args.headless,
    )

    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    base_body_index = body_names.index(B2Z1RoughCfg.asset.base_name)
    gripper_body_index = body_names.index(B2Z1RoughCfg.asset.gripper_name)
    arm_dof_indices = np.array(
        [dof_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int64
    )
    print(f"Loaded asset: {asset_path}")
    print(f"Rigid bodies: {len(body_names)}")
    print(f"DOFs: {len(dof_names)}")
    print(
        "Geometry view:",
        "collision" if args.show_collisions else "visual",
        (
            f"({collision_visual_count} collision shapes)"
            if collision_visual_count is not None
            else ""
        ),
    )
    if args.print_names:
        print("Body names:", body_names)
        print("DOF names:", dof_names)
    if args.show_ee_range:
        print("EE goal sampling range: task=b2z1")
        print(
            "  center offset [m]:",
            [
                ee_goal_cfg.sphere_center.x_offset,
                ee_goal_cfg.sphere_center.y_offset,
                ee_goal_cfg.sphere_center.z_invariant_offset,
            ],
        )
        print("  radius [m]:", list(ee_goal_cfg.ranges.pos_l))
        print(
            "  pitch [rad] / [deg]:",
            list(ee_goal_cfg.ranges.pos_p),
            np.degrees(ee_goal_cfg.ranges.pos_p).round(2).tolist(),
        )
        print(
            "  yaw [rad] / [deg]:",
            list(ee_goal_cfg.ranges.pos_y),
            np.degrees(ee_goal_cfg.ranges.pos_y).round(2).tolist(),
        )
        print(
            f"  accepted sample markers: {args.ee_samples} "
            "(green; training trajectory filters applied)"
        )
        print(
            "  cyan=raw spherical bounds, red=collision exclusion, "
            "orange=minimum local z, yellow=center"
        )

    gym.prepare_sim(sim)

    viewer = None
    mouse_ik = None
    jacobian_descriptor = None
    jacobian = None
    target_marker = None
    current_marker = None
    gripper_handle = None
    jacobian_body_index = None
    jacobian_dof_columns = None
    arm_dof_lower = None
    arm_dof_upper = None
    if not args.headless:
        viewer_props = gymapi.CameraProperties()
        viewer_props.use_collision_geometry = use_native_collision_render
        viewer = gym.create_viewer(sim, viewer_props)
        if viewer is None:
            raise RuntimeError("Failed to create viewer")
        gym.viewer_camera_look_at(
            viewer,
            None,
            gymapi.Vec3(*args.camera_pos),
            gymapi.Vec3(*args.camera_lookat),
        )
        if args.mouse_ik:
            from isaacgym import gymtorch

            mouse_ik = MouseIKController(
                ee_goal_cfg, viewer_props.horizontal_fov
            )
            mouse_ik.subscribe(gym, viewer)
            jacobian_descriptor = gym.acquire_jacobian_tensor(sim, "b2_z1")
            jacobian = gymtorch.wrap_tensor(jacobian_descriptor)
            gripper_handle = gym.find_actor_rigid_body_handle(
                env, actor, B2Z1RoughCfg.asset.gripper_name
            )
            jacobian_body_index = (
                gripper_body_index - 1 if args.fix_base else gripper_body_index
            )
            jacobian_dof_columns = (
                arm_dof_indices if args.fix_base else arm_dof_indices + 6
            ).tolist()
            arm_dof_lower = np.asarray(dof_props["lower"])[arm_dof_indices]
            arm_dof_upper = np.asarray(dof_props["upper"])[arm_dof_indices]
            target_marker = gymutil.WireframeSphereGeometry(
                0.025, 12, 12, None, color=(1.0, 0.0, 1.0)
            )
            current_marker = gymutil.WireframeSphereGeometry(
                0.018, 8, 8, None, color=(0.1, 0.3, 1.0)
            )
            print("Mouse IK control enabled:")
            print("  left-drag: move the magenta EE target in the view plane")
            print("  mouse wheel: change target radius r by 0.03 m")
            print("  target is clamped to b2z1 r/p/y limits and IK drives joint1..6")

    step = 0
    current_ee_world = None
    target_ee_world = None
    ik_position_error = None
    while True:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        if args.steps > 0 and step >= args.steps:
            break

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        anchor_pose = None
        if viewer is not None and (
            ee_range_geometry is not None or mouse_ik is not None
        ):
            anchor_pose = get_ee_range_anchor_pose(
                gym, env, actor, base_body_index
            )

        if mouse_ik is not None:
            mouse_ik.process_events(gym, viewer, env, anchor_pose)
            target_ee_world = mouse_ik.world_target(anchor_pose)
            current_ee_world, ik_position_error = update_arm_position_ik(
                gym,
                sim,
                env,
                actor,
                gripper_handle,
                jacobian,
                jacobian_body_index,
                jacobian_dof_columns,
                arm_dof_indices,
                dof_targets,
                arm_dof_lower,
                arm_dof_upper,
                target_ee_world,
            )

        gym.step_graphics(sim)

        if viewer is not None:
            if ee_range_geometry is not None or mouse_ik is not None:
                gym.clear_lines(viewer)
            if ee_range_geometry is not None:
                gymutil.draw_lines(
                    ee_range_geometry, gym, viewer, env, anchor_pose
                )
            if mouse_ik is not None:
                target_pose = gymapi.Transform()
                target_pose.p = gymapi.Vec3(*target_ee_world)
                gymutil.draw_lines(
                    target_marker, gym, viewer, env, target_pose
                )
                current_pose = gymapi.Transform()
                current_pose.p = gymapi.Vec3(*current_ee_world)
                gymutil.draw_lines(
                    current_marker, gym, viewer, env, current_pose
                )
                gymutil.draw_line(
                    current_pose.p,
                    target_pose.p,
                    gymapi.Vec3(1.0, 1.0, 0.0),
                    gym,
                    viewer,
                    env,
                )
            gym.draw_viewer(viewer, sim, use_native_collision_render)
            gym.sync_frame_time(sim)

        step += 1

    if viewer is not None:
        gym.destroy_viewer(viewer)
    if mouse_ik is not None:
        print(
            "Final EE target [r, p, y]:",
            np.round(mouse_ik.target_sphere, 4).tolist(),
        )
        if ik_position_error is not None:
            final_dof_pos = gym.get_actor_dof_states(
                env, actor, gymapi.STATE_POS
            )["pos"]
            print(
                "Final EE current/target world [m]:",
                np.round(current_ee_world, 4).tolist(),
                np.round(target_ee_world, 4).tolist(),
            )
            print(f"Final IK position error [m]: {ik_position_error:.4f}")
            print(
                "Final arm joint positions [rad]:",
                np.round(final_dof_pos[arm_dof_indices], 4).tolist(),
            )
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
