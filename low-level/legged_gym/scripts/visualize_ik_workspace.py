#!/usr/bin/env python3
"""Visualize reachable and unreachable B1+Z1 end-effector targets.

The sampled position range matches ``B1Z1RoughCfg.goal_ee.ranges``:

    radius in [0.4, 0.95] m
    pitch  in [-pi / 2.5, pi / 3] rad
    yaw    in [-1.2, 1.2] rad

IK feasibility is pose-dependent.  By default this script uses the nominal
end-effector orientation constructed by ``ManipLoco._update_curr_ee_goal``
with zero random orientation delta.  Multiple joint-space seeds are tested to
reduce false negatives caused by a local numerical IK solve.

Green points have at least one converged IK seed.  Red points have no
converged seed under the selected solver settings; red therefore means
"numerically unresolved for the tested pose and seeds", not a mathematical
proof that no IK solution exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
LOW_LEVEL_ROOT = SCRIPT_PATH.parents[2]
if str(LOW_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.controllers.pfg_kinematics import (  # noqa: E402
    PFGKinematics,
    PFGKinematicsConfig,
)


ARM_JOINT_NAMES: Tuple[str, ...] = (
    "z1_waist",
    "z1_shoulder",
    "z1_elbow",
    "z1_wrist_angle",
    "z1_forearm_roll",
    "z1_wrist_rotate",
)

# Matches B1Z1RoughCfg.init_state.default_joint_angles.
DEFAULT_ARM_Q = torch.tensor(
    [0.0, 1.48, -0.63, -0.84, 0.0, 1.57],
    dtype=torch.float32,
)

# Matches B1Z1RoughCfg.init_state.default_joint_angles.  These values are used
# only to draw one recognizable reference pose; they do not affect IK results.
DEFAULT_ROBOT_Q: Dict[str, float] = {
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

TRUNK_SIZE = np.asarray([0.647, 0.3, 0.15], dtype=np.float32)

# The target sphere center is [base_x + 0.3, base_y, 0.7] in the world frame:
# its z coordinate is terrain-invariant, not relative to the floating base.
SPHERE_CENTER_WORLD = torch.tensor([0.3, 0.0, 0.7], dtype=torch.float32)
# Z1 link00 is fixed to this location in the floating-base/body frame.
ARM_MOUNT_BODY = torch.tensor([0.3, 0.0, 0.09], dtype=torch.float32)
COORDINATE_CONVENTION_VERSION = 2

DEFAULT_URDF = (
    LOW_LEVEL_ROOT / "resources" / "robots" / "b1z1" / "urdf" / "b1z1.urdf"
)
DEFAULT_OUTPUT_DIR = LOW_LEVEL_ROOT / "logs" / "ik_workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the configured spherical EE target range, run batched "
            "multi-start IK, and plot reachable points in green and unresolved "
            "points in red."
        )
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help=(
            "Only redraw output-dir/ik_workspace.npz; skip IK. Useful when "
            "changing the view or robot overlay."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, e.g. auto, cpu, cuda, or cuda:1 (default: auto).",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.5,
        help=(
            "Level floating-base height above terrain in metres. The default "
            "matches B1Z1RoughCfg.init_state.pos[2] (default: 0.5)."
        ),
    )

    parser.add_argument("--radius-min", type=float, default=0.4)
    parser.add_argument("--radius-max", type=float, default=0.95)
    parser.add_argument("--pitch-min", type=float, default=-math.pi / 2.5)
    parser.add_argument("--pitch-max", type=float, default=math.pi / 3.0)
    parser.add_argument("--yaw-min", type=float, default=-1.2)
    parser.add_argument("--yaw-max", type=float, default=1.2)
    parser.add_argument("--n-radius", type=int, default=18)
    parser.add_argument("--n-pitch", type=int, default=24)
    parser.add_argument("--n-yaw", type=int, default=32)

    parser.add_argument(
        "--orientation-mode",
        choices=("nominal", "fixed"),
        default="nominal",
        help=(
            "nominal reproduces the environment orientation as a function of "
            "target pitch/yaw; fixed uses --fixed-rpy for every point."
        ),
    )
    parser.add_argument(
        "--orientation-delta",
        type=float,
        nargs=3,
        metavar=("DROLL", "DPITCH", "DYAW"),
        default=(0.0, 0.0, 0.0),
        help="RPY delta added in nominal mode (radians).",
    )
    parser.add_argument(
        "--fixed-rpy",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(math.pi / 2.0, 0.38, 0.0),
        help="Fixed target RPY in the level base frame (radians).",
    )
    parser.add_argument(
        "--arm-induced-pitch",
        type=float,
        default=0.38,
        help="Nominal environment arm_induced_pitch value.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="Number of joint-space IK initializations per target.",
    )
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--error-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--damping-delta", type=float, default=1.0e-4)
    parser.add_argument("--joint-limit-margin", type=float, default=1.0e-4)
    parser.add_argument("--max-joint-step", type=float, default=0.25)

    parser.add_argument(
        "--plot-frame",
        choices=("arm", "base", "sphere"),
        default="arm",
        help="Coordinate frame used in the figure (IK always runs in base frame).",
    )
    parser.add_argument("--max-plot-points", type=int, default=50000)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-58.0)
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Do not overlay the B1+Z1 nominal pose on the figure.",
    )
    parser.add_argument(
        "--robot-opacity",
        type=float,
        default=0.9,
        help="Opacity of the B1+Z1 overlay (default: 0.9).",
    )
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.radius_min <= 0.0 or args.radius_min >= args.radius_max:
        raise ValueError("Require 0 < radius-min < radius-max")
    if args.pitch_min >= args.pitch_max or args.yaw_min >= args.yaw_max:
        raise ValueError("Pitch/yaw lower bounds must be below upper bounds")
    for name in ("n_radius", "n_pitch", "n_yaw"):
        if getattr(args, name) < 2:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 2")
    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1")
    if args.batch_size < 1 or args.max_iterations < 1:
        raise ValueError("--batch-size and --max-iterations must be positive")
    if args.error_tolerance <= 0.0 or args.damping_delta <= 0.0:
        raise ValueError("IK tolerance and damping must be positive")
    if not math.isfinite(args.base_height) or args.base_height <= 0.0:
        raise ValueError("--base-height must be a positive finite value")
    if not 0.0 <= args.robot_opacity <= 1.0:
        raise ValueError("--robot-opacity must be between 0 and 1")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def load_joint_limits(
    urdf_path: Path,
    joint_names: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read configured arm joint limits without requiring a simulator."""

    urdf_text = urdf_path.read_text(encoding="utf-8").replace(
        'xyz="0.3 0 0.09>>"',
        'xyz="0.3 0 0.09"',
    )
    root = ET.fromstring(urdf_text)
    elements: Dict[str, ET.Element] = {
        element.attrib.get("name", ""): element for element in root.findall("joint")
    }

    lower = []
    upper = []
    for name in joint_names:
        if name not in elements:
            raise RuntimeError(f"Arm joint is absent from URDF: {name}")
        limit = elements[name].find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise RuntimeError(f"Arm joint has no finite limits in URDF: {name}")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))

    return torch.tensor(lower, dtype=torch.float32), torch.tensor(
        upper, dtype=torch.float32
    )


def _parse_xyz(text: str | None, default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    # The supplied B1+Z1 URDF contains one harmless trailing ">>" typo.
    cleaned = text.replace(">", " ")
    values = [float(value) for value in cleaned.split()]
    if len(values) != 3:
        raise ValueError(f"Expected a 3-vector, got: {text!r}")
    return np.asarray(values, dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation_x = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    rotation_y = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
    rotation_z = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    return rotation_z @ rotation_y @ rotation_x


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12 or abs(angle) < 1.0e-12:
        return np.eye(3)
    x, y, z = axis / norm
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        (
            (
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ),
            (
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ),
            (
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ),
        ),
        dtype=np.float64,
    )


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def load_robot_schematic(
    urdf_path: Path,
    plot_frame: str,
    base_height: float,
) -> Dict[str, np.ndarray]:
    """Build a lightweight B1+Z1 drawing from URDF joint transforms."""

    urdf_text = urdf_path.read_text(encoding="utf-8").replace(
        'xyz="0.3 0 0.09>>"',
        'xyz="0.3 0 0.09"',
    )
    root = ET.fromstring(urdf_text)
    children: Dict[str, list[ET.Element]] = {}
    child_links = set()
    all_links = {link.attrib["name"] for link in root.findall("link")}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib["link"]
        child_name = child.attrib["link"]
        children.setdefault(parent_name, []).append(joint)
        child_links.add(child_name)

    root_links = sorted(all_links - child_links)
    if "base" in root_links:
        root_link = "base"
    elif len(root_links) == 1:
        root_link = root_links[0]
    else:
        raise RuntimeError(f"Could not identify one URDF root link: {root_links}")

    link_transforms: Dict[str, np.ndarray] = {root_link: np.eye(4)}
    leg_segments = []
    arm_segments = []
    pending = [root_link]
    while pending:
        parent_name = pending.pop()
        parent_transform = link_transforms[parent_name]
        for joint in children.get(parent_name, []):
            child_name = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            xyz = _parse_xyz(
                None if origin is None else origin.attrib.get("xyz"),
                (0.0, 0.0, 0.0),
            )
            rpy = _parse_xyz(
                None if origin is None else origin.attrib.get("rpy"),
                (0.0, 0.0, 0.0),
            )
            origin_transform = _homogeneous(_rpy_matrix(rpy), xyz)

            motion_transform = np.eye(4)
            if joint.attrib.get("type") in ("revolute", "continuous"):
                axis_element = joint.find("axis")
                axis = _parse_xyz(
                    None if axis_element is None else axis_element.attrib.get("xyz"),
                    (1.0, 0.0, 0.0),
                )
                angle = DEFAULT_ROBOT_Q.get(joint.attrib.get("name", ""), 0.0)
                motion_transform[:3, :3] = _axis_angle_matrix(axis, angle)

            child_transform = parent_transform @ origin_transform @ motion_transform
            link_transforms[child_name] = child_transform
            pending.append(child_name)

            segment = np.stack(
                (parent_transform[:3, 3], child_transform[:3, 3]),
                axis=0,
            )
            joint_name = joint.attrib.get("name", "")
            if joint_name.startswith(("FL_", "FR_", "RL_", "RR_")):
                leg_segments.append(segment)
            elif joint_name.startswith("z1_") or joint_name in (
                "base_static_joint",
                "ee_gripper",
            ):
                arm_segments.append(segment)

    required_links = ("trunk", "link00", "ee_gripper_link")
    missing = [name for name in required_links if name not in link_transforms]
    if missing:
        raise RuntimeError(f"Robot schematic is missing URDF links: {missing}")

    if plot_frame == "arm":
        frame_origin_body = ARM_MOUNT_BODY.numpy().astype(np.float64)
    elif plot_frame == "sphere":
        frame_origin_body = SPHERE_CENTER_WORLD.numpy().astype(np.float64)
        frame_origin_body[2] -= base_height
    else:
        frame_origin_body = np.zeros(3, dtype=np.float64)

    leg_array = np.asarray(leg_segments, dtype=np.float64) - frame_origin_body
    arm_array = np.asarray(arm_segments, dtype=np.float64) - frame_origin_body
    feet = np.stack(
        [link_transforms[f"{prefix}_foot"][:3, 3] for prefix in ("FL", "FR", "RL", "RR")]
    ) - frame_origin_body
    body_center = link_transforms["trunk"][:3, 3] - frame_origin_body
    arm_base = link_transforms["link00"][:3, 3] - frame_origin_body
    end_effector = link_transforms["ee_gripper_link"][:3, 3] - frame_origin_body

    return {
        "leg_segments": leg_array,
        "arm_segments": arm_array,
        "feet": feet,
        "body_center": body_center,
        "body_size": TRUNK_SIZE.astype(np.float64),
        "arm_base": arm_base,
        "end_effector": end_effector,
    }


def sample_spherical_grid(
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    radius = torch.linspace(args.radius_min, args.radius_max, args.n_radius)
    pitch = torch.linspace(args.pitch_min, args.pitch_max, args.n_pitch)
    yaw = torch.linspace(args.yaw_min, args.yaw_max, args.n_yaw)
    radius_grid, pitch_grid, yaw_grid = torch.meshgrid(
        radius,
        pitch,
        yaw,
        indexing="ij",
    )
    radius_flat = radius_grid.reshape(-1)
    pitch_flat = pitch_grid.reshape(-1)
    yaw_flat = yaw_grid.reshape(-1)

    cosine_pitch = torch.cos(pitch_flat)
    sphere_xyz = torch.stack(
        (
            radius_flat * cosine_pitch * torch.cos(yaw_flat),
            radius_flat * cosine_pitch * torch.sin(yaw_flat),
            radius_flat * torch.sin(pitch_flat),
        ),
        dim=-1,
    )
    return radius_flat, pitch_flat, yaw_flat, sphere_xyz


def euler_xyz_to_quaternion_xyzw(rpy: torch.Tensor) -> torch.Tensor:
    """Match Isaac Gym's quat_from_euler_xyz convention."""

    roll, pitch, yaw = rpy.unbind(dim=-1)
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    sr, cr = torch.sin(half_roll), torch.cos(half_roll)
    sp, cp = torch.sin(half_pitch), torch.cos(half_pitch)
    sy, cy = torch.sin(half_yaw), torch.cos(half_yaw)
    return torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )


def build_target_quaternions(
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.orientation_mode == "nominal":
        delta_roll, delta_pitch, delta_yaw = args.orientation_delta
        roll_target = torch.full_like(pitch, math.pi / 2.0 + delta_roll)
        pitch_target = -pitch + args.arm_induced_pitch + delta_pitch
        yaw_target = yaw + delta_yaw
    else:
        fixed_roll, fixed_pitch, fixed_yaw = args.fixed_rpy
        roll_target = torch.full_like(pitch, fixed_roll)
        pitch_target = torch.full_like(pitch, fixed_pitch)
        yaw_target = torch.full_like(pitch, fixed_yaw)

    return euler_xyz_to_quaternion_xyzw(
        torch.stack((roll_target, pitch_target, yaw_target), dim=-1)
    )


def make_joint_seeds(
    lower: torch.Tensor,
    upper: torch.Tensor,
    count: int,
    random_seed: int,
) -> torch.Tensor:
    default = torch.maximum(torch.minimum(DEFAULT_ARM_Q, upper), lower).unsqueeze(0)
    if count == 1:
        return default

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    random_unit = torch.rand((count - 1, lower.numel()), generator=generator)
    random_q = lower.unsqueeze(0) + random_unit * (upper - lower).unsqueeze(0)
    return torch.cat((default, random_q), dim=0)


@torch.no_grad()
def solve_workspace(
    kinematics: PFGKinematics,
    target_position_body: torch.Tensor,
    target_quaternion_body: torch.Tensor,
    joint_seeds: torch.Tensor,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    point_count = target_position_body.shape[0]
    seed_count = joint_seeds.shape[0]
    device = kinematics.device

    success_all = torch.zeros(point_count, dtype=torch.bool)
    min_energy_all = torch.full((point_count,), float("inf"), dtype=torch.float32)
    min_position_error_all = torch.full_like(min_energy_all, float("inf"))
    min_rotation_error_all = torch.full_like(min_energy_all, float("inf"))

    for start in range(0, point_count, batch_size):
        stop = min(start + batch_size, point_count)
        points_in_batch = stop - start

        positions = target_position_body[start:stop].to(device)
        quaternions = target_quaternion_body[start:stop].to(device)
        repeated_positions = positions.repeat_interleave(seed_count, dim=0)
        repeated_quaternions = quaternions.repeat_interleave(seed_count, dim=0)
        repeated_seeds = joint_seeds.to(device).repeat(points_in_batch, 1)

        _, success, energy, position_error, rotation_error = kinematics.solve(
            target_position_body=repeated_positions,
            target_quaternion_body_xyzw=repeated_quaternions,
            current_arm_q=repeated_seeds,
            previous_solution=None,
        )

        success = success.reshape(points_in_batch, seed_count)
        energy = energy.reshape(points_in_batch, seed_count)
        position_error = position_error.reshape(points_in_batch, seed_count)
        rotation_error = rotation_error.reshape(points_in_batch, seed_count)
        best_seed = torch.argmin(energy, dim=1)
        row = torch.arange(points_in_batch, device=device)

        success_all[start:stop] = torch.any(success, dim=1).cpu()
        min_energy_all[start:stop] = energy[row, best_seed].cpu()
        min_position_error_all[start:stop] = position_error[row, best_seed].cpu()
        min_rotation_error_all[start:stop] = rotation_error[row, best_seed].cpu()

        print(
            f"IK batch {stop:>7d}/{point_count} "
            f"({100.0 * stop / point_count:5.1f}%)",
            flush=True,
        )

    return {
        "success": success_all,
        "min_energy": min_energy_all,
        "min_position_error": min_position_error_all,
        "min_rotation_error": min_rotation_error_all,
    }


def choose_plot_points(
    success: np.ndarray,
    maximum: int,
    random_seed: int,
) -> np.ndarray:
    total = success.size
    if maximum <= 0 or total <= maximum:
        return np.arange(total)

    rng = np.random.default_rng(random_seed)
    success_indices = np.flatnonzero(success)
    failure_indices = np.flatnonzero(~success)
    success_budget = min(success_indices.size, maximum // 2)
    failure_budget = min(failure_indices.size, maximum - success_budget)
    remaining = maximum - success_budget - failure_budget
    if remaining > 0:
        success_extra = min(success_indices.size - success_budget, remaining)
        success_budget += success_extra
        remaining -= success_extra
        failure_budget += min(failure_indices.size - failure_budget, remaining)

    selected_success = rng.choice(success_indices, success_budget, replace=False)
    selected_failure = rng.choice(failure_indices, failure_budget, replace=False)
    return np.concatenate((selected_success, selected_failure))


def set_axes_equal_3d(axis, xyz: np.ndarray) -> None:
    minima = np.min(xyz, axis=0)
    maxima = np.max(xyz, axis=0)
    center = 0.5 * (minima + maxima)
    radius = 0.5 * float(np.max(maxima - minima))
    radius = max(radius, 1.0e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    try:
        axis.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def _cuboid_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half_size = 0.5 * size
    signs = np.asarray(
        (
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ),
        dtype=np.float64,
    )
    return center + signs * half_size


def _robot_extent_points(robot: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            robot["leg_segments"].reshape(-1, 3),
            robot["arm_segments"].reshape(-1, 3),
            robot["feet"],
            _cuboid_corners(robot["body_center"], robot["body_size"]),
            robot["end_effector"].reshape(1, 3),
        ),
        axis=0,
    )


def draw_robot_3d(axis, robot: Dict[str, np.ndarray], opacity: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    body_color = "#25364a"
    leg_color = "#334e68"
    arm_color = "#f0a202"
    joint_color = "#e6edf3"

    corners = _cuboid_corners(robot["body_center"], robot["body_size"])
    face_indices = (
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    )
    body_faces = [[corners[index] for index in face] for face in face_indices]
    body = Poly3DCollection(
        body_faces,
        facecolor=body_color,
        edgecolor="#0b1726",
        linewidth=0.8,
        alpha=0.82 * opacity,
    )
    axis.add_collection3d(body)

    for index, segment in enumerate(robot["leg_segments"]):
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=leg_color,
            linewidth=7.0,
            alpha=opacity,
            solid_capstyle="round",
            label="B1+Z1 nominal pose" if index == 0 else None,
            zorder=20,
        )
    for segment in robot["arm_segments"]:
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=arm_color,
            linewidth=7.0,
            alpha=opacity,
            solid_capstyle="round",
            zorder=21,
        )

    leg_joints = robot["leg_segments"].reshape(-1, 3)
    arm_joints = robot["arm_segments"].reshape(-1, 3)
    axis.scatter(
        leg_joints[:, 0],
        leg_joints[:, 1],
        leg_joints[:, 2],
        c=joint_color,
        edgecolors=leg_color,
        linewidths=0.8,
        s=22,
        alpha=opacity,
        depthshade=False,
        zorder=22,
    )
    axis.scatter(
        arm_joints[:, 0],
        arm_joints[:, 1],
        arm_joints[:, 2],
        c=joint_color,
        edgecolors=arm_color,
        linewidths=0.8,
        s=24,
        alpha=opacity,
        depthshade=False,
        zorder=23,
    )
    axis.scatter(
        robot["feet"][:, 0],
        robot["feet"][:, 1],
        robot["feet"][:, 2],
        c=leg_color,
        edgecolors="white",
        linewidths=0.7,
        s=58,
        alpha=opacity,
        depthshade=False,
        zorder=24,
    )
    end_effector = robot["end_effector"]
    axis.scatter(
        [end_effector[0]],
        [end_effector[1]],
        [end_effector[2]],
        c=arm_color,
        edgecolors="black",
        linewidths=0.7,
        marker="*",
        s=105,
        alpha=opacity,
        depthshade=False,
        zorder=25,
    )


def draw_robot_projection(
    axis,
    robot: Dict[str, np.ndarray],
    dimensions: Tuple[int, int],
    opacity: float,
) -> None:
    from matplotlib.patches import Rectangle

    body_color = "#25364a"
    leg_color = "#334e68"
    arm_color = "#f0a202"
    first_dimension, second_dimension = dimensions
    body_center = robot["body_center"]
    body_size = robot["body_size"]
    rectangle = Rectangle(
        (
            body_center[first_dimension] - 0.5 * body_size[first_dimension],
            body_center[second_dimension] - 0.5 * body_size[second_dimension],
        ),
        body_size[first_dimension],
        body_size[second_dimension],
        facecolor=body_color,
        edgecolor="#0b1726",
        linewidth=1.0,
        alpha=0.82 * opacity,
        zorder=20,
    )
    axis.add_patch(rectangle)

    for segment in robot["leg_segments"]:
        axis.plot(
            segment[:, first_dimension],
            segment[:, second_dimension],
            color=leg_color,
            linewidth=6.0,
            alpha=opacity,
            solid_capstyle="round",
            zorder=21,
        )
    for segment in robot["arm_segments"]:
        axis.plot(
            segment[:, first_dimension],
            segment[:, second_dimension],
            color=arm_color,
            linewidth=6.0,
            alpha=opacity,
            solid_capstyle="round",
            zorder=22,
        )

    leg_joints = robot["leg_segments"].reshape(-1, 3)
    arm_joints = robot["arm_segments"].reshape(-1, 3)
    axis.scatter(
        leg_joints[:, first_dimension],
        leg_joints[:, second_dimension],
        c="#e6edf3",
        edgecolors=leg_color,
        linewidths=0.6,
        s=18,
        alpha=opacity,
        zorder=23,
    )
    axis.scatter(
        arm_joints[:, first_dimension],
        arm_joints[:, second_dimension],
        c="#e6edf3",
        edgecolors=arm_color,
        linewidths=0.6,
        s=20,
        alpha=opacity,
        zorder=24,
    )
    axis.scatter(
        robot["feet"][:, first_dimension],
        robot["feet"][:, second_dimension],
        c=leg_color,
        edgecolors="white",
        linewidths=0.6,
        s=42,
        alpha=opacity,
        zorder=25,
    )
    end_effector = robot["end_effector"]
    axis.scatter(
        [end_effector[first_dimension]],
        [end_effector[second_dimension]],
        c=arm_color,
        edgecolors="black",
        linewidths=0.6,
        marker="*",
        s=80,
        alpha=opacity,
        zorder=26,
    )


def plot_workspace(
    xyz: np.ndarray,
    success: np.ndarray,
    robot: Dict[str, np.ndarray] | None,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = choose_plot_points(success, args.max_plot_points, args.random_seed)
    xyz_plot = xyz[selected]
    success_plot = success[selected]
    green = "#2ca02c"
    red = "#d62728"

    figure = plt.figure(figsize=(17, 6), constrained_layout=True)
    axis_3d = figure.add_subplot(1, 3, 1, projection="3d")

    # Draw failures first so reachable points remain visible at their boundary.
    for mask, color, label, alpha in (
        (~success_plot, red, "IK unresolved", 0.28),
        (success_plot, green, "IK reachable", 0.58),
    ):
        axis_3d.scatter(
            xyz_plot[mask, 0],
            xyz_plot[mask, 1],
            xyz_plot[mask, 2],
            c=color,
            s=args.point_size,
            alpha=alpha,
            linewidths=0,
            label=label,
        )

    axis_3d.scatter([0.0], [0.0], [0.0], c="black", s=38, marker="x", label="frame origin")
    axis_3d.set_xlabel("x [m]")
    axis_3d.set_ylabel("y [m]")
    axis_3d.set_zlabel("z [m]")
    axis_3d.view_init(elev=args.elevation, azim=args.azimuth)
    if robot is not None:
        draw_robot_3d(axis_3d, robot, args.robot_opacity)
        extent_xyz = np.concatenate((xyz_plot, _robot_extent_points(robot)), axis=0)
    else:
        extent_xyz = xyz_plot
    set_axes_equal_3d(axis_3d, extent_xyz)
    axis_3d.legend(loc="upper left", fontsize=8)

    for subplot_index, dimensions, labels, title in (
        (2, (0, 1), ("x [m]", "y [m]"), "XY projection"),
        (3, (0, 2), ("x [m]", "z [m]"), "XZ projection"),
    ):
        axis = figure.add_subplot(1, 3, subplot_index)
        axis.scatter(
            xyz_plot[~success_plot, dimensions[0]],
            xyz_plot[~success_plot, dimensions[1]],
            c=red,
            s=args.point_size,
            alpha=0.25,
            linewidths=0,
        )
        axis.scatter(
            xyz_plot[success_plot, dimensions[0]],
            xyz_plot[success_plot, dimensions[1]],
            c=green,
            s=args.point_size,
            alpha=0.55,
            linewidths=0,
        )
        axis.scatter([0.0], [0.0], c="black", s=30, marker="x")
        if robot is not None:
            draw_robot_projection(axis, robot, dimensions, args.robot_opacity)
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")

    reachable_count = int(np.count_nonzero(success))
    total_count = int(success.size)
    rate = reachable_count / max(total_count, 1)
    orientation_description = (
        "environment nominal orientation"
        if args.orientation_mode == "nominal"
        else f"fixed RPY={tuple(args.fixed_rpy)}"
    )
    figure.suptitle(
        "B1+Z1 sampled IK workspace "
        f"({args.plot_frame} frame)\n"
        f"green={reachable_count:,}, red={total_count - reachable_count:,}, "
        f"solvability={rate:.2%}, seeds={args.seeds}, "
        f"level base h={args.base_height:.2f} m, {orientation_description}",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=220)
    if args.show:
        plt.show()
    plt.close(figure)


def save_csv(
    path: Path,
    radius: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    target_body: np.ndarray,
    target_arm: np.ndarray,
    results: Dict[str, np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            (
                "radius_m",
                "pitch_rad",
                "yaw_rad",
                "body_x_m",
                "body_y_m",
                "body_z_m",
                "arm_x_m",
                "arm_y_m",
                "arm_z_m",
                "ik_success",
                "min_energy",
                "min_position_error_m",
                "min_rotation_error_rad",
            )
        )
        for index in range(radius.size):
            writer.writerow(
                (
                    radius[index],
                    pitch[index],
                    yaw[index],
                    *target_body[index].tolist(),
                    *target_arm[index].tolist(),
                    int(results["success"][index]),
                    results["min_energy"][index],
                    results["min_position_error"][index],
                    results["min_rotation_error"][index],
                )
            )


def redraw_saved_workspace(
    args: argparse.Namespace,
    urdf_path: Path,
    output_dir: Path,
) -> None:
    data_path = output_dir / "ik_workspace.npz"
    figure_path = output_dir / "ik_workspace.png"
    summary_path = output_dir / "ik_workspace_summary.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Saved IK data not found: {data_path}")

    required_keys = (
        "radius",
        "pitch",
        "yaw",
        "target_xyz_body",
        "target_xyz_arm",
        "sphere_xyz",
        "success",
        "min_energy",
        "min_position_error",
        "min_rotation_error",
    )
    with np.load(data_path, allow_pickle=False) as saved:
        metadata_keys = ("coordinate_convention_version", "base_height_m")
        missing_metadata = [key for key in metadata_keys if key not in saved.files]
        if missing_metadata:
            raise RuntimeError(
                "Saved IK data uses the old terrain/body z convention and its "
                "success labels cannot be reused. Rerun without --reuse-data."
            )
        convention_version = int(saved["coordinate_convention_version"])
        if convention_version != COORDINATE_CONVENTION_VERSION:
            raise RuntimeError(
                f"Unsupported saved coordinate convention {convention_version}; "
                "rerun without --reuse-data."
            )
        # Reuse must preserve the torso pose used during IK classification.
        saved_base_height = float(saved["base_height_m"])
        if not math.isclose(
            args.base_height,
            saved_base_height,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise RuntimeError(
                f"Saved IK used base height {saved_base_height:.6g} m, but "
                f"--base-height is {args.base_height:.6g} m. Reuse with the "
                "saved height or rerun IK without --reuse-data."
            )
        args.base_height = saved_base_height
        missing = [key for key in required_keys if key not in saved.files]
        if missing:
            raise RuntimeError(f"Saved IK data is missing arrays: {missing}")
        arrays = {key: saved[key] for key in required_keys}

    previous_summary = {}
    if summary_path.is_file():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        args.seeds = int(previous_summary.get("ik_seeds", args.seeds))
        saved_orientation_mode = previous_summary.get("orientation_mode")
        if saved_orientation_mode in ("nominal", "fixed"):
            args.orientation_mode = saved_orientation_mode
        if "fixed_rpy" in previous_summary:
            args.fixed_rpy = tuple(previous_summary["fixed_rpy"])

    if args.plot_frame == "arm":
        plot_xyz = arrays["target_xyz_arm"]
    elif args.plot_frame == "base":
        plot_xyz = arrays["target_xyz_body"]
    else:
        plot_xyz = arrays["sphere_xyz"]

    robot = None
    if not args.no_robot:
        robot = load_robot_schematic(
            urdf_path,
            args.plot_frame,
            args.base_height,
        )
    plot_workspace(plot_xyz, arrays["success"], robot, args, figure_path)

    if args.save_csv:
        csv_path = output_dir / "ik_workspace.csv"
        save_csv(
            csv_path,
            arrays["radius"],
            arrays["pitch"],
            arrays["yaw"],
            arrays["target_xyz_body"],
            arrays["target_xyz_arm"],
            arrays,
        )
        print(f"CSV:     {csv_path}")

    success = arrays["success"].astype(bool)
    reachable_count = int(np.count_nonzero(success))
    total_count = int(success.size)
    previous_summary.update(
        {
            "total_points": total_count,
            "reachable_points": reachable_count,
            "unresolved_points": total_count - reachable_count,
            "solvability_rate": reachable_count / max(total_count, 1),
            "coordinate_convention_version": COORDINATE_CONVENTION_VERSION,
            "base_height_m": args.base_height,
            "plot_frame": args.plot_frame,
            "robot_overlay": robot is not None,
            "figure": str(figure_path),
            "data": str(data_path),
        }
    )
    summary_path.write_text(
        json.dumps(previous_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Reused saved IK results: {reachable_count:,}/{total_count:,} reachable")
    print(f"Figure:  {figure_path}")
    print(f"Data:    {data_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_data:
        redraw_saved_workspace(args, urdf_path, output_dir)
        return

    lower, upper = load_joint_limits(urdf_path, ARM_JOINT_NAMES)
    joint_seeds = make_joint_seeds(lower, upper, args.seeds, args.random_seed)
    radius, pitch, yaw, sphere_xyz = sample_spherical_grid(args)
    sphere_center_body = SPHERE_CENTER_WORLD.clone()
    sphere_center_body[2] -= args.base_height
    target_body = sphere_xyz + sphere_center_body.unsqueeze(0)
    target_arm = target_body - ARM_MOUNT_BODY.unsqueeze(0)
    target_quaternion = build_target_quaternions(pitch, yaw, args)

    solver_config = PFGKinematicsConfig(
        max_iterations=args.max_iterations,
        error_tolerance=args.error_tolerance,
        damping_delta=args.damping_delta,
        joint_limit_margin=args.joint_limit_margin,
        max_joint_step=args.max_joint_step,
        pose_error_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    kinematics = PFGKinematics(
        urdf_path=str(urdf_path),
        end_link_name="ee_gripper_link",
        arm_joint_names=ARM_JOINT_NAMES,
        joint_lower_limits=lower,
        joint_upper_limits=upper,
        device=device,
        dtype=torch.float32,
        config=solver_config,
    )

    print(
        f"Sampling {radius.numel():,} target positions; "
        f"testing {args.seeds} IK seeds per target on {device}."
    )
    solved = solve_workspace(
        kinematics=kinematics,
        target_position_body=target_body,
        target_quaternion_body=target_quaternion,
        joint_seeds=joint_seeds,
        batch_size=args.batch_size,
    )

    arrays = {
        "radius": radius.numpy(),
        "pitch": pitch.numpy(),
        "yaw": yaw.numpy(),
        "sphere_xyz": sphere_xyz.numpy(),
        "target_xyz_body": target_body.numpy(),
        "target_xyz_arm": target_arm.numpy(),
        "target_quaternion_body_xyzw": target_quaternion.numpy(),
        "success": solved["success"].numpy(),
        "min_energy": solved["min_energy"].numpy(),
        "min_position_error": solved["min_position_error"].numpy(),
        "min_rotation_error": solved["min_rotation_error"].numpy(),
    }

    if args.plot_frame == "arm":
        plot_xyz = arrays["target_xyz_arm"]
    elif args.plot_frame == "base":
        plot_xyz = arrays["target_xyz_body"]
    else:
        plot_xyz = arrays["sphere_xyz"]

    figure_path = output_dir / "ik_workspace.png"
    data_path = output_dir / "ik_workspace.npz"
    summary_path = output_dir / "ik_workspace_summary.json"
    robot = None
    if not args.no_robot:
        robot = load_robot_schematic(
            urdf_path,
            args.plot_frame,
            args.base_height,
        )
    plot_workspace(plot_xyz, arrays["success"], robot, args, figure_path)

    np.savez_compressed(
        data_path,
        **arrays,
        joint_names=np.asarray(ARM_JOINT_NAMES),
        joint_lower=lower.numpy(),
        joint_upper=upper.numpy(),
        joint_seeds=joint_seeds.numpy(),
        coordinate_convention_version=np.asarray(
            COORDINATE_CONVENTION_VERSION,
            dtype=np.int32,
        ),
        base_height_m=np.asarray(args.base_height, dtype=np.float32),
    )

    reachable_count = int(np.count_nonzero(arrays["success"]))
    total_count = int(arrays["success"].size)
    summary = {
        "total_points": total_count,
        "reachable_points": reachable_count,
        "unresolved_points": total_count - reachable_count,
        "solvability_rate": reachable_count / max(total_count, 1),
        "coordinate_convention_version": COORDINATE_CONVENTION_VERSION,
        "base_height_m": args.base_height,
        "sphere_center_world_m": SPHERE_CENTER_WORLD.tolist(),
        "device": str(device),
        "orientation_mode": args.orientation_mode,
        "orientation_delta_rpy": list(args.orientation_delta),
        "fixed_rpy": list(args.fixed_rpy),
        "ik_seeds": args.seeds,
        "ik_max_iterations": args.max_iterations,
        "ik_error_tolerance": args.error_tolerance,
        "plot_frame": args.plot_frame,
        "robot_overlay": robot is not None,
        "figure": str(figure_path),
        "data": str(data_path),
        "classification_note": (
            "Red means no tested seed converged for the selected target pose; "
            "it is not a proof of global IK infeasibility."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.save_csv:
        csv_path = output_dir / "ik_workspace.csv"
        save_csv(
            csv_path,
            arrays["radius"],
            arrays["pitch"],
            arrays["yaw"],
            arrays["target_xyz_body"],
            arrays["target_xyz_arm"],
            arrays,
        )
        print(f"CSV:     {csv_path}")

    print(f"Reachable: {reachable_count:,}/{total_count:,} ({reachable_count / total_count:.2%})")
    print(f"Figure:  {figure_path}")
    print(f"Data:    {data_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
