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

# The target sphere is centered at [0.3, 0.0, 0.7] in a level base frame.
# The Z1 link00 mount is [0.3, 0.0, 0.09] in the same frame.
SPHERE_CENTER_BODY = torch.tensor([0.3, 0.0, 0.7], dtype=torch.float32)
ARM_MOUNT_BODY = torch.tensor([0.3, 0.0, 0.09], dtype=torch.float32)

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
        "--device",
        default="auto",
        help="PyTorch device, e.g. auto, cpu, cuda, or cuda:1 (default: auto).",
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


def plot_workspace(
    xyz: np.ndarray,
    success: np.ndarray,
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
    set_axes_equal_3d(axis_3d, xyz_plot)
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
        f"solvability={rate:.2%}, seeds={args.seeds}, {orientation_description}",
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lower, upper = load_joint_limits(urdf_path, ARM_JOINT_NAMES)
    joint_seeds = make_joint_seeds(lower, upper, args.seeds, args.random_seed)
    radius, pitch, yaw, sphere_xyz = sample_spherical_grid(args)
    target_body = sphere_xyz + SPHERE_CENTER_BODY.unsqueeze(0)
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
    plot_workspace(plot_xyz, arrays["success"], args, figure_path)

    np.savez_compressed(
        data_path,
        **arrays,
        joint_names=np.asarray(ARM_JOINT_NAMES),
        joint_lower=lower.numpy(),
        joint_upper=upper.numpy(),
        joint_seeds=joint_seeds.numpy(),
    )

    reachable_count = int(np.count_nonzero(arrays["success"]))
    total_count = int(arrays["success"].size)
    summary = {
        "total_points": total_count,
        "reachable_points": reachable_count,
        "unresolved_points": total_count - reachable_count,
        "solvability_rate": reachable_count / max(total_count, 1),
        "device": str(device),
        "orientation_mode": args.orientation_mode,
        "orientation_delta_rpy": list(args.orientation_delta),
        "fixed_rpy": list(args.fixed_rpy),
        "ik_seeds": args.seeds,
        "ik_max_iterations": args.max_iterations,
        "ik_error_tolerance": args.error_tolerance,
        "plot_frame": args.plot_frame,
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
