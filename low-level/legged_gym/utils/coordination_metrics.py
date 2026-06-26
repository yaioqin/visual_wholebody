import csv
import json
import math
import os
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


_AUTO_QUAT_ORDER_WARNED = False


def safe_get_nested_attr(obj: Any, path: str) -> Any:
    """Return a nested attribute/dict value, or None when any segment is absent."""
    if obj is None or not path:
        return None

    parts = path.split(".")
    candidates = [parts]
    if parts[0] == "env":
        candidates.append(parts[1:])

    for candidate in candidates:
        cur = obj
        ok = True
        for part in candidate:
            if cur is None:
                ok = False
                break
            if isinstance(cur, dict):
                if part not in cur:
                    ok = False
                    break
                cur = cur[part]
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def safe_get_tensor(env: Any, candidates: Iterable[str]) -> Optional[torch.Tensor]:
    """Return the first existing tensor-like env attribute from candidates."""
    for name in candidates:
        value = safe_get_nested_attr(env, name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (np.ndarray, list, tuple)):
            try:
                return torch.as_tensor(value)
            except (TypeError, ValueError):
                continue
    return None


def angle_wrap_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)


def _as_xyzw(q: torch.Tensor, quat_order: str) -> torch.Tensor:
    if quat_order == "wxyz":
        return torch.cat([q[..., 1:], q[..., :1]], dim=-1)
    return q


def quat_geodesic_distance(
    q1: torch.Tensor,
    q2: torch.Tensor,
    quat_order: str = "auto",
) -> torch.Tensor:
    """Quaternion geodesic distance in radians.

    Isaac Gym convention is xyzw. When quat_order="auto" and no stronger
    convention is available, this helper uses xyzw and emits one warning.
    """
    global _AUTO_QUAT_ORDER_WARNED

    if q1 is None or q2 is None:
        raise ValueError("q1 and q2 must not be None")
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError("q1 and q2 must have last dimension 4")

    if quat_order == "auto":
        quat_order = "xyzw"
        if not _AUTO_QUAT_ORDER_WARNED:
            warnings.warn(
                "quat_order='auto' defaulting to Isaac Gym xyzw convention.",
                RuntimeWarning,
            )
            _AUTO_QUAT_ORDER_WARNED = True
    if quat_order not in ("xyzw", "wxyz"):
        raise ValueError("quat_order must be 'auto', 'xyzw', or 'wxyz'")

    q1 = _normalize_quat(_as_xyzw(q1, quat_order))
    q2 = _normalize_quat(_as_xyzw(q2, quat_order))
    dot = torch.abs(torch.sum(q1 * q2, dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        values = x.reshape(-1)
    else:
        mask = mask.bool()
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        values = x[mask.expand_as(x)]
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    return values.mean()


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        device = x.device if isinstance(x, torch.Tensor) else y.device
        return torch.tensor(float("nan"), device=device)
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    n = min(x.numel(), y.numel())
    x = x[:n]
    y = y[:n]
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2:
        return torch.tensor(float("nan"), device=x.device)
    x = x - x.mean()
    y = y - y.mean()
    x_std = torch.sqrt(torch.mean(x * x))
    y_std = torch.sqrt(torch.mean(y * y))
    if x_std.item() < 1.0e-8 or y_std.item() < 1.0e-8:
        return torch.tensor(float("nan"), device=x.device)
    return torch.mean(x * y) / (x_std * y_std)


class RunningMetricAccumulator:
    """GPU-friendly running stats for scalar/vector samples."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.data: Dict[str, Dict[str, torch.Tensor]] = {}

    def add(
        self,
        name: str,
        value: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if value is None:
            return
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device)
        value = value.detach()
        if mask is not None:
            mask = mask.detach().bool().to(value.device)
            while mask.dim() < value.dim():
                mask = mask.unsqueeze(-1)
            value = value[mask.expand_as(value)]
        else:
            value = value.reshape(-1)

        value = value.reshape(-1)
        value = value[torch.isfinite(value)]
        if value.numel() == 0:
            return

        if name not in self.data:
            self.data[name] = {
                "sum": torch.zeros((), device=value.device, dtype=torch.float64),
                "sum_sq": torch.zeros((), device=value.device, dtype=torch.float64),
                "count": torch.zeros((), device=value.device, dtype=torch.float64),
                "min": torch.full((), float("inf"), device=value.device, dtype=torch.float64),
                "max": torch.full((), -float("inf"), device=value.device, dtype=torch.float64),
            }

        v64 = value.double()
        item = self.data[name]
        item["sum"] += v64.sum()
        item["sum_sq"] += torch.sum(v64 * v64)
        item["count"] += torch.tensor(v64.numel(), device=value.device, dtype=torch.float64)
        item["min"] = torch.minimum(item["min"], v64.min())
        item["max"] = torch.maximum(item["max"], v64.max())

    def _nan(self, name: Optional[str] = None) -> torch.Tensor:
        if self.device is not None:
            return torch.tensor(float("nan"), device=self.device)
        if name is not None and name in self.data:
            return torch.tensor(float("nan"), device=self.data[name]["sum"].device)
        return torch.tensor(float("nan"))

    def mean(self, name: Optional[str] = None):
        if name is None:
            return {key: self.mean(key) for key in self.data.keys()}
        if name not in self.data or self.data[name]["count"] <= 0:
            return self._nan(name)
        return self.data[name]["sum"] / self.data[name]["count"]

    def rmse(self, name: Optional[str] = None):
        if name is None:
            return {key: self.rmse(key) for key in self.data.keys()}
        if name not in self.data or self.data[name]["count"] <= 0:
            return self._nan(name)
        return torch.sqrt(self.data[name]["sum_sq"] / self.data[name]["count"])

    def min(self, name: Optional[str] = None):
        if name is None:
            return {key: self.min(key) for key in self.data.keys()}
        if name not in self.data or self.data[name]["count"] <= 0:
            return self._nan(name)
        return self.data[name]["min"]

    def max(self, name: Optional[str] = None):
        if name is None:
            return {key: self.max(key) for key in self.data.keys()}
        if name not in self.data or self.data[name]["count"] <= 0:
            return self._nan(name)
        return self.data[name]["max"]

    def count(self, name: Optional[str] = None):
        if name is None:
            return {key: self.count(key) for key in self.data.keys()}
        if name not in self.data:
            return torch.zeros((), device=self.device)
        return self.data[name]["count"]


def compute_convex_hull(points: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute 3D volume and XY/XZ projected hull areas with optional scipy."""
    try:
        from scipy.spatial import ConvexHull
    except Exception as exc:  # pragma: no cover - optional dependency path
        warnings.warn(f"scipy is not available; skipping convex hull metrics: {exc}")
        return None

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    result = {
        "hull_volume": float("nan"),
        "hull_area_xy": float("nan"),
        "hull_area_xz": float("nan"),
    }
    points = np.unique(points, axis=0)
    if 0 < points.shape[0] < 4:
        result["hull_volume"] = 0.0
    if 0 < points.shape[0] < 3:
        result["hull_area_xy"] = 0.0
        result["hull_area_xz"] = 0.0
    if points.shape[0] >= 4:
        try:
            if np.linalg.matrix_rank(points - points.mean(axis=0, keepdims=True)) < 3:
                result["hull_volume"] = 0.0
            else:
                result["hull_volume"] = float(ConvexHull(points).volume)
        except Exception as exc:
            warnings.warn(f"3D convex hull failed: {exc}")
    if points.shape[0] >= 3:
        try:
            points_xy = points[:, [0, 1]]
            if np.linalg.matrix_rank(points_xy - points_xy.mean(axis=0, keepdims=True)) < 2:
                result["hull_area_xy"] = 0.0
            else:
                result["hull_area_xy"] = float(ConvexHull(points_xy).volume)
        except Exception as exc:
            warnings.warn(f"XY convex hull failed: {exc}")
        try:
            points_xz = points[:, [0, 2]]
            if np.linalg.matrix_rank(points_xz - points_xz.mean(axis=0, keepdims=True)) < 2:
                result["hull_area_xz"] = 0.0
            else:
                result["hull_area_xz"] = float(ConvexHull(points_xz).volume)
        except Exception as exc:
            warnings.warn(f"XZ convex hull failed: {exc}")
    return result


def _to_index_tensor(indices: Any, device: torch.device) -> Optional[torch.Tensor]:
    if indices is None:
        return None
    if isinstance(indices, torch.Tensor):
        out = indices.detach().long().to(device)
    elif isinstance(indices, slice):
        return None
    else:
        try:
            out = torch.as_tensor(indices, device=device, dtype=torch.long)
        except (TypeError, ValueError):
            return None
    if out.numel() == 0:
        return None
    return out.reshape(-1)


def infer_leg_arm_indices(env: Any, cfg: Any = None) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Infer leg and arm DOF indices from env fields, names, or a conservative split."""
    device = torch.device(getattr(env, "device", "cpu"))
    warnings_out: List[str] = []

    leg = _to_index_tensor(safe_get_nested_attr(env, "leg_dof_indices"), device)
    arm = _to_index_tensor(safe_get_nested_attr(env, "arm_dof_indices"), device)
    if arm is None:
        arm = _to_index_tensor(safe_get_nested_attr(env, "arm_indices"), device)

    dof_names = safe_get_nested_attr(env, "dof_names")
    if dof_names is not None:
        leg_keywords = ("FL", "FR", "RL", "RR", "hip", "thigh", "calf")
        arm_keywords = (
            "z1",
            "arm",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "wrist",
            "gripper",
        )
        if leg is None:
            leg_ids = [
                i
                for i, name in enumerate(dof_names)
                if any(keyword.lower() in str(name).lower() for keyword in leg_keywords)
            ]
            leg = _to_index_tensor(leg_ids, device)
        if arm is None:
            arm_names = safe_get_nested_attr(cfg, "multi_agent.arm_dof_names")
            if arm_names and hasattr(env, "dof_names_to_idx"):
                arm_ids = [
                    env.dof_names_to_idx[name]
                    for name in arm_names
                    if name in env.dof_names_to_idx
                ]
            else:
                arm_ids = [
                    i
                    for i, name in enumerate(dof_names)
                    if any(keyword.lower() in str(name).lower() for keyword in arm_keywords)
                ]
            arm = _to_index_tensor(arm_ids, device)

    num_dof = int(
        getattr(
            env,
            "num_dofs",
            getattr(getattr(cfg, "env", None), "num_torques", getattr(env, "num_actions", 18)),
        )
    )
    if leg is None:
        leg = torch.arange(0, min(12, num_dof), device=device, dtype=torch.long)
        warnings_out.append("Using fallback leg DOF split: first 12 joints.")
    if arm is None:
        arm = torch.arange(12, min(num_dof, 18), device=device, dtype=torch.long)
        warnings_out.append("Using fallback arm DOF split: joints 12 through 17.")

    if leg.numel() == 0:
        leg = torch.arange(0, min(12, num_dof), device=device, dtype=torch.long)
        warnings_out.append("Using fallback leg DOF split: first 12 joints.")
    if arm.numel() == 0:
        arm = torch.arange(12, min(num_dof, 18), device=device, dtype=torch.long)
        warnings_out.append("Using fallback arm DOF split: joints 12 through 17.")
    return leg, arm, warnings_out


def _quat_to_euler_xyz(q: torch.Tensor, quat_order: str = "xyzw") -> torch.Tensor:
    q = _normalize_quat(_as_xyzw(q, quat_order))
    x, y, z, w = q.unbind(-1)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    t2 = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    return torch.stack([roll, pitch, yaw], dim=-1)


def _cart_to_sphere(pos: torch.Tensor) -> torch.Tensor:
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    radius = torch.linalg.norm(pos, dim=-1).clamp_min(1.0e-9)
    pitch = torch.atan2(z, torch.sqrt(x * x + y * y).clamp_min(1.0e-9))
    yaw = torch.atan2(y, x)
    return torch.stack([radius, pitch, yaw], dim=-1)


def _quat_apply_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat(q)
    q_xyz = q[:, :3]
    q_w = q[:, 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v - q_w * t + torch.cross(q_xyz, t, dim=-1)


class CoordinationMetrics:
    def __init__(
        self,
        env: Any,
        cfg: Any = None,
        num_envs: Optional[int] = None,
        device: Optional[torch.device] = None,
        dt: Optional[float] = None,
        warmup_steps: int = 0,
        success_pos_thr: float = 0.03,
        success_ori_thr: float = math.pi / 36,
        fall_height_thr: float = 0.26,
        save_eval_traj: bool = False,
    ):
        self.cfg = cfg if cfg is not None else getattr(env, "cfg", None)
        self.num_envs = int(num_envs if num_envs is not None else getattr(env, "num_envs", 1))
        self.device = torch.device(device if device is not None else getattr(env, "device", "cpu"))
        self.dt = float(dt if dt is not None else getattr(env, "dt", 0.02))
        self.warmup_steps = int(warmup_steps or 0)
        self.success_pos_thr = float(success_pos_thr)
        self.success_ori_thr = float(success_ori_thr)
        cfg_fall_height = safe_get_nested_attr(self.cfg, "env.termination_height")
        self.fall_height_thr = float(cfg_fall_height if cfg_fall_height is not None else fall_height_thr)
        self.save_eval_traj = bool(save_eval_traj)

        self.leg_dof_indices, self.arm_dof_indices, init_warnings = infer_leg_arm_indices(env, self.cfg)
        self.leg_action_indices, self.arm_action_indices, action_warnings = self._infer_action_indices(env)
        self.reset()
        for message in init_warnings + action_warnings:
            self._warn_once(message)

    def reset(self) -> None:
        self.acc = RunningMetricAccumulator(self.device)
        self.rollout_steps = 0
        self.eval_steps = 0
        self.total_samples = torch.zeros((), device=self.device, dtype=torch.float64)
        self.energy_sums = {
            "leg": torch.zeros((), device=self.device, dtype=torch.float64),
            "arm": torch.zeros((), device=self.device, dtype=torch.float64),
            "total": torch.zeros((), device=self.device, dtype=torch.float64),
        }
        self.episodes_finished = torch.zeros((), device=self.device, dtype=torch.float64)
        self.episode_success_count = torch.zeros((), device=self.device, dtype=torch.float64)
        self.episode_survival_count = torch.zeros((), device=self.device, dtype=torch.float64)
        self.current_episode_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.current_episode_alive = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.current_episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.prev_actions: Optional[torch.Tensor] = None
        self.prev_base_ang_vel: Optional[torch.Tensor] = None
        self.prev_base_lin_vel: Optional[torch.Tensor] = None

        self.history: Dict[str, List[torch.Tensor]] = {}
        self.traj: Dict[str, List[torch.Tensor]] = {}
        self.success_points: List[torch.Tensor] = []
        self.time_to_success: List[torch.Tensor] = []

        self.skipped: List[str] = []
        self.warnings: List[str] = []
        self._warned_messages = set()

    def add_warning(self, message: str) -> None:
        self._warn_once(message)

    def update(self, env: Any, actions: torch.Tensor, obs: Optional[torch.Tensor] = None) -> None:
        del obs
        self.rollout_steps += 1

        actions = self._batch_tensor(actions, "actions")
        base_lin_vel = self._get_base_lin_vel(env)
        base_ang_vel = self._get_base_ang_vel(env)

        if self.rollout_steps <= self.warmup_steps:
            self._cache_previous(actions, base_lin_vel, base_ang_vel)
            return

        self.eval_steps += 1
        self.total_samples += self.num_envs
        all_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        commands = self._get_commands(env)
        vx_err, vy_err, yaw_err, vel_l1 = self._update_velocity_metrics(base_lin_vel, base_ang_vel, commands)

        base_height = self._get_base_height(env)
        reset_event, timeout_mask = self._get_reset_masks(env)
        collision_mask = self._get_collision_mask(env)
        fall_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if base_height is not None:
            fall_mask |= base_height < self.fall_height_thr
            self.acc.add("base_height", base_height)
            self._append_history("base_height", base_height)
            self._append_traj("base_height", base_height)
        else:
            self._skip_once("stability/base_height", "base height tensor not found")

        termination = self._get_bool_tensor(env, ["termination_buf"])
        if termination is not None:
            fall_mask |= termination
        if reset_event is not None:
            if timeout_mask is not None:
                fall_mask |= reset_event & ~timeout_mask
            else:
                fall_mask |= reset_event
        elif base_height is None and termination is None:
            self._skip_once("stability/survival_rate_step", "no base height or reset tensor found")

        if collision_mask is not None:
            fall_mask |= collision_mask

        alive_mask = ~fall_mask
        self.acc.add("survival", alive_mask.float())
        self.acc.add("fall", fall_mask.float())

        base_rpy = self._get_base_rpy(env)
        base_roll = base_rpy[:, 0] if base_rpy is not None else None
        base_pitch = base_rpy[:, 1] if base_rpy is not None else None

        base_ang_acc_norm, base_lin_acc_norm = self._update_acceleration_metrics(base_ang_vel, base_lin_vel)

        ee_pos_err, ee_pos_l1, target_ee_pos = self._update_ee_position_metrics(env)
        ee_ori_err = self._update_ee_orientation_metrics(env)
        self._update_spherical_metrics(env)

        success_mask = None
        if ee_pos_err is not None and ee_ori_err is not None:
            success_alive = alive_mask.clone()
            if reset_event is not None:
                success_alive &= ~reset_event
            success_mask = (
                (ee_pos_err <= self.success_pos_thr)
                & (ee_ori_err <= self.success_ori_thr)
                & success_alive
            )
            self.acc.add("ee_success_step", success_mask.float())
            self._append_history("success_mask", success_mask.float())
            self._append_traj("success_mask", success_mask.float())
            points = target_ee_pos if target_ee_pos is not None else self._get_ee_actual_pos(env)
            if points is not None:
                self.success_points.append(points[success_mask].detach())
        else:
            self._skip_once("ee/success_rate", "EE position or orientation error unavailable")

        self._update_episode_metrics(success_mask, alive_mask, reset_event)
        self._update_energy_metrics(env)

        leg_action_rate = None
        arm_action_rate = None
        total_action_rate = None
        arm_action_delta = None
        arm_action_norm = None
        if actions is not None and self.prev_actions is not None:
            delta = actions - self.prev_actions
            total_action_rate = torch.linalg.norm(delta, dim=-1) / max(self.dt, 1.0e-9)
            self.acc.add("total_action_rate", total_action_rate)
            if self.leg_action_indices.numel() > 0:
                leg_delta = delta.index_select(1, self._valid_indices(self.leg_action_indices, delta.shape[1]))
                leg_action_rate = torch.linalg.norm(leg_delta, dim=-1) / max(self.dt, 1.0e-9)
                self.acc.add("leg_action_rate", leg_action_rate)
            if self.arm_action_indices.numel() > 0:
                valid_arm = self._valid_indices(self.arm_action_indices, delta.shape[1])
                arm_delta = delta.index_select(1, valid_arm)
                arm_action_rate = torch.linalg.norm(arm_delta, dim=-1) / max(self.dt, 1.0e-9)
                arm_action_delta = torch.linalg.norm(arm_delta, dim=-1)
                arm_action_norm = torch.linalg.norm(actions.index_select(1, valid_arm), dim=-1)
                self.acc.add("arm_action_rate", arm_action_rate)
        elif actions is None:
            self._skip_once("smoothness/action_rate", "actions tensor unavailable")

        self._append_coordination_histories(
            target_ee_pos=target_ee_pos,
            base_roll=base_roll,
            base_pitch=base_pitch,
            ee_pos_err=ee_pos_err,
            base_ang_acc_norm=base_ang_acc_norm,
            base_lin_acc_norm=base_lin_acc_norm,
            arm_action_delta=arm_action_delta,
            arm_action_norm=arm_action_norm,
            vel_l1=vel_l1,
            survival=alive_mask.float(),
        )

        if ee_pos_err is not None:
            self._append_traj("ee_pos_err", ee_pos_err)
        if ee_ori_err is not None:
            self._append_traj("ee_ori_err", ee_ori_err)
        if base_ang_acc_norm is not None:
            self._append_traj("base_ang_acc", base_ang_acc_norm)
        if vel_l1 is not None:
            self._append_traj("vel_err", vel_l1)

        self._cache_previous(actions, base_lin_vel, base_ang_vel)

    def summarize(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "meta/num_envs": self.num_envs,
            "meta/eval_steps": self.eval_steps,
            "meta/warmup_steps": self.warmup_steps,
            "meta/dt": self.dt,
            "meta/total_samples": self._scalar(self.total_samples),
        }

        summary.update(
            {
                "vel/vx_mae": self._scalar(self.acc.mean("vx_err")),
                "vel/vx_rmse": self._scalar(self.acc.rmse("vx_err")),
                "vel/vx_max": self._scalar(self.acc.max("vx_err")),
                "vel/vy_mae": self._scalar(self.acc.mean("vy_err")),
                "vel/vy_rmse": self._scalar(self.acc.rmse("vy_err")),
                "vel/vy_max": self._scalar(self.acc.max("vy_err")),
                "vel/yaw_mae": self._scalar(self.acc.mean("yaw_err")),
                "vel/yaw_rmse": self._scalar(self.acc.rmse("yaw_err")),
                "vel/yaw_max": self._scalar(self.acc.max("yaw_err")),
                "vel/l1_mean": self._scalar(self.acc.mean("vel_l1")),
                "vel/l1_rmse": self._scalar(self.acc.rmse("vel_l1")),
            }
        )

        ee_pos_stats = self._history_stats("ee_pos_err")
        ee_ori_stats = self._history_stats("ee_ori_err")
        summary.update(
            {
                "ee/pos_mean": ee_pos_stats["mean"],
                "ee/pos_rmse": ee_pos_stats["rmse"],
                "ee/pos_median": ee_pos_stats["median"],
                "ee/pos_p90": ee_pos_stats["p90"],
                "ee/pos_max": ee_pos_stats["max"],
                "ee/ori_geodesic_mean": ee_ori_stats["mean"],
                "ee/ori_geodesic_rmse": ee_ori_stats["rmse"],
                "ee/ori_geodesic_median": ee_ori_stats["median"],
                "ee/ori_geodesic_p90": ee_ori_stats["p90"],
                "ee/ori_geodesic_max": ee_ori_stats["max"],
                "ee/l1_mean": self._scalar(self.acc.mean("ee_l1")),
                "ee/l1_rmse": self._scalar(self.acc.rmse("ee_l1")),
                "ee/success_rate_step": self._scalar(self.acc.mean("ee_success_step")),
            }
        )

        episode_total, episode_success, episode_survival = self._episode_totals()
        summary["ee/success_rate_episode"] = self._safe_ratio(episode_success, episode_total)
        t_success = self._cat_history_list(self.time_to_success)
        if t_success.numel() > 0:
            summary["ee/time_to_success_mean"] = self._scalar(t_success.mean())
            summary["ee/time_to_success_median"] = self._scalar(torch.quantile(t_success.float(), 0.5))
        else:
            summary["ee/time_to_success_mean"] = float("nan")
            summary["ee/time_to_success_median"] = float("nan")

        base_height_stats = self._history_stats("base_height", p_low=0.05)
        base_ang_acc_stats = self._history_stats("base_ang_acc")
        base_lin_acc_stats = self._history_stats("base_lin_acc")
        summary.update(
            {
                "stability/survival_rate_step": self._scalar(self.acc.mean("survival")),
                "stability/fall_rate_step": self._scalar(self.acc.mean("fall")),
                "stability/episode_survival_rate": self._safe_ratio(episode_survival, episode_total),
                "stability/base_height_mean": base_height_stats["mean"],
                "stability/base_height_min": base_height_stats["min"],
                "stability/base_height_p05": base_height_stats["p_low"],
                "stability/base_ang_acc_mean": base_ang_acc_stats["mean"],
                "stability/base_ang_acc_rms": base_ang_acc_stats["rmse"],
                "stability/base_ang_acc_p90": base_ang_acc_stats["p90"],
                "stability/base_lin_acc_mean": base_lin_acc_stats["mean"],
                "stability/base_lin_acc_rms": base_lin_acc_stats["rmse"],
                "stability/base_lin_acc_p90": base_lin_acc_stats["p90"],
            }
        )

        total_energy = self.energy_sums["total"]
        summary.update(
            {
                "energy/leg_power_abs_mean": self._scalar(self.acc.mean("leg_power_abs")),
                "energy/arm_power_abs_mean": self._scalar(self.acc.mean("arm_power_abs")),
                "energy/total_power_abs_mean": self._scalar(self.acc.mean("total_power_abs")),
                "energy/leg_power_squared_mean": self._scalar(self.acc.mean("leg_power_squared")),
                "energy/arm_power_squared_mean": self._scalar(self.acc.mean("arm_power_squared")),
                "energy/total_power_squared_mean": self._scalar(self.acc.mean("total_power_squared")),
                "energy/leg_energy_sum": self._scalar(self.energy_sums["leg"]),
                "energy/arm_energy_sum": self._scalar(self.energy_sums["arm"]),
                "energy/total_energy_sum": self._scalar(total_energy),
                "energy/total_energy_per_step": self._safe_ratio(total_energy, self.total_samples),
                "energy/total_energy_per_episode": self._safe_ratio(total_energy, episode_total),
            }
        )

        summary.update(
            {
                "smoothness/leg_action_rate_mean": self._scalar(self.acc.mean("leg_action_rate")),
                "smoothness/arm_action_rate_mean": self._scalar(self.acc.mean("arm_action_rate")),
                "smoothness/total_action_rate_mean": self._scalar(self.acc.mean("total_action_rate")),
            }
        )

        summary.update(self._coordination_summary())
        workspace_summary = self._workspace_summary(summary["ee/success_rate_episode"])
        summary.update(workspace_summary)
        summary["metrics/skipped"] = self.skipped
        summary["metrics/warnings"] = self.warnings
        return summary

    def save(self, out_dir: str, prefix: str = "coordination_eval") -> Dict[str, Optional[str]]:
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary = self.summarize()

        json_path = os.path.join(out_dir, f"{prefix}_{timestamp}.json")
        csv_path = os.path.join(out_dir, f"{prefix}_{timestamp}.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in summary.items():
                writer.writerow([key, self._csv_value(value)])

        traj_path = None
        if self.save_eval_traj:
            arrays = {}
            for key, values in self.traj.items():
                tensor = self._cat_history_list(values)
                arrays[key] = tensor.detach().cpu().numpy()
            traj_path = os.path.join(out_dir, f"{prefix}_traj_{timestamp}.npz")
            np.savez_compressed(traj_path, **arrays)

        return {"json": json_path, "csv": csv_path, "npz": traj_path}

    def _infer_action_indices(self, env: Any) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        warnings_out: List[str] = []
        leg = _to_index_tensor(safe_get_nested_attr(env, "leg_action_indices"), self.device)
        arm = _to_index_tensor(safe_get_nested_attr(env, "arm_action_indices"), self.device)
        num_actions = int(getattr(env, "num_actions", getattr(getattr(self.cfg, "env", None), "num_actions", 18)))
        if leg is None:
            leg = torch.arange(0, min(12, num_actions), device=self.device, dtype=torch.long)
            warnings_out.append("Using fallback leg/arm action split: first 12 leg, next 6 arm.")
        if arm is None:
            arm = torch.arange(12, min(num_actions, 18), device=self.device, dtype=torch.long)
        return leg, arm, warnings_out

    def _batch_tensor(self, value: Any, name: str = "tensor") -> Optional[torch.Tensor]:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.as_tensor(value, device=self.device)
            except (TypeError, ValueError):
                self._skip_once(name, "not tensor-like")
                return None
        value = value.detach().to(self.device)
        if value.dim() == 0:
            self._skip_once(name, "scalar tensor is not batched")
            return None
        if value.shape[0] != self.num_envs:
            self._skip_once(name, f"expected first dimension {self.num_envs}, got {value.shape[0]}")
            return None
        return value

    def _get_base_lin_vel(self, env: Any) -> Optional[torch.Tensor]:
        return self._batch_tensor(
            safe_get_tensor(env, ["base_lin_vel", "base_linear_velocity", "base_vel"]),
            "base linear velocity",
        )

    def _get_base_ang_vel(self, env: Any) -> Optional[torch.Tensor]:
        return self._batch_tensor(
            safe_get_tensor(env, ["base_ang_vel", "base_angular_velocity"]),
            "base angular velocity",
        )

    def _get_commands(self, env: Any) -> Optional[torch.Tensor]:
        commands = self._batch_tensor(
            safe_get_tensor(env, ["commands", "command", "commands_scale"]),
            "commands",
        )
        if commands is not None and commands.dim() == 1:
            commands = commands.unsqueeze(-1)
        return commands

    def _get_base_height(self, env: Any) -> Optional[torch.Tensor]:
        root_states = self._batch_tensor(safe_get_tensor(env, ["root_states"]), "root_states")
        if root_states is not None and root_states.shape[-1] > 2:
            return root_states[:, 2]
        base_pos = self._batch_tensor(
            safe_get_tensor(env, ["base_pos", "base_position"]),
            "base position",
        )
        if base_pos is not None and base_pos.shape[-1] > 2:
            return base_pos[:, 2]
        return None

    def _get_base_rpy(self, env: Any) -> Optional[torch.Tensor]:
        base_rpy = self._batch_tensor(safe_get_tensor(env, ["base_rpy"]), "base_rpy")
        if base_rpy is not None and base_rpy.shape[-1] >= 3:
            return base_rpy[:, :3]

        roll = self._batch_tensor(safe_get_tensor(env, ["roll"]), "base roll")
        pitch = self._batch_tensor(safe_get_tensor(env, ["pitch"]), "base pitch")
        yaw = self._batch_tensor(safe_get_tensor(env, ["yaw"]), "base yaw")
        if roll is not None and pitch is not None and yaw is not None:
            return torch.stack([roll.reshape(-1), pitch.reshape(-1), yaw.reshape(-1)], dim=-1)

        base_quat = self._batch_tensor(safe_get_tensor(env, ["base_quat"]), "base_quat")
        if base_quat is None:
            root_states = self._batch_tensor(safe_get_tensor(env, ["root_states"]), "root_states")
            if root_states is not None and root_states.shape[-1] >= 7:
                base_quat = root_states[:, 3:7]
        if base_quat is not None and base_quat.shape[-1] == 4:
            return _quat_to_euler_xyz(base_quat, quat_order="xyzw")

        self._skip_once("coordination/base_orientation", "base orientation tensor not found")
        return None

    def _get_ee_actual_pos(self, env: Any) -> Optional[torch.Tensor]:
        pos = self._batch_tensor(
            safe_get_tensor(
                env,
                [
                    "ee_pos",
                    "ee_pos_world",
                    "ee_pos_local",
                    "curr_ee_pos",
                    "end_effector_pos",
                    "gripper_pos",
                ],
            ),
            "EE actual position",
        )
        if pos is not None and pos.shape[-1] >= 3:
            return pos[:, :3]
        rigid_body_state = self._batch_tensor(
            safe_get_tensor(env, ["rigid_body_states", "rigid_body_state"]),
            "rigid body state",
        )
        ee_index = safe_get_nested_attr(env, "ee_body_idx")
        if ee_index is None:
            ee_index = safe_get_nested_attr(env, "gripper_idx")
        if rigid_body_state is not None and ee_index is not None and rigid_body_state.dim() >= 3:
            return rigid_body_state[:, int(ee_index), :3]
        return None

    def _get_ee_target_pos(self, env: Any) -> Optional[torch.Tensor]:
        pos = self._batch_tensor(
            safe_get_tensor(
                env,
                [
                    "curr_ee_goal_cart_world",
                    "ee_goal_pos",
                    "ee_target_pos",
                    "target_ee_pos",
                    "ee_goal",
                    "commands_ee",
                    "curr_ee_goal",
                    "goal_ee_pos",
                    "curr_ee_goal_cart",
                    "ee_goal_cart",
                ],
            ),
            "EE target position",
        )
        if pos is not None and pos.shape[-1] >= 3:
            return pos[:, :3]
        return None

    def _get_ee_actual_ori(self, env: Any) -> Optional[torch.Tensor]:
        return self._batch_tensor(
            safe_get_tensor(
                env,
                [
                    "ee_quat",
                    "ee_quat_world",
                    "ee_orn",
                    "ee_rot",
                    "ee_rpy",
                    "end_effector_quat",
                    "gripper_quat",
                ],
            ),
            "EE actual orientation",
        )

    def _get_ee_target_ori(self, env: Any) -> Optional[torch.Tensor]:
        return self._batch_tensor(
            safe_get_tensor(
                env,
                [
                    "ee_goal_quat",
                    "ee_target_quat",
                    "target_ee_quat",
                    "goal_ee_quat",
                    "commands_ee_quat",
                    "ee_goal_orn_quat",
                    "ee_goal_orn_euler",
                ],
            ),
            "EE target orientation",
        )

    def _get_bool_tensor(self, env: Any, candidates: Iterable[str]) -> Optional[torch.Tensor]:
        tensor = self._batch_tensor(safe_get_tensor(env, candidates), "/".join(candidates))
        if tensor is None:
            return None
        return tensor.reshape(self.num_envs, -1).any(dim=1).bool()

    def _get_reset_masks(self, env: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        reset_event = self._get_bool_tensor(env, ["reset_buf"])
        termination = self._get_bool_tensor(env, ["termination_buf"])
        if reset_event is None:
            reset_event = termination
        elif termination is not None:
            reset_event = reset_event | termination
        timeout = self._get_bool_tensor(env, ["time_out_buf"])
        return reset_event, timeout

    def _get_collision_mask(self, env: Any) -> Optional[torch.Tensor]:
        collision = self._get_bool_tensor(env, ["collision_buf", "self_collision_buf"])

        contact_forces = self._batch_tensor(safe_get_tensor(env, ["contact_forces"]), "contact_forces")
        penalized = _to_index_tensor(safe_get_nested_attr(env, "penalized_contact_indices"), self.device)
        if contact_forces is not None and penalized is not None and contact_forces.dim() >= 3:
            valid = self._valid_indices(penalized, contact_forces.shape[1])
            if valid.numel() > 0:
                contact_collision = torch.any(
                    torch.linalg.norm(contact_forces.index_select(1, valid), dim=-1) > 1.0,
                    dim=1,
                )
                collision = contact_collision if collision is None else (collision | contact_collision)
        return collision

    def _update_velocity_metrics(
        self,
        base_lin_vel: Optional[torch.Tensor],
        base_ang_vel: Optional[torch.Tensor],
        commands: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        vx_err = vy_err = yaw_err = vel_l1 = None
        if commands is None:
            self._skip_once("vel", "command tensor not found")
            return vx_err, vy_err, yaw_err, vel_l1
        if base_lin_vel is None:
            self._skip_once("vel/linear", "base linear velocity not found")
        if base_ang_vel is None:
            self._skip_once("vel/yaw", "base angular velocity not found")

        if base_lin_vel is not None and base_lin_vel.shape[-1] >= 1 and commands.shape[-1] >= 1:
            vx_err = torch.abs(base_lin_vel[:, 0] - commands[:, 0])
            self.acc.add("vx_err", vx_err)
        if base_lin_vel is not None and base_lin_vel.shape[-1] >= 2 and commands.shape[-1] >= 3:
            vy_err = torch.abs(base_lin_vel[:, 1] - commands[:, 1])
            self.acc.add("vy_err", vy_err)
        else:
            self._skip_once("vel/vy", "vy command unavailable")
        yaw_idx = 2 if commands.shape[-1] >= 3 else (1 if commands.shape[-1] >= 2 else None)
        if base_ang_vel is not None and base_ang_vel.shape[-1] >= 3 and yaw_idx is not None:
            yaw_err = torch.abs(base_ang_vel[:, 2] - commands[:, yaw_idx])
            self.acc.add("yaw_err", yaw_err)
        components = [item for item in (vx_err, vy_err, yaw_err) if item is not None]
        if components:
            vel_l1 = torch.stack(components, dim=-1).sum(dim=-1)
            self.acc.add("vel_l1", vel_l1)
            self._append_history("vel_l1", vel_l1)
        return vx_err, vy_err, yaw_err, vel_l1

    def _update_acceleration_metrics(
        self,
        base_ang_vel: Optional[torch.Tensor],
        base_lin_vel: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        base_ang_acc_norm = None
        base_lin_acc_norm = None
        if base_ang_vel is not None and self.prev_base_ang_vel is not None:
            base_ang_acc = (base_ang_vel - self.prev_base_ang_vel) / max(self.dt, 1.0e-9)
            base_ang_acc_norm = torch.linalg.norm(base_ang_acc, dim=-1)
            self.acc.add("base_ang_acc", base_ang_acc_norm)
            self._append_history("base_ang_acc", base_ang_acc_norm)
        elif base_ang_vel is None:
            self._skip_once("stability/base_ang_acc", "base angular velocity unavailable")

        if base_lin_vel is not None and self.prev_base_lin_vel is not None:
            base_lin_acc = (base_lin_vel - self.prev_base_lin_vel) / max(self.dt, 1.0e-9)
            base_lin_acc_norm = torch.linalg.norm(base_lin_acc, dim=-1)
            self.acc.add("base_lin_acc", base_lin_acc_norm)
            self._append_history("base_lin_acc", base_lin_acc_norm)
        elif base_lin_vel is None:
            self._skip_once("stability/base_lin_acc", "base linear velocity unavailable")
        return base_ang_acc_norm, base_lin_acc_norm

    def _update_ee_position_metrics(self, env: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        actual = self._get_ee_actual_pos(env)
        target = self._get_ee_target_pos(env)
        if actual is None or target is None:
            self._skip_once("ee/position", "EE actual or target position tensor not found")
            return None, None, target
        err_vec = actual[:, :3] - target[:, :3]
        pos_err = torch.linalg.norm(err_vec, dim=-1)
        pos_l1 = torch.sum(torch.abs(err_vec), dim=-1)
        self.acc.add("ee_pos_err", pos_err)
        self.acc.add("ee_l1", pos_l1)
        self._append_history("ee_pos_err", pos_err)
        return pos_err, pos_l1, target

    def _update_ee_orientation_metrics(self, env: Any) -> Optional[torch.Tensor]:
        actual = self._get_ee_actual_ori(env)
        target = self._get_ee_target_ori(env)
        if actual is None or target is None:
            self._skip_once("ee/orientation", "EE actual or target orientation tensor not found")
            return None
        if actual.shape[-1] == 4 and target.shape[-1] == 4:
            ori_err = quat_geodesic_distance(actual[:, :4], target[:, :4], quat_order="xyzw")
        elif actual.shape[-1] >= 3 and target.shape[-1] >= 3:
            ori_err = torch.linalg.norm(angle_wrap_error(actual[:, :3], target[:, :3]), dim=-1)
        else:
            self._skip_once("ee/orientation", "orientation tensors are neither quaternion nor rpy")
            return None
        self.acc.add("ee_ori_err", ori_err)
        self._append_history("ee_ori_err", ori_err)
        return ori_err

    def _update_spherical_metrics(self, env: Any) -> None:
        target_sphere = self._batch_tensor(
            safe_get_tensor(env, ["curr_ee_goal_sphere", "ee_goal_sphere", "target_ee_sphere"]),
            "EE spherical target",
        )
        actual_pos = self._get_ee_actual_pos(env)
        if target_sphere is None or actual_pos is None or target_sphere.shape[-1] < 3:
            return

        center = None
        if hasattr(env, "_get_ee_goal_spherical_center"):
            try:
                center = env._get_ee_goal_spherical_center()
            except Exception:
                center = None
        center = self._batch_tensor(center, "EE spherical center")
        if center is None:
            center = self._batch_tensor(safe_get_tensor(env, ["ee_goal_center", "ee_goal_center_offset"]), "EE spherical center")
        if center is None:
            return
        local = actual_pos[:, :3] - center[:, :3]
        base_yaw_quat = self._batch_tensor(safe_get_tensor(env, ["base_yaw_quat"]), "base_yaw_quat")
        if base_yaw_quat is not None and base_yaw_quat.shape[-1] == 4:
            local = _quat_apply_inverse_xyzw(base_yaw_quat, local)
        actual_sphere = _cart_to_sphere(local)
        sphere_err = torch.abs(actual_sphere - target_sphere[:, :3])
        sphere_err[:, 1:] = torch.abs(angle_wrap_error(actual_sphere[:, 1:], target_sphere[:, 1:3]))
        self.acc.add("ee_sphere_l_err", sphere_err[:, 0])
        self.acc.add("ee_sphere_p_err", sphere_err[:, 1])
        self.acc.add("ee_sphere_y_err", sphere_err[:, 2])

    def _update_episode_metrics(
        self,
        success_mask: Optional[torch.Tensor],
        alive_mask: torch.Tensor,
        reset_event: Optional[torch.Tensor],
    ) -> None:
        self.current_episode_steps += 1
        self.current_episode_alive &= alive_mask
        if success_mask is not None:
            first_success = success_mask & ~self.current_episode_success
            self.current_episode_success |= success_mask
            success_times = self.current_episode_steps.float() * self.dt
            self.time_to_success.append(success_times[first_success].detach())

        if reset_event is None:
            return
        finished = reset_event & (self.current_episode_steps > 0)
        self.episodes_finished += finished.float().sum()
        self.episode_success_count += (self.current_episode_success & finished).float().sum()
        self.episode_survival_count += (self.current_episode_alive & finished).float().sum()
        self.current_episode_success[reset_event] = False
        self.current_episode_alive[reset_event] = True
        self.current_episode_steps[reset_event] = 0

    def _update_energy_metrics(self, env: Any) -> None:
        torques = self._batch_tensor(safe_get_tensor(env, ["torques"]), "torques")
        dof_vel = self._batch_tensor(safe_get_tensor(env, ["dof_vel"]), "dof_vel")
        if torques is None or dof_vel is None:
            self._skip_once("energy", "torques or dof_vel tensor not found")
            return
        dim = min(torques.shape[-1], dof_vel.shape[-1])
        torques = torques[:, :dim]
        dof_vel = dof_vel[:, :dim]
        power_abs = torch.abs(torques * dof_vel)
        total_power = power_abs.sum(dim=-1)
        total_power_sq = torch.square(power_abs).sum(dim=-1)
        self.acc.add("total_power_abs", total_power)
        self.acc.add("total_power_squared", total_power_sq)
        self.energy_sums["total"] += total_power.double().sum() * self.dt

        leg_idx = self._valid_indices(self.leg_dof_indices, dim)
        arm_idx = self._valid_indices(self.arm_dof_indices, dim)
        if leg_idx.numel() > 0:
            leg_power_values = power_abs.index_select(1, leg_idx)
            leg_power = leg_power_values.sum(dim=-1)
            self.acc.add("leg_power_abs", leg_power)
            self.acc.add("leg_power_squared", torch.square(leg_power_values).sum(dim=-1))
            self.energy_sums["leg"] += leg_power.double().sum() * self.dt
        else:
            self._skip_once("energy/leg", "leg DOF indices unavailable")
        if arm_idx.numel() > 0:
            arm_power_values = power_abs.index_select(1, arm_idx)
            arm_power = arm_power_values.sum(dim=-1)
            self.acc.add("arm_power_abs", arm_power)
            self.acc.add("arm_power_squared", torch.square(arm_power_values).sum(dim=-1))
            self.energy_sums["arm"] += arm_power.double().sum() * self.dt
        else:
            self._skip_once("energy/arm", "arm DOF indices unavailable")

    def _append_coordination_histories(
        self,
        target_ee_pos: Optional[torch.Tensor],
        base_roll: Optional[torch.Tensor],
        base_pitch: Optional[torch.Tensor],
        ee_pos_err: Optional[torch.Tensor],
        base_ang_acc_norm: Optional[torch.Tensor],
        base_lin_acc_norm: Optional[torch.Tensor],
        arm_action_delta: Optional[torch.Tensor],
        arm_action_norm: Optional[torch.Tensor],
        vel_l1: Optional[torch.Tensor],
        survival: torch.Tensor,
    ) -> None:
        if target_ee_pos is not None:
            self._append_history("target_ee_x", target_ee_pos[:, 0])
            self._append_history("target_ee_y", target_ee_pos[:, 1])
            self._append_history("target_ee_z", target_ee_pos[:, 2])
        if base_roll is not None:
            self._append_history("base_roll", base_roll)
        if base_pitch is not None:
            self._append_history("base_pitch", base_pitch)
        if ee_pos_err is not None:
            self._append_history("coord_ee_pos_err", ee_pos_err)
        if base_ang_acc_norm is not None:
            self._append_history("coord_base_ang_acc", base_ang_acc_norm)
        if base_lin_acc_norm is not None:
            self._append_history("coord_base_lin_acc", base_lin_acc_norm)
        if arm_action_delta is not None:
            self._append_history("arm_action_delta", arm_action_delta)
        if arm_action_norm is not None:
            self._append_history("arm_action_norm", arm_action_norm)
        if vel_l1 is not None:
            self._append_history("coord_vel_l1", vel_l1)
        self._append_history("coord_survival", survival)

    def _coordination_summary(self) -> Dict[str, float]:
        out = {
            "coordination/corr_target_ee_x_base_pitch": self._corr("target_ee_x", "base_pitch"),
            "coordination/corr_target_ee_z_base_pitch": self._corr("target_ee_z", "base_pitch"),
            "coordination/corr_target_ee_y_base_roll": self._corr("target_ee_y", "base_roll"),
            "coordination/corr_ee_pos_err_base_ang_acc": self._corr("coord_ee_pos_err", "coord_base_ang_acc"),
            "coordination/corr_arm_action_norm_base_ang_acc": self._corr("arm_action_norm", "coord_base_ang_acc"),
            "coordination/base_ang_acc_when_arm_large": float("nan"),
            "coordination/base_lin_acc_when_arm_large": float("nan"),
            "coordination/ee_pos_err_when_arm_large": float("nan"),
            "coordination/vel_err_when_arm_large": float("nan"),
            "coordination/survival_when_arm_large": float("nan"),
        }

        arm_delta = self._history_tensor("arm_action_delta")
        if arm_delta.numel() == 0:
            self._skip_once("coordination/large_arm_motion", "arm action delta unavailable")
            return out
        finite = torch.isfinite(arm_delta)
        if finite.sum().item() == 0:
            return out
        threshold = torch.quantile(arm_delta[finite].float(), 0.75)
        large_mask = arm_delta > threshold
        if large_mask.sum().item() == 0:
            return out

        for metric_key, hist_key in [
            ("coordination/base_ang_acc_when_arm_large", "coord_base_ang_acc"),
            ("coordination/base_lin_acc_when_arm_large", "coord_base_lin_acc"),
            ("coordination/ee_pos_err_when_arm_large", "coord_ee_pos_err"),
            ("coordination/vel_err_when_arm_large", "coord_vel_l1"),
            ("coordination/survival_when_arm_large", "coord_survival"),
        ]:
            values = self._history_tensor(hist_key)
            if values.numel() == 0:
                continue
            n = min(values.numel(), large_mask.numel())
            mask = large_mask[:n] & torch.isfinite(values[:n])
            if mask.sum().item() > 0:
                out[metric_key] = self._scalar(values[:n][mask].float().mean())
        return out

    def _workspace_summary(self, fallback_solvability: float) -> Dict[str, float]:
        points = self._cat_history_list(self.success_points)
        success_count = int(points.shape[0]) if points.dim() >= 2 else 0
        out = {
            "workspace/solvability_rate": fallback_solvability,
            "workspace/success_points": success_count,
            "workspace/hull_volume": float("nan"),
            "workspace/hull_area_xy": float("nan"),
            "workspace/hull_area_xz": float("nan"),
        }
        if success_count < 3:
            self._skip_once("workspace/convex_hull", "not enough successful EE points")
            return out
        hull = compute_convex_hull(points[:, :3].detach().cpu().numpy())
        if hull is None:
            self._skip_once("workspace/convex_hull", "scipy unavailable")
            return out
        out["workspace/hull_volume"] = hull["hull_volume"]
        out["workspace/hull_area_xy"] = hull["hull_area_xy"]
        out["workspace/hull_area_xz"] = hull["hull_area_xz"]
        return out

    def _episode_totals(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = self.current_episode_steps > 0
        episode_total = self.episodes_finished + active.float().sum()
        episode_success = self.episode_success_count + (self.current_episode_success & active).float().sum()
        episode_survival = self.episode_survival_count + (self.current_episode_alive & active).float().sum()
        return episode_total, episode_success, episode_survival

    def _safe_ratio(self, numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        if not isinstance(numerator, torch.Tensor):
            numerator = torch.as_tensor(numerator, device=self.device, dtype=torch.float64)
        if not isinstance(denominator, torch.Tensor):
            denominator = torch.as_tensor(denominator, device=self.device, dtype=torch.float64)
        if denominator.item() <= 0:
            return float("nan")
        return self._scalar(numerator.double() / denominator.double())

    def _history_stats(self, name: str, p_low: Optional[float] = None) -> Dict[str, float]:
        values = self._history_tensor(name)
        out = {
            "mean": float("nan"),
            "rmse": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
            "p_low": float("nan"),
        }
        if values.numel() == 0:
            return out
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return out
        values = values.float()
        out["mean"] = self._scalar(values.mean())
        out["rmse"] = self._scalar(torch.sqrt(torch.mean(values * values)))
        out["median"] = self._scalar(torch.quantile(values, 0.5))
        out["p90"] = self._scalar(torch.quantile(values, 0.9))
        out["max"] = self._scalar(values.max())
        out["min"] = self._scalar(values.min())
        if p_low is not None:
            out["p_low"] = self._scalar(torch.quantile(values, p_low))
        return out

    def _corr(self, x_key: str, y_key: str) -> float:
        x = self._history_tensor(x_key)
        y = self._history_tensor(y_key)
        if x.numel() == 0 or y.numel() == 0:
            return float("nan")
        return self._scalar(pearson_corr(x, y))

    def _history_tensor(self, name: str) -> torch.Tensor:
        return self._cat_history_list(self.history.get(name, []))

    def _cat_history_list(self, values: List[torch.Tensor]) -> torch.Tensor:
        tensors = [v.reshape(-1, v.shape[-1]) if v.dim() > 1 and v.shape[-1] == 3 else v.reshape(-1) for v in values if v is not None]
        if not tensors:
            return torch.empty(0, device=self.device)
        if tensors[0].dim() == 2:
            return torch.cat(tensors, dim=0)
        return torch.cat(tensors, dim=0)

    def _append_history(self, name: str, value: Optional[torch.Tensor]) -> None:
        if value is None:
            return
        self.history.setdefault(name, []).append(value.detach().clone())

    def _append_traj(self, name: str, value: Optional[torch.Tensor]) -> None:
        if not self.save_eval_traj or value is None:
            return
        self.traj.setdefault(name, []).append(value.detach().clone())

    def _cache_previous(
        self,
        actions: Optional[torch.Tensor],
        base_lin_vel: Optional[torch.Tensor],
        base_ang_vel: Optional[torch.Tensor],
    ) -> None:
        if actions is not None:
            self.prev_actions = actions.detach().clone()
        if base_lin_vel is not None:
            self.prev_base_lin_vel = base_lin_vel.detach().clone()
        if base_ang_vel is not None:
            self.prev_base_ang_vel = base_ang_vel.detach().clone()

    def _valid_indices(self, indices: torch.Tensor, upper: int) -> torch.Tensor:
        if indices is None:
            return torch.empty(0, device=self.device, dtype=torch.long)
        indices = indices.to(self.device).long().reshape(-1)
        return indices[(indices >= 0) & (indices < upper)]

    def _skip_once(self, key: str, reason: str) -> None:
        message = f"{key}: {reason}"
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        self.skipped.append(message)
        print(f"[CoordinationMetrics] warning: skipping {message}")

    def _warn_once(self, message: str) -> None:
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        self.warnings.append(message)
        print(f"[CoordinationMetrics] warning: {message}")

    @staticmethod
    def _scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float("nan")
            return float(value.detach().double().cpu().item())
        if isinstance(value, np.generic):
            return float(value)
        return float(value)

    @staticmethod
    def _csv_value(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "; ".join(str(v) for v in value)
        return str(value)
