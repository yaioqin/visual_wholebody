"""Deterministic command-space coverage for play-time evaluation.

This module deliberately has no Isaac Gym dependency.  It only builds a finite
schedule from the ranges exposed by the instantiated environment/config and
tracks which scheduled commands were feasible and actually executed.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


BASE_COMMAND_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("lin_vel_x", 0),
    ("lin_vel_y", 1),
    ("ang_vel_yaw", 2),
    ("base_pitch", 3),
    ("base_height", 4),
)
EE_COMMAND_NAMES: Tuple[str, ...] = (
    "pos_l",
    "pos_p",
    "pos_y",
    "delta_orn_r",
    "delta_orn_p",
    "delta_orn_y",
)


def resolve_play_num_envs_value(
    configured_num_envs: int,
    requested_num_envs: Optional[int],
    *,
    coverage_default: Optional[int] = 256,
) -> int:
    """Pure resolver used by play and its no-Isaac unit test.

    An explicit CLI value always wins and is never capped.  The evaluation
    default applies only when the user did not supply ``--num_envs``.
    """
    if requested_num_envs is not None:
        if int(requested_num_envs) <= 0:
            raise ValueError("requested_num_envs must be positive")
        return int(requested_num_envs)
    if coverage_default is not None:
        if int(coverage_default) <= 0:
            raise ValueError("coverage_default must be positive")
        return int(coverage_default)
    if int(configured_num_envs) <= 0:
        raise ValueError("configured_num_envs must be positive")
    return int(configured_num_envs)


@dataclass(frozen=True)
class CoverageBatch:
    sample_ids: torch.Tensor
    base_commands: torch.Tensor
    ee_commands: torch.Tensor
    sources: Tuple[str, ...]

    def __len__(self) -> int:
        return int(self.sample_ids.numel())


def infer_base_command_layout(
    command_ranges: Mapping[str, Sequence[float]],
    use_5d_base_command: bool,
) -> Tuple[Tuple[str, int], ...]:
    """Return only command dimensions the current policy/environment enables."""

    enabled_names = (
        ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "base_pitch", "base_height")
        if use_5d_base_command
        else ("lin_vel_x", "ang_vel_yaw")
    )
    layout_by_name = dict(BASE_COMMAND_LAYOUT)
    missing = [name for name in enabled_names if name not in command_ranges]
    if missing:
        raise ValueError(
            "Enabled base command ranges are missing from env.command_ranges: "
            + ", ".join(missing)
        )
    return tuple((name, layout_by_name[name]) for name in enabled_names)


def _range_tensor(
    ranges: Mapping[str, Sequence[float]], names: Sequence[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    bounds = []
    for name in names:
        value = ranges.get(name)
        if value is None or len(value) != 2:
            raise ValueError(f"Range {name!r} must contain [minimum, maximum]")
        low, high = float(value[0]), float(value[1])
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError(f"Invalid range for {name}: {value}")
        bounds.append((low, high))
    tensor = torch.tensor(bounds, dtype=torch.float64)
    return tensor[:, 0], tensor[:, 1]


def _map_unit(unit: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low + unit.to(dtype=low.dtype) * (high - low)


def _axis_boundary_samples(dim: int) -> torch.Tensor:
    """Center plus min/25%/75%/max along every individual axis."""

    center = torch.full((dim,), 0.5, dtype=torch.float64)
    rows = [center]
    for axis in range(dim):
        for fraction in (0.0, 0.25, 0.75, 1.0):
            row = center.clone()
            row[axis] = fraction
            rows.append(row)
    if dim > 1:
        rows.extend(
            [
                torch.zeros(dim, dtype=torch.float64),
                torch.ones(dim, dtype=torch.float64),
                torch.arange(dim, dtype=torch.float64).remainder(2.0),
                1.0 - torch.arange(dim, dtype=torch.float64).remainder(2.0),
            ]
        )
    return _unique_rows(torch.stack(rows))


def _unique_rows(rows: torch.Tensor) -> torch.Tensor:
    seen = set()
    ordered = []
    for row in rows:
        key = tuple(round(float(value), 12) for value in row.tolist())
        if key not in seen:
            seen.add(key)
            ordered.append(row)
    if not ordered:
        return rows[:0]
    return torch.stack(ordered)


class EvaluationCoverageScheduler:
    """Generate and batch a finite, reproducible evaluation command schedule."""

    MODES = ("ee", "base", "joint")

    def __init__(
        self,
        *,
        command_ranges: Mapping[str, Sequence[float]],
        goal_ee_ranges: Mapping[str, Sequence[float]],
        use_5d_base_command: bool,
        num_samples: int,
        mode: str = "joint",
        seed: int = 1,
        ee_position_bins: int = 11,
        base_height_nominal: Optional[float] = None,
        ee_nominal: Optional[Sequence[float]] = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if ee_position_bins < 3:
            raise ValueError("ee_position_bins must be at least 3")

        self.mode = mode
        self.seed = int(seed)
        self.num_samples = int(num_samples)
        self.ee_position_bins = int(ee_position_bins)
        self.base_layout = infer_base_command_layout(
            command_ranges, use_5d_base_command
        )
        self.base_names = tuple(name for name, _ in self.base_layout)
        self.base_indices = tuple(index for _, index in self.base_layout)
        self.command_dim = 5 if use_5d_base_command else 3
        self.base_low, self.base_high = _range_tensor(command_ranges, self.base_names)
        self.ee_low, self.ee_high = _range_tensor(goal_ee_ranges, EE_COMMAND_NAMES)

        base_nominal_enabled = []
        for name, low, high in zip(self.base_names, self.base_low, self.base_high):
            nominal = 0.0 if float(low) <= 0.0 <= float(high) else float((low + high) / 2)
            if name == "base_height" and base_height_nominal is not None:
                nominal = min(max(float(base_height_nominal), float(low)), float(high))
            base_nominal_enabled.append(nominal)
        self.base_nominal_enabled = torch.tensor(base_nominal_enabled, dtype=torch.float64)
        self.base_nominal = torch.zeros(self.command_dim, dtype=torch.float64)
        for value, (_, index) in zip(self.base_nominal_enabled, self.base_layout):
            self.base_nominal[index] = value

        if ee_nominal is None:
            ee_nominal = [float((low + high) / 2) for low, high in zip(self.ee_low, self.ee_high)]
        if len(ee_nominal) == 3:
            ee_nominal = list(ee_nominal) + [0.0, 0.0, 0.0]
        if len(ee_nominal) != 6:
            raise ValueError("ee_nominal must have 3 or 6 values")
        self.ee_nominal = torch.minimum(
            torch.maximum(torch.tensor(ee_nominal, dtype=torch.float64), self.ee_low),
            self.ee_high,
        )

        base_enabled, ee_commands, sources = self._generate_schedule()
        self.base_commands = self._expand_base_commands(base_enabled).to(torch.float32)
        self.ee_commands = ee_commands.to(torch.float32)
        self.sources = tuple(sources)
        self.sample_ids = torch.arange(self.num_samples, dtype=torch.long)
        self._cursor = 0

        self._assert_inside_ranges()

    def _generate_schedule(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        base_dim = len(self.base_names)
        ee_dim = len(EE_COMMAND_NAMES)
        base_center_unit = torch.where(
            (self.base_high - self.base_low) > 0,
            (self.base_nominal_enabled - self.base_low) / (self.base_high - self.base_low),
            torch.full_like(self.base_low, 0.5),
        ).clamp(0.0, 1.0)
        ee_center_unit = torch.where(
            (self.ee_high - self.ee_low) > 0,
            (self.ee_nominal - self.ee_low) / (self.ee_high - self.ee_low),
            torch.full_like(self.ee_low, 0.5),
        ).clamp(0.0, 1.0)

        base_rows: List[torch.Tensor] = []
        ee_rows: List[torch.Tensor] = []
        sources: List[str] = []

        def append(unit_base: torch.Tensor, unit_ee: torch.Tensor, source: str) -> None:
            if len(base_rows) >= self.num_samples:
                return
            base_rows.append(_map_unit(unit_base, self.base_low, self.base_high))
            ee_rows.append(_map_unit(unit_ee, self.ee_low, self.ee_high))
            sources.append(source)

        base_boundary = _axis_boundary_samples(base_dim)
        ee_boundary = _axis_boundary_samples(ee_dim)

        if self.mode == "base":
            for row in base_boundary:
                append(row, ee_center_unit, "base_boundary")
        elif self.mode == "ee":
            for row in ee_boundary:
                append(base_center_unit, row, "ee_boundary")
        else:
            append(base_center_unit, ee_center_unit, "joint_center")
            boundary_count = max(len(base_boundary), len(ee_boundary))
            for index in range(boundary_count):
                append(
                    base_boundary[index % len(base_boundary)],
                    ee_boundary[index % len(ee_boundary)],
                    "joint_boundary",
                )

        if self.mode in ("ee", "joint") and len(base_rows) < self.num_samples:
            position_axes = [
                torch.linspace(0.0, 1.0, self.ee_position_bins, dtype=torch.float64)
                for _ in range(3)
            ]
            position_grid = torch.cartesian_prod(*position_axes)
            structured_limit = min(
                len(position_grid),
                max(0, self.num_samples // 2),
            )
            if self.num_samples >= len(position_grid) + len(ee_boundary):
                structured_limit = len(position_grid)
            if structured_limit > 0:
                auxiliary_dim = base_dim + 3
                auxiliary = torch.quasirandom.SobolEngine(
                    auxiliary_dim, scramble=True, seed=self.seed + 17
                ).draw(structured_limit).to(torch.float64)
                for index in range(structured_limit):
                    ee_unit = ee_center_unit.clone()
                    ee_unit[:3] = position_grid[index]
                    if self.mode == "joint":
                        base_unit = auxiliary[index, :base_dim]
                        ee_unit[3:] = auxiliary[index, base_dim:]
                    else:
                        base_unit = base_center_unit
                        ee_unit[3:] = auxiliary[index, base_dim:]
                    append(base_unit, ee_unit, "ee_position_grid")

        remaining = self.num_samples - len(base_rows)
        if remaining > 0:
            if self.mode == "base":
                sobol_dim = base_dim
            elif self.mode == "ee":
                sobol_dim = ee_dim
            else:
                sobol_dim = base_dim + ee_dim
            sobol = torch.quasirandom.SobolEngine(
                sobol_dim, scramble=True, seed=self.seed
            ).draw(remaining).to(torch.float64)
            for row in sobol:
                if self.mode == "base":
                    append(row, ee_center_unit, "sobol")
                elif self.mode == "ee":
                    append(base_center_unit, row, "sobol")
                else:
                    append(row[:base_dim], row[base_dim:], "sobol")

        return torch.stack(base_rows), torch.stack(ee_rows), sources

    def _expand_base_commands(self, enabled: torch.Tensor) -> torch.Tensor:
        commands = self.base_nominal.repeat(enabled.shape[0], 1)
        for source_col, (_, target_col) in enumerate(self.base_layout):
            commands[:, target_col] = enabled[:, source_col]
        return commands

    def _assert_inside_ranges(self) -> None:
        enabled = torch.stack(
            [self.base_commands[:, index] for index in self.base_indices], dim=-1
        ).to(torch.float64)
        tolerance = 1.0e-6
        if not torch.all(
            (enabled >= self.base_low - tolerance)
            & (enabled <= self.base_high + tolerance)
        ):
            raise AssertionError("Generated base commands escaped configured ranges")
        ee = self.ee_commands.to(torch.float64)
        if not torch.all(
            (ee >= self.ee_low - tolerance) & (ee <= self.ee_high + tolerance)
        ):
            raise AssertionError("Generated EE commands escaped configured ranges")

    @property
    def remaining(self) -> int:
        return self.num_samples - self._cursor

    def reset(self) -> None:
        self._cursor = 0

    def next_batch(self, max_samples: int) -> Optional[CoverageBatch]:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self._cursor >= self.num_samples:
            return None
        end = min(self.num_samples, self._cursor + int(max_samples))
        selection = slice(self._cursor, end)
        batch = CoverageBatch(
            sample_ids=self.sample_ids[selection].clone(),
            base_commands=self.base_commands[selection].clone(),
            ee_commands=self.ee_commands[selection].clone(),
            sources=self.sources[selection],
        )
        self._cursor = end
        return batch


class CoverageMetrics:
    """Track requested, feasible and executed coverage without step-wise IO."""

    def __init__(self, scheduler: EvaluationCoverageScheduler, num_bins: int = 10):
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        self.scheduler = scheduler
        self.num_bins = int(num_bins)
        self.records: List[Dict[str, Any]] = []
        for sample_id in range(scheduler.num_samples):
            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "source": scheduler.sources[sample_id],
                "status": "planned",
                "valid": None,
                "rejection_reason": "",
            }
            for name, index in scheduler.base_layout:
                row[f"scheduled_base/{name}"] = float(scheduler.base_commands[sample_id, index])
            for dim, name in enumerate(EE_COMMAND_NAMES):
                row[f"scheduled_ee/{name}"] = float(scheduler.ee_commands[sample_id, dim])
            self.records.append(row)

    def mark_rejected(self, sample_ids: Iterable[int], reasons: Iterable[str]) -> None:
        for sample_id, reason in zip(sample_ids, reasons):
            row = self.records[int(sample_id)]
            row.update(status="rejected", valid=False, rejection_reason=str(reason))
            for name in EE_COMMAND_NAMES:
                row[f"rejected_ee/{name}"] = row[f"scheduled_ee/{name}"]

    def mark_executed(self, sample_id: int, **outcome: Any) -> None:
        row = self.records[int(sample_id)]
        row.update(status="executed", valid=True)
        for name in EE_COMMAND_NAMES:
            row[f"valid_ee/{name}"] = row[f"scheduled_ee/{name}"]
        for key, value in outcome.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            row[key] = value

    def _occupancy(
        self,
        values: Sequence[float],
        low: float,
        high: float,
    ) -> Dict[str, Any]:
        occupied = [False] * self.num_bins
        in_range = 0
        for value in values:
            value = float(value)
            if not math.isfinite(value) or value < low or value > high:
                continue
            in_range += 1
            if high == low:
                index = 0
            else:
                index = min(self.num_bins - 1, int((value - low) / (high - low) * self.num_bins))
            occupied[index] = True
        count = sum(occupied)
        return {
            "coverage_percent": 100.0 * count / self.num_bins,
            "occupied_bins": count,
            "total_bins": self.num_bins,
            "in_range_samples": in_range,
        }

    def _pairwise_occupancy(
        self,
        x_values: Sequence[float],
        y_values: Sequence[float],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
    ) -> Dict[str, Any]:
        occupied = set()
        for x_value, y_value in zip(x_values, y_values):
            x_value, y_value = float(x_value), float(y_value)
            if not (math.isfinite(x_value) and math.isfinite(y_value)):
                continue
            if not (x_range[0] <= x_value <= x_range[1] and y_range[0] <= y_value <= y_range[1]):
                continue
            x_bin = 0 if x_range[1] == x_range[0] else min(
                self.num_bins - 1,
                int((x_value - x_range[0]) / (x_range[1] - x_range[0]) * self.num_bins),
            )
            y_bin = 0 if y_range[1] == y_range[0] else min(
                self.num_bins - 1,
                int((y_value - y_range[0]) / (y_range[1] - y_range[0]) * self.num_bins),
            )
            occupied.add((x_bin, y_bin))
        total = self.num_bins * self.num_bins
        return {
            "coverage_percent": 100.0 * len(occupied) / total,
            "occupied_bins": len(occupied),
            "total_bins": total,
        }

    def report(self) -> Dict[str, Any]:
        statuses = [row["status"] for row in self.records]
        executed_rows = [row for row in self.records if row["status"] == "executed"]
        valid_rows = [row for row in self.records if row.get("valid") is True]
        report: Dict[str, Any] = {
            "mode": self.scheduler.mode,
            "seed": self.scheduler.seed,
            "planned_samples": len(self.records),
            "valid_samples": len(valid_rows),
            "executed_samples": statuses.count("executed"),
            "rejected_samples": statuses.count("rejected"),
            "unfinished_samples": statuses.count("planned"),
            "num_coverage_bins": self.num_bins,
            "ranges": {
                "base": {
                    name: [
                        float(self.scheduler.base_low[index]),
                        float(self.scheduler.base_high[index]),
                    ]
                    for index, name in enumerate(self.scheduler.base_names)
                },
                "ee": {
                    name: [
                        float(self.scheduler.ee_low[index]),
                        float(self.scheduler.ee_high[index]),
                    ]
                    for index, name in enumerate(EE_COMMAND_NAMES)
                },
            },
            "requested": {},
            "valid_feasible": {},
            "executed": {},
            "achieved_ee": {},
            "successful_target_ee": {},
            "pairwise": {},
        }

        dimensions: List[Tuple[str, str, float, float]] = []
        for index, name in enumerate(self.scheduler.base_names):
            dimensions.append(("base", name, float(self.scheduler.base_low[index]), float(self.scheduler.base_high[index])))
        for index, name in enumerate(EE_COMMAND_NAMES):
            dimensions.append(("ee", name, float(self.scheduler.ee_low[index]), float(self.scheduler.ee_high[index])))

        for group, name, low, high in dimensions:
            key = f"scheduled_{group}/{name}"
            path = f"{group}/{name}"
            report["requested"][path] = self._occupancy(
                [row[key] for row in self.records], low, high
            )
            report["valid_feasible"][path] = self._occupancy(
                [row[key] for row in valid_rows], low, high
            )
            report["executed"][path] = self._occupancy(
                [row[key] for row in executed_rows], low, high
            )

        successful_rows = [row for row in executed_rows if bool(row.get("success", False))]
        for index, name in enumerate(EE_COMMAND_NAMES[:3]):
            low, high = float(self.scheduler.ee_low[index]), float(self.scheduler.ee_high[index])
            achieved_key = f"achieved_ee/{name}"
            report["achieved_ee"][name] = self._occupancy(
                [row[achieved_key] for row in executed_rows if achieved_key in row], low, high
            )
            target_key = f"scheduled_ee/{name}"
            report["successful_target_ee"][name] = self._occupancy(
                [row[target_key] for row in successful_rows], low, high
            )

        pair_specs = []
        if "lin_vel_x" in self.scheduler.base_names and "ang_vel_yaw" in self.scheduler.base_names:
            pair_specs.append(("base", "lin_vel_x", "ang_vel_yaw"))
        pair_specs.extend(
            [
                ("ee", "pos_l", "pos_p"),
                ("ee", "pos_l", "pos_y"),
                ("ee", "pos_p", "pos_y"),
            ]
        )
        base_range_by_name = {
            name: (float(self.scheduler.base_low[i]), float(self.scheduler.base_high[i]))
            for i, name in enumerate(self.scheduler.base_names)
        }
        ee_range_by_name = {
            name: (float(self.scheduler.ee_low[i]), float(self.scheduler.ee_high[i]))
            for i, name in enumerate(EE_COMMAND_NAMES)
        }
        for group, x_name, y_name in pair_specs:
            ranges = base_range_by_name if group == "base" else ee_range_by_name
            x_key, y_key = f"scheduled_{group}/{x_name}", f"scheduled_{group}/{y_name}"
            report["pairwise"][f"{group}/{x_name}_x_{y_name}"] = self._pairwise_occupancy(
                [row[x_key] for row in executed_rows],
                [row[y_key] for row in executed_rows],
                ranges[x_name],
                ranges[y_name],
            )
        return report

    def save(self, out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "coverage_report.json")
        samples_path = os.path.join(out_dir, "coverage_samples.csv")
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(self.report(), file, indent=2, sort_keys=True, allow_nan=True)

        fieldnames = sorted({key for row in self.records for key in row})
        with open(samples_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        return {"json": report_path, "csv": samples_path}


__all__ = [
    "BASE_COMMAND_LAYOUT",
    "EE_COMMAND_NAMES",
    "CoverageBatch",
    "CoverageMetrics",
    "EvaluationCoverageScheduler",
    "infer_base_command_layout",
    "resolve_play_num_envs_value",
]
