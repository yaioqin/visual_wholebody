#!/usr/bin/env python3
"""Build a self-contained comparison report for coordination evaluation logs.

The script recursively scans a log root such as:

    low-level/legged_gym/scripts/low-level/logs/b1z1-low

Each model/checkpoint directory may contain one or more
``coordination_eval_*.json`` files. By default, the latest file in each
evaluation directory is used, so repeated evaluations do not create duplicate
rows. New model folders or checkpoint folders are picked up automatically.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


EVAL_FILE_RE = re.compile(r"coordination_eval_(\d{8}_\d{6})\.(json|csv)$")


DEFAULT_CHART_METRICS = [
    "ee/success_rate_step",
    "ee/success_rate_episode",
    "ee/pos_mean",
    "ee/ori_geodesic_mean",
    "ee/time_to_success_mean",
    "vel/l1_mean",
    "vel/vx_mae",
    "vel/yaw_mae",
    "stability/survival_rate_step",
    "stability/fall_rate_step",
    "stability/base_ang_acc_rms",
    "stability/base_lin_acc_rms",
    "energy/total_energy_per_step",
    "energy/total_power_abs_mean",
    "smoothness/total_action_rate_mean",
    "smoothness/arm_motion_rate_mean",
    "coordination/survival_when_arm_large",
    "coordination/base_ang_acc_when_arm_large",
    "coordination/ee_pos_err_when_arm_large",
    "coordination/vel_err_when_arm_large",
    "workspace/solvability_rate",
    "workspace/hull_volume",
    "workspace/success_points",
]


EXPLICIT_DIRECTIONS = {
    "ee/pos_mean": "lower",
    "ee/pos_median": "lower",
    "ee/pos_p90": "lower",
    "ee/pos_max": "lower",
    "ee/pos_rmse": "lower",
    "ee/success_rate_step": "higher",
    "ee/success_rate_episode": "higher",
    "stability/survival_rate_step": "higher",
    "stability/episode_survival_rate": "higher",
    "coordination/survival_when_arm_large": "higher",
    "workspace/solvability_rate": "higher",
    "workspace/success_points": "higher",
    "workspace/hull_volume": "higher",
    "workspace/hull_area_xy": "higher",
    "workspace/hull_area_xz": "higher",
    "stability/base_height_min": "higher",
    "stability/base_height_p05": "higher",
}


DIRECTION_LABELS = {
    "higher": "\u8d8a\u5927\u8d8a\u597d / higher is better",
    "lower": "\u8d8a\u5c0f\u8d8a\u597d / lower is better",
    "zero": "\u8d8a\u63a5\u8fd1 0 \u8d8a\u597d / closer to 0 is better",
    "neutral": "\u4ec5\u4f9b\u53c2\u8003 / reference only",
}


DIRECTION_CSS_CLASS = {
    "higher": "direction-higher",
    "lower": "direction-lower",
    "zero": "direction-zero",
    "neutral": "direction-neutral",
}


GROUP_LABELS = {
    "coordination": "协调性 / coordination",
    "ee": "末端执行器 / end-effector",
    "energy": "能耗 / energy",
    "meta": "评估元信息 / metadata",
    "smoothness": "动作平滑度 / smoothness",
    "stability": "稳定性 / stability",
    "vel": "速度跟踪 / velocity tracking",
    "workspace": "可达工作空间 / workspace",
}


GROUP_DESCRIPTIONS = {
    "coordination": "衡量机械臂运动、末端误差和机身运动之间的耦合关系；常用于看手臂动作是否扰动了底盘。",
    "ee": "衡量末端执行器实际位姿和目标位姿之间的误差，以及达到目标的成功率和耗时。",
    "energy": "根据关节力矩和关节速度估算功率或能量；当前臂部能耗可能使用位置控制力矩代理值。",
    "meta": "评估设置本身，不直接代表模型优劣。",
    "smoothness": "衡量相邻 step 的 action 或 arm target 变化速度；通常越小代表动作越平滑。",
    "stability": "衡量机器人是否存活、是否摔倒、机身高度和机身加速度等稳定性信号。",
    "vel": "衡量机身实际速度和命令速度之间的跟踪误差。",
    "workspace": "衡量成功到达目标时末端覆盖的空间范围和可解比例。",
}


METRIC_EXPLANATIONS = {
    "coordination/base_ang_acc_when_arm_large": "当机械臂运动幅度处于本次评估前 25% 大的 step 时，机身角加速度范数的平均值。数值越小，说明大幅动臂时底盘转动扰动越小。",
    "coordination/base_lin_acc_when_arm_large": "当机械臂运动幅度处于本次评估前 25% 大的 step 时，机身线加速度范数的平均值。数值越小，说明大幅动臂时底盘平动扰动越小。",
    "coordination/ee_pos_err_when_arm_large": "当机械臂运动幅度处于本次评估前 25% 大的 step 时，末端位置误差的平均值。数值越小，说明大幅动臂时末端仍能跟准目标。",
    "coordination/vel_err_when_arm_large": "当机械臂运动幅度处于本次评估前 25% 大的 step 时，机身速度 L1 跟踪误差的平均值。数值越小，说明动臂时行走速度更不容易被干扰。",
    "coordination/survival_when_arm_large": "当机械臂运动幅度处于本次评估前 25% 大的 step 时，机器人仍处于存活状态的比例。数值越大越好。",
    "coordination/corr_arm_action_norm_base_ang_acc": "机械臂 action 幅度和机身角加速度之间的 Pearson 相关系数。越接近 0，表示动臂幅度和底盘角加速度耦合越弱。",
    "coordination/corr_arm_motion_norm_base_ang_acc": "机械臂目标/运动信号幅度和机身角加速度之间的 Pearson 相关系数。越接近 0，表示手臂运动和底盘角加速度耦合越弱。",
    "coordination/corr_ee_pos_err_base_ang_acc": "末端位置误差和机身角加速度之间的 Pearson 相关系数。越接近 0，表示末端误差不太随底盘转动抖动一起变大。",
    "coordination/corr_target_ee_x_base_pitch": "目标末端 x 坐标和机身 pitch 之间的 Pearson 相关系数。越接近 0，表示目标前后位置变化和机身俯仰耦合越弱。",
    "coordination/corr_target_ee_y_base_roll": "目标末端 y 坐标和机身 roll 之间的 Pearson 相关系数。越接近 0，表示目标左右位置变化和机身横滚耦合越弱。",
    "coordination/corr_target_ee_z_base_pitch": "目标末端 z 坐标和机身 pitch 之间的 Pearson 相关系数。越接近 0，表示目标高度变化和机身俯仰耦合越弱。",
    "ee/success_rate_step": "逐 step 成功率：该 step 中末端位置误差和姿态误差都低于阈值，且机器人未失败的比例。数值越大越好。",
    "ee/success_rate_episode": "逐 episode 成功率：一个 episode 内至少成功到达过一次目标的比例。数值越大越好。",
    "ee/time_to_success_mean": "从 episode 开始到首次成功到达目标所需时间的平均值。只统计成功过的 episode；越小代表更快到达。",
    "ee/time_to_success_median": "从 episode 开始到首次成功到达目标所需时间的中位数。只统计成功过的 episode；越小代表更快到达。",
    "stability/survival_rate_step": "逐 step 存活率：该 step 未发生 reset/termination/碰撞等失败事件的比例。数值越大越好。",
    "stability/fall_rate_step": "逐 step 失败/摔倒率：发生 reset/termination/碰撞等失败事件的比例。数值越小越好。",
    "stability/episode_survival_rate": "逐 episode 存活率：整个 episode 内一直保持存活的比例。数值越大越好。",
    "workspace/solvability_rate": "可解率；当前实现用 episode 成功率作为兜底，即有多少 episode 至少成功到达过目标。数值越大越好。",
    "workspace/success_points": "成功 step 中收集到的末端位置点数量。更多点通常说明成功样本更充分，但也会受评估时长影响。",
    "workspace/hull_volume": "成功末端位置点在 3D 空间中的凸包体积。数值越大，表示成功覆盖的三维工作空间越大。",
    "workspace/hull_area_xy": "成功末端位置点投影到 xy 平面的凸包面积。数值越大，表示水平面成功覆盖范围越大。",
    "workspace/hull_area_xz": "成功末端位置点投影到 xz 平面的凸包面积。数值越大，表示前后-高度平面成功覆盖范围越大。",
    "meta/num_envs": "并行评估环境数量。它描述评估配置，不代表模型好坏。",
    "meta/eval_steps": "去掉 warmup 后实际纳入统计的 rollout step 数。它描述评估长度，不代表模型好坏。",
    "meta/warmup_steps": "评估前丢弃、不纳入统计的 warmup step 数。它描述评估配置，不代表模型好坏。",
    "meta/dt": "仿真控制周期，单位秒。它描述评估配置，不代表模型好坏。",
    "meta/total_samples": "纳入统计的总样本数，通常等于 eval_steps * num_envs。它描述统计样本量，不代表模型好坏。",
}


METRIC_BASE_EXPLANATIONS = {
    "vel/vx": "机身 x 方向速度跟踪误差，来自实际 base_lin_vel_x 与命令 command_x 的绝对差。",
    "vel/vy": "机身 y 方向速度跟踪误差，来自实际 base_lin_vel_y 与命令 command_y 的绝对差。",
    "vel/yaw": "机身 yaw 角速度跟踪误差，来自实际 base_ang_vel_z 与命令 yaw 速度的绝对差。",
    "vel/l1": "速度总误差，把可用的 vx、vy、yaw 绝对误差相加得到。",
    "ee/pos": "末端位置误差，实际末端 xyz 和目标末端 xyz 之间的欧氏距离，单位通常是米。",
    "ee/l1": "末端位置 L1 误差，实际末端 xyz 和目标末端 xyz 的逐轴绝对误差之和，单位通常是米。",
    "ee/ori_geodesic": "末端姿态误差；四元数时使用 geodesic distance，RPY 时使用 wrap 后的角度误差范数，单位通常是弧度。",
    "stability/base_height": "机身高度统计，单位通常是米。高度过低容易表示蹲塌或摔倒；均值本身更多用于参考。",
    "stability/base_ang_acc": "机身角加速度范数，由相邻 step 的 base_ang_vel 差分除以 dt 得到，越小通常越稳。",
    "stability/base_lin_acc": "机身线加速度范数，由相邻 step 的 base_lin_vel 差分除以 dt 得到，越小通常越稳。",
    "energy/leg_power_abs": "腿部关节绝对功率之和，按 abs(torque * dof_vel) 估算。",
    "energy/arm_power_abs": "机械臂关节绝对功率之和，按 abs(torque * dof_vel) 或位置控制力矩代理值估算。",
    "energy/total_power_abs": "腿部和机械臂合计绝对功率之和。",
    "energy/leg_power_squared": "腿部关节功率平方和，对尖峰功率更敏感。",
    "energy/arm_power_squared": "机械臂关节功率平方和，对尖峰功率更敏感。",
    "energy/total_power_squared": "全身关节功率平方和，对尖峰功率更敏感。",
    "energy/leg_energy": "腿部累计能量，由腿部绝对功率随时间积分得到。",
    "energy/arm_energy": "机械臂累计能量，由机械臂绝对功率随时间积分得到。",
    "energy/total_energy": "全身累计能量，由总绝对功率随时间积分得到。",
    "smoothness/leg_action_rate": "腿部 action 相邻 step 变化量的范数除以 dt。越小代表腿部动作更平滑。",
    "smoothness/arm_action_rate": "机械臂 action/target 相邻 step 变化量的范数除以 dt。越小代表机械臂动作更平滑。",
    "smoothness/arm_motion_rate": "机械臂目标/运动信号相邻 step 变化量的范数除以 dt。越小代表手臂目标变化更平滑。",
    "smoothness/total_action_rate": "全身 action 相邻 step 变化量的范数除以 dt。越小代表整体控制输出更平滑。",
}


STAT_EXPLANATIONS = {
    "mean": "平均值，反映整体水平。",
    "mae": "平均绝对误差，反映平均跟踪偏差。",
    "rmse": "均方根误差/均方根值，对较大的误差或抖动更敏感。",
    "rms": "均方根值，对较大的加速度或抖动更敏感。",
    "median": "中位数，反映典型样本表现，比平均值更不容易被极端值影响。",
    "p90": "90% 分位数，表示 90% 样本不超过该值，用来看较差但非最极端的情况。",
    "p05": "5% 分位数，常用于看高度等安全下界。",
    "max": "最大值，用来看最坏瞬时误差或峰值。",
    "min": "最小值，用来看最低高度等下界。",
    "sum": "累计和，通常会随评估时长和环境数量增加而变大。",
}


EXACT_ENERGY_EXPLANATIONS = {
    "energy/total_energy_per_step": "全身累计能量除以总统计样本数，便于比较不同评估长度下的单位 step 能耗。越小越省能。",
    "energy/total_energy_per_episode": "全身累计能量除以 episode 数，便于比较单个 episode 平均能耗。越小越省能。",
}


@dataclass
class EvalRecord:
    label: str
    model: str
    eval_name: str
    timestamp: str
    path: Path
    metrics: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "low-level" / "logs" / "b1z1-low"

    parser = argparse.ArgumentParser(
        description=(
            "Compare coordination_eval JSON/CSV files and generate an HTML "
            "model comparison report."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Evaluation log root to scan. Default: {default_root}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path. Default: <root>/model_comparison_report.html",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CSV output path. Default: <root>/model_comparison_summary.csv",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_CHART_METRICS),
        help=(
            "Comma-separated metrics to chart, or 'all' for every numeric "
            "metric. The full numeric table is always written."
        ),
    )
    parser.add_argument(
        "--all-evals",
        action="store_true",
        help="Include every coordination_eval file instead of latest per eval directory.",
    )
    parser.add_argument(
        "--latest-per-model",
        action="store_true",
        help="After scanning, keep only the latest evaluation for each model folder.",
    )
    parser.add_argument(
        "--max-charts",
        type=int,
        default=None,
        help="Limit the number of chart cards rendered in the HTML report.",
    )
    return parser.parse_args()


def load_eval_file(path: Path) -> Dict[str, Any]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    metrics: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and {"key", "value"}.issubset(reader.fieldnames):
            for row in reader:
                metrics[row["key"]] = parse_scalar(row["value"])
        else:
            f.seek(0)
            simple_reader = csv.reader(f)
            next(simple_reader, None)
            for row in simple_reader:
                if len(row) >= 2:
                    metrics[row[0]] = parse_scalar(row[1])
    return metrics


def parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "nan":
        return float("nan")
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return value


def timestamp_from_path(path: Path) -> str:
    match = EVAL_FILE_RE.search(path.name)
    if match:
        return match.group(1)
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")


def timestamp_sort_key(timestamp: str) -> Tuple[int, str]:
    try:
        return (1, datetime.strptime(timestamp, "%Y%m%d_%H%M%S").isoformat())
    except ValueError:
        return (0, timestamp)


def discover_eval_files(root: Path, include_all: bool) -> List[Path]:
    json_files = sorted(root.rglob("coordination_eval_*.json"))
    csv_files = sorted(root.rglob("coordination_eval_*.csv"))

    json_stems = {p.with_suffix("") for p in json_files}
    files: List[Path] = list(json_files)
    files.extend(p for p in csv_files if p.with_suffix("") not in json_stems)

    if include_all:
        return sorted(files, key=lambda p: (str(p.parent), timestamp_from_path(p), p.name))

    latest_by_dir: Dict[Path, Path] = {}
    for path in files:
        current = latest_by_dir.get(path.parent)
        if current is None:
            latest_by_dir[path.parent] = path
            continue
        if timestamp_sort_key(timestamp_from_path(path)) > timestamp_sort_key(timestamp_from_path(current)):
            latest_by_dir[path.parent] = path
    return sorted(latest_by_dir.values(), key=lambda p: str(p.relative_to(root)))


def record_from_path(root: Path, path: Path) -> EvalRecord:
    rel = path.relative_to(root)
    parts = rel.parts
    model = parts[0] if parts else path.parent.name
    eval_name = "/".join(parts[1:-1]) if len(parts) > 2 else path.parent.name
    timestamp = timestamp_from_path(path)
    label_parts = [model]
    if eval_name and eval_name != model:
        label_parts.append(eval_name)
    label = " / ".join(label_parts)
    return EvalRecord(
        label=label,
        model=model,
        eval_name=eval_name,
        timestamp=timestamp,
        path=path,
        metrics=load_eval_file(path),
    )


def discover_records(root: Path, include_all: bool, latest_per_model: bool) -> List[EvalRecord]:
    if not root.exists():
        raise FileNotFoundError(f"Log root does not exist: {root}")

    records = [record_from_path(root, path) for path in discover_eval_files(root, include_all)]
    if latest_per_model:
        latest: Dict[str, EvalRecord] = {}
        for record in records:
            current = latest.get(record.model)
            if current is None:
                latest[record.model] = record
                continue
            if timestamp_sort_key(record.timestamp) > timestamp_sort_key(current.timestamp):
                latest[record.model] = record
        records = list(latest.values())

    return sorted(records, key=lambda r: (r.model.lower(), r.eval_name.lower(), r.timestamp))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_float(value: Any) -> Optional[float]:
    if not is_number(value):
        return None
    number = float(value)
    if math.isfinite(number):
        return number
    return None


def numeric_metrics(records: Sequence[EvalRecord]) -> List[str]:
    keys = set()
    for record in records:
        for key, value in record.metrics.items():
            if finite_float(value) is not None:
                keys.add(key)
    return sorted(keys)


def metric_direction(metric: str) -> str:
    if metric in EXPLICIT_DIRECTIONS:
        return EXPLICIT_DIRECTIONS[metric]

    lower_metric = metric.lower()
    if lower_metric.startswith("meta/"):
        return "neutral"
    if "/corr_" in lower_metric or lower_metric.startswith("coordination/corr_"):
        return "zero"

    lower_tokens = (
        "err",
        "error",
        "mae",
        "rmse",
        "l1",
        "geodesic",
        "fall_rate",
        "time_to_success",
        "energy",
        "power",
        "action_rate",
        "motion_rate",
        "acc",
    )
    if any(token in lower_metric for token in lower_tokens):
        return "lower"

    higher_tokens = (
        "success_rate",
        "survival_rate",
        "survival_when",
        "solvability",
        "success_points",
        "hull_",
    )
    if any(token in lower_metric for token in higher_tokens):
        return "higher"

    if lower_metric.endswith("_max") or lower_metric.endswith("_p90"):
        return "lower"
    return "neutral"


def metric_group(metric: str) -> str:
    if "/" not in metric:
        return "other"
    return metric.split("/", 1)[0]


def split_metric_stat(metric: str) -> Tuple[str, Optional[str]]:
    for suffix in ("_median", "_rmse", "_mean", "_mae", "_p90", "_p05", "_max", "_min", "_rms", "_sum"):
        if metric.endswith(suffix):
            return metric[: -len(suffix)], suffix[1:]
    return metric, None


def metric_explanation(metric: str) -> str:
    if metric in EXACT_ENERGY_EXPLANATIONS:
        return EXACT_ENERGY_EXPLANATIONS[metric]
    if metric in METRIC_EXPLANATIONS:
        return METRIC_EXPLANATIONS[metric]

    base, stat = split_metric_stat(metric)
    base_text = METRIC_BASE_EXPLANATIONS.get(base)
    if base_text is not None:
        if stat is None:
            return base_text
        stat_text = STAT_EXPLANATIONS.get(stat, f"{stat} 统计量。")
        return f"{base_text} 统计口径：{stat_text}"

    group = metric_group(metric)
    group_text = GROUP_DESCRIPTIONS.get(group)
    if group_text is not None:
        return f"{group_text} 这个指标没有单独写入词典，请结合 metric 名称和方向标签理解。"
    return "这个指标没有单独写入词典，请结合 metric 名称、数值方向和生成该评估的代码理解。"


def direction_hint(direction: str) -> str:
    if direction == "higher":
        return "比较模型时，数值越大通常排名越好。"
    if direction == "lower":
        return "比较模型时，数值越小通常排名越好。"
    if direction == "zero":
        return "比较模型时，绝对值越接近 0 通常越好，表示相关性或耦合更弱。"
    return "该值主要用于检查评估设置或提供背景，不直接参与优劣判断。"


def metric_info(metric: str) -> Dict[str, str]:
    group = metric_group(metric)
    direction = metric_direction(metric)
    return {
        "metric": metric,
        "group": GROUP_LABELS.get(group, group),
        "direction": DIRECTION_LABELS[direction],
        "meaning": metric_explanation(metric),
        "hint": direction_hint(direction),
    }


def value_for(record: EvalRecord, metric: str) -> Optional[float]:
    return finite_float(record.metrics.get(metric))


def format_number(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value < 0.001 or abs_value >= 10000:
        return f"{value:.3e}"
    if abs_value < 1:
        return f"{value:.4f}"
    return f"{value:.3f}"


def best_indices(records: Sequence[EvalRecord], metric: str) -> List[int]:
    direction = metric_direction(metric)
    values = [(idx, value_for(record, metric)) for idx, record in enumerate(records)]
    finite_values = [(idx, value) for idx, value in values if value is not None]
    if not finite_values or direction == "neutral":
        return []

    if direction == "higher":
        best_value = max(value for _, value in finite_values)
        return [idx for idx, value in finite_values if math.isclose(value, best_value)]
    if direction == "lower":
        best_value = min(value for _, value in finite_values)
        return [idx for idx, value in finite_values if math.isclose(value, best_value)]
    if direction == "zero":
        best_abs = min(abs(value) for _, value in finite_values)
        return [idx for idx, value in finite_values if math.isclose(abs(value), best_abs)]
    return []


def normalized_scores(values: Sequence[Optional[float]], direction: str) -> List[float]:
    finite_values = [v for v in values if v is not None]
    if not finite_values:
        return [0.0 for _ in values]

    if direction == "zero":
        max_abs = max(abs(v) for v in finite_values)
        if max_abs == 0:
            return [1.0 if v is not None else 0.0 for v in values]
        return [max(0.0, 1.0 - abs(v) / max_abs) if v is not None else 0.0 for v in values]

    min_value = min(finite_values)
    max_value = max(finite_values)
    if math.isclose(min_value, max_value):
        return [1.0 if v is not None else 0.0 for v in values]

    if direction == "lower":
        return [(max_value - v) / (max_value - min_value) if v is not None else 0.0 for v in values]

    return [(v - min_value) / (max_value - min_value) if v is not None else 0.0 for v in values]


def choose_chart_metrics(records: Sequence[EvalRecord], metric_arg: str) -> List[str]:
    available = set(numeric_metrics(records))
    if metric_arg.strip().lower() == "all":
        return sorted(available)

    metrics = []
    for metric in metric_arg.split(","):
        metric = metric.strip()
        if metric and metric in available:
            metrics.append(metric)
    return metrics


def csv_value(value: Any) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return str(value)


def write_summary_csv(path: Path, records: Sequence[EvalRecord], metrics: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "model", "eval_name", "timestamp", "source_path", *metrics]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "label": record.label,
                "model": record.model,
                "eval_name": record.eval_name,
                "timestamp": record.timestamp,
                "source_path": str(record.path),
            }
            for metric in metrics:
                row[metric] = csv_value(record.metrics.get(metric, ""))
            writer.writerow(row)


def write_direction_csv(path: Path, metrics: Sequence[str]) -> None:
    direction_path = path.with_name(path.stem + "_metric_directions.csv")
    with direction_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "direction", "label", "description"])
        for metric in metrics:
            direction = metric_direction(metric)
            writer.writerow([metric, direction, DIRECTION_LABELS[direction], metric_explanation(metric)])


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def render_metric_button(metric: str, heading: bool = False) -> str:
    class_name = "metric-button metric-button-heading" if heading else "metric-button"
    escaped_metric = html.escape(metric, quote=True)
    return (
        f'<button type="button" class="{class_name}" data-metric="{escaped_metric}" '
        f'aria-label="Show explanation for {escaped_metric}">'
        f"<code>{html.escape(metric)}</code><span class=\"help-dot\">?</span>"
        "</button>"
    )


def render_chart(records: Sequence[EvalRecord], metric: str) -> str:
    direction = metric_direction(metric)
    values = [value_for(record, metric) for record in records]
    scores = normalized_scores(values, direction)
    best = set(best_indices(records, metric))
    css_class = DIRECTION_CSS_CLASS[direction]
    label = DIRECTION_LABELS[direction]

    rows = []
    for idx, (record, value, score) in enumerate(zip(records, values, scores)):
        width = max(0.02, score) * 100 if value is not None else 0
        best_class = " is-best" if idx in best else ""
        best_text = " best" if idx in best else ""
        rows.append(
            "\n".join(
                [
                    f'<div class="bar-row{best_class}">',
                    f'  <div class="bar-label" title="{html.escape(record.label)}">{html.escape(record.label)}</div>',
                    '  <div class="bar-track">',
                    f'    <div class="bar-fill {css_class}" style="width: {width:.2f}%"></div>',
                    "  </div>",
                    f'  <div class="bar-value">{html.escape(format_number(value))}{best_text}</div>',
                    "</div>",
                ]
            )
        )

    return "\n".join(
        [
            '<section class="card">',
            '<div class="card-head">',
            f"<h2>{render_metric_button(metric, heading=True)}</h2>",
            f'<span class="pill {css_class}">{html.escape(label)}</span>',
            "</div>",
            '<div class="chart">',
            "\n".join(rows),
            "</div>",
            "</section>",
        ]
    )


def render_records_table(root: Path, records: Sequence[EvalRecord]) -> str:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record.label)}</td>"
            f"<td>{html.escape(record.model)}</td>"
            f"<td>{html.escape(record.eval_name)}</td>"
            f"<td>{html.escape(record.timestamp)}</td>"
            f"<td><code>{html.escape(relative_or_absolute(record.path, root))}</code></td>"
            "</tr>"
        )
    return (
        '<section class="panel">'
        "<h2>Included Evaluations</h2>"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr><th>label</th><th>model</th><th>eval</th><th>timestamp</th><th>source</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def render_metric_table(records: Sequence[EvalRecord], metrics: Sequence[str]) -> str:
    header_cells = [
        "<th>metric</th>",
        "<th>direction</th>",
        *[f"<th>{html.escape(record.label)}</th>" for record in records],
    ]

    rows = []
    current_group = None
    for metric in metrics:
        group = metric_group(metric)
        if group != current_group:
            current_group = group
            rows.append(
                f'<tr class="group-row"><td colspan="{2 + len(records)}">'
                f"{html.escape(group)}</td></tr>"
            )

        best = set(best_indices(records, metric))
        direction = metric_direction(metric)
        cells = [
            f"<td>{render_metric_button(metric)}</td>",
            (
                f'<td><span class="pill {DIRECTION_CSS_CLASS[direction]}">'
                f"{html.escape(DIRECTION_LABELS[direction])}</span></td>"
            ),
        ]
        for idx, record in enumerate(records):
            value = value_for(record, metric)
            cell_class = ' class="best-cell"' if idx in best else ""
            cells.append(f"<td{cell_class}>{html.escape(format_number(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<section class="panel">'
        "<h2>All Numeric Metrics</h2>"
        '<div class="table-wrap metric-table">'
        "<table>"
        f"<thead><tr>{''.join(header_cells)}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def render_direction_legend() -> str:
    items = []
    for direction in ("higher", "lower", "zero", "neutral"):
        items.append(
            f'<span class="pill {DIRECTION_CSS_CLASS[direction]}">'
            f"{html.escape(DIRECTION_LABELS[direction])}</span>"
        )
    return (
        '<section class="legend">'
        "<h2>Direction Labels</h2>"
        "<p>Chart bars are normalized by each metric direction, so longer bars are better when a direction is defined. Raw values are shown beside each bar and in the table. Click any metric name to see what it means.</p>"
        f'<div class="legend-pills">{"".join(items)}</div>'
        "</section>"
    )


def render_html(
    root: Path,
    output_csv: Path,
    records: Sequence[EvalRecord],
    chart_metrics: Sequence[str],
    all_metrics: Sequence[str],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    charts = "\n".join(render_chart(records, metric) for metric in chart_metrics)
    if not charts:
        charts = '<section class="panel"><p>No requested chart metrics were found.</p></section>'
    metric_info_json = json.dumps(
        {metric: metric_info(metric) for metric in all_metrics},
        ensure_ascii=False,
        sort_keys=True,
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>B1Z1 Low-Level Model Comparison</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #687384;
      --line: #dde3ea;
      --higher: #18865b;
      --lower: #2868b7;
      --zero: #8a5b12;
      --neutral: #657080;
      --best: #fff4c2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 28px 24px 48px; }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 20px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 17px; letter-spacing: 0; }}
    p {{ color: var(--muted); margin: 6px 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    button {{ font: inherit; }}
    .metric-button {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      max-width: 100%;
      padding: 0;
      border: 0;
      background: transparent;
      color: #1f4f8f;
      cursor: pointer;
      text-align: left;
      vertical-align: middle;
    }}
    .metric-button code {{
      color: inherit;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .metric-button-heading code {{
      font-size: 15px;
      font-weight: 700;
    }}
    .metric-button:hover code,
    .metric-button:focus-visible code {{
      text-decoration: underline;
    }}
    .metric-button:focus-visible {{
      outline: 2px solid #79a8e8;
      outline-offset: 3px;
      border-radius: 4px;
    }}
    .help-dot {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
      width: 17px;
      height: 17px;
      border-radius: 999px;
      background: #e9f1fb;
      color: #275f9f;
      font-size: 11px;
      font-weight: 700;
      line-height: 1;
    }}
    .meta {{
      text-align: right;
      color: var(--muted);
      font-size: 13px;
    }}
    .panel, .legend, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(22, 32, 42, 0.04);
    }}
    .legend, .panel {{ padding: 18px; margin-bottom: 18px; }}
    .legend-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .card {{ padding: 16px; min-width: 0; }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 9px;
      border-radius: 999px;
      color: #fff;
      font-size: 12px;
      white-space: nowrap;
    }}
    .direction-higher {{ background: var(--higher); }}
    .direction-lower {{ background: var(--lower); }}
    .direction-zero {{ background: var(--zero); }}
    .direction-neutral {{ background: var(--neutral); }}
    .chart {{ display: grid; gap: 9px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(120px, 1.1fr) minmax(160px, 2.2fr) minmax(92px, auto);
      gap: 10px;
      align-items: center;
      min-height: 28px;
    }}
    .bar-label {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #26313d;
      font-size: 13px;
    }}
    .bar-track {{
      height: 12px;
      background: #eef2f6;
      border: 1px solid #dfe5ec;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{ height: 100%; min-width: 2px; border-radius: 999px; }}
    .bar-value {{
      color: #26313d;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      text-align: right;
      white-space: nowrap;
    }}
    .is-best .bar-label, .is-best .bar-value {{ font-weight: 700; }}
    .is-best .bar-track {{ outline: 2px solid rgba(255, 196, 0, 0.65); }}
    .table-wrap {{
      overflow-x: auto;
      scrollbar-gutter: stable;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{
      color: #445162;
      background: #f9fafb;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .metric-table td:not(:first-child):not(:nth-child(2)),
    .metric-table th:not(:first-child):not(:nth-child(2)) {{
      text-align: right;
    }}
    .metric-table table {{
      width: max-content;
      min-width: 100%;
    }}
    .metric-table th,
    .metric-table td {{
      height: 42px;
    }}
    .metric-table th:first-child,
    .metric-table tbody tr:not(.group-row) td:first-child {{
      position: sticky;
      left: 0;
      z-index: 2;
      min-width: 360px;
      max-width: 360px;
      background: #ffffff;
      box-shadow: 1px 0 0 var(--line);
    }}
    .metric-table th:first-child {{
      z-index: 4;
      background: #f9fafb;
    }}
    .metric-table th:nth-child(2),
    .metric-table tbody tr:not(.group-row) td:nth-child(2) {{
      min-width: 230px;
      text-align: left;
    }}
    .metric-table th:not(:first-child):not(:nth-child(2)),
    .metric-table tbody tr:not(.group-row) td:not(:first-child):not(:nth-child(2)) {{
      min-width: 180px;
    }}
    .metric-table .metric-button code {{
      white-space: nowrap;
      overflow-wrap: normal;
    }}
    .metric-table tbody tr:not(.group-row):nth-child(even) td {{
      background: #fcfdff;
    }}
    .metric-table tbody tr:not(.group-row):hover td {{
      background: #f3f7fb;
    }}
    .metric-table tbody tr:not(.group-row) td.best-cell,
    .metric-table tbody tr:not(.group-row):nth-child(even) td.best-cell,
    .metric-table tbody tr:not(.group-row):hover td.best-cell {{
      background: var(--best);
      font-weight: 700;
    }}
    .group-row td {{
      background: #eef3f8;
      color: #344154;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .metric-modal[hidden] {{ display: none; }}
    .metric-modal {{
      position: fixed;
      inset: 0;
      z-index: 20;
      display: grid;
      place-items: center;
      padding: 20px;
    }}
    .modal-backdrop {{
      position: absolute;
      inset: 0;
      background: rgba(13, 22, 32, 0.48);
    }}
    .modal-panel {{
      position: relative;
      width: min(680px, 100%);
      max-height: min(80vh, 720px);
      overflow: auto;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 24px 70px rgba(13, 22, 32, 0.28);
      padding: 22px;
    }}
    .modal-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .modal-title {{
      margin: 0;
      font-size: 20px;
      overflow-wrap: anywhere;
    }}
    .modal-close {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f7f9fb;
      color: #344154;
      cursor: pointer;
      font-size: 18px;
      line-height: 1;
    }}
    .modal-close:hover,
    .modal-close:focus-visible {{
      background: #eef3f8;
    }}
    .info-block {{
      display: grid;
      gap: 5px;
      padding: 12px 0;
      border-top: 1px solid var(--line);
    }}
    .info-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .info-value {{
      color: #26313d;
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 720px) {{
      main {{ padding: 20px 14px 36px; }}
      header {{ display: block; }}
      .meta {{ text-align: left; margin-top: 10px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 5px; }}
      .bar-value {{ text-align: left; }}
      .modal-panel {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>B1Z1 Low-Level Model Comparison</h1>
      <p>Scanned root: <code>{html.escape(str(root))}</code></p>
      <p>CSV summary: <code>{html.escape(str(output_csv))}</code></p>
    </div>
    <div class="meta">
      <div>Generated: {html.escape(now)}</div>
      <div>{len(records)} evaluations, {len(all_metrics)} numeric metrics</div>
    </div>
  </header>
  {render_direction_legend()}
  {render_records_table(root, records)}
  <div class="grid">
    {charts}
  </div>
  {render_metric_table(records, all_metrics)}
</main>
<div id="metric-modal" class="metric-modal" hidden role="dialog" aria-modal="true" aria-labelledby="metric-modal-title">
  <div class="modal-backdrop" data-modal-close></div>
  <section class="modal-panel">
    <div class="modal-head">
      <h2 id="metric-modal-title" class="modal-title"></h2>
      <button type="button" class="modal-close" data-modal-close aria-label="Close metric explanation">&times;</button>
    </div>
    <div class="info-block">
      <div class="info-label">Category</div>
      <div id="metric-modal-group" class="info-value"></div>
    </div>
    <div class="info-block">
      <div class="info-label">Meaning</div>
      <div id="metric-modal-meaning" class="info-value"></div>
    </div>
    <div class="info-block">
      <div class="info-label">Direction</div>
      <div id="metric-modal-direction" class="info-value"></div>
    </div>
    <div class="info-block">
      <div class="info-label">How to compare</div>
      <div id="metric-modal-hint" class="info-value"></div>
    </div>
  </section>
</div>
<script>
  const metricInfo = {metric_info_json};
  const metricModal = document.getElementById("metric-modal");
  const modalTitle = document.getElementById("metric-modal-title");
  const modalGroup = document.getElementById("metric-modal-group");
  const modalMeaning = document.getElementById("metric-modal-meaning");
  const modalDirection = document.getElementById("metric-modal-direction");
  const modalHint = document.getElementById("metric-modal-hint");
  let lastMetricTrigger = null;

  function showMetricInfo(metric, trigger) {{
    const info = metricInfo[metric] || {{
      metric,
      group: "unknown",
      meaning: "No explanation is available for this metric yet.",
      direction: "reference only",
      hint: "Use the raw metric name and source evaluation code for interpretation."
    }};
    lastMetricTrigger = trigger || null;
    modalTitle.textContent = info.metric;
    modalGroup.textContent = info.group;
    modalMeaning.textContent = info.meaning;
    modalDirection.textContent = info.direction;
    modalHint.textContent = info.hint;
    metricModal.hidden = false;
    const closeButton = metricModal.querySelector(".modal-close");
    if (closeButton) {{
      closeButton.focus();
    }}
  }}

  function closeMetricInfo() {{
    metricModal.hidden = true;
    if (lastMetricTrigger) {{
      lastMetricTrigger.focus();
    }}
  }}

  document.addEventListener("click", (event) => {{
    const metricButton = event.target.closest("[data-metric]");
    if (metricButton) {{
      showMetricInfo(metricButton.dataset.metric, metricButton);
      return;
    }}
    if (event.target.closest("[data-modal-close]")) {{
      closeMetricInfo();
    }}
  }});

  document.addEventListener("keydown", (event) => {{
    if (event.key === "Escape" && !metricModal.hidden) {{
      closeMetricInfo();
    }}
  }});
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "model_comparison_report.html"
    csv_output = args.csv_output.resolve() if args.csv_output else root / "model_comparison_summary.csv"

    records = discover_records(root, args.all_evals, args.latest_per_model)
    if not records:
        raise SystemExit(f"No coordination_eval JSON/CSV files found under {root}")

    all_metrics = numeric_metrics(records)
    chart_metrics = choose_chart_metrics(records, args.metrics)
    if args.max_charts is not None:
        chart_metrics = chart_metrics[: max(0, args.max_charts)]

    write_summary_csv(csv_output, records, all_metrics)
    write_direction_csv(csv_output, all_metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_html(root, csv_output, records, chart_metrics, all_metrics),
        encoding="utf-8",
    )

    print(f"Loaded {len(records)} evaluations from {root}")
    print(f"Wrote HTML report: {output}")
    print(f"Wrote CSV summary: {csv_output}")
    print(f"Wrote metric directions: {csv_output.with_name(csv_output.stem + '_metric_directions.csv')}")


if __name__ == "__main__":
    main()



# python compare_coordination_evals.py
