# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint or run systematic parallel coverage/hard-case evaluation."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - task registration side effects
from legged_gym.utils.coordination_metrics import CoordinationMetrics
from legged_gym.utils.eval_coverage import (
    CoverageMetrics,
    EvaluationCoverageScheduler,
    resolve_play_num_envs_value,
)
from legged_gym.utils.hard_case_mining import HardCaseMiner
from legged_gym.utils.helpers import export_policy_as_jit, get_args, get_load_path
from legged_gym.utils.logger import Logger
from legged_gym.utils.task_registry import task_registry


np.set_printoptions(precision=3, suppress=True)


def _format_eval_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return value


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _print_coordination_summary(summary):
    keys = [
        "meta/num_envs", "meta/eval_steps", "vel/vx_mae", "vel/yaw_mae",
        "ee/pos_mean", "ee/ori_geodesic_mean", "ee/success_rate_step",
        "stability/survival_rate_step", "stability/base_ang_acc_rms",
        "energy/total_power_abs_mean", "smoothness/arm_action_rate_mean",
        "coordination/base_ang_acc_when_arm_large", "workspace/solvability_rate",
        "workspace/hull_volume",
    ]
    print("========== Coordination Evaluation ==========")
    for key in keys:
        print(f"{key}: {_format_eval_value(summary.get(key, float('nan')))}")
    warnings = list(summary.get("metrics/warnings", []))
    warnings.extend(summary.get("metrics/skipped", []))
    print(f"warnings: {'; '.join(warnings) if warnings else 'none'}")
    print("=============================================")


def resolve_play_num_envs(env_cfg: Any, args: Any) -> int:
    """Resolve play env count without ever overriding an explicit CLI value."""
    resolved = resolve_play_num_envs_value(
        env_cfg.env.num_envs,
        args.num_envs,
        coverage_default=(256 if getattr(args, "eval_coverage", False) else None),
    )
    env_cfg.env.num_envs = resolved
    return int(env_cfg.env.num_envs)


def _apply_deterministic_eval_cfg(env_cfg: Any) -> Dict[str, Any]:
    """Control known config-time randomization for command-space attribution."""
    changes: Dict[str, Any] = {}
    for name in (
        "push_robots", "randomize_friction", "randomize_base_mass",
        "randomize_base_com", "randomize_motor", "randomize_gripper_mass",
    ):
        if hasattr(env_cfg.domain_rand, name):
            changes[f"domain_rand.{name}"] = False
            setattr(env_cfg.domain_rand, name, False)
    for name in ("origin_perturb_range", "init_vel_perturb_range", "rand_yaw_range"):
        if hasattr(env_cfg.init_state, name):
            changes[f"init_state.{name}"] = 0.0
            setattr(env_cfg.init_state, name, 0.0)
    if hasattr(env_cfg, "noise") and hasattr(env_cfg.noise, "add_noise"):
        changes["noise.add_noise"] = False
        env_cfg.noise.add_noise = False
    if hasattr(env_cfg.terrain, "height"):
        changes["terrain.height"] = [0.0, 0.0]
        env_cfg.terrain.height = [0.0, 0.0]
    return changes


def _prepare_play_cfg(env_cfg: Any, args: Any) -> Tuple[int, Dict[str, Any]]:
    expected_num_envs = resolve_play_num_envs(env_cfg, args)
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 3
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_com = False
    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    deterministic_changes = (
        _apply_deterministic_eval_cfg(env_cfg) if args.eval_deterministic else {}
    )
    if args.record_video:
        env_cfg.env.record_video_env_ids = [0]
    return expected_num_envs, deterministic_changes


def _load_policy(env, train_cfg, args, log_pth):
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, checkpoint, log_pth = task_registry.make_alg_runner(
        log_root=log_pth, env=env, name=args.task, args=args,
        train_cfg=train_cfg, return_log_dir=True,
    )
    policy = ppo_runner.get_inference_policy(device=env.device, stochastic=args.stochastic)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
            "exported", "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
    if SAVE_ACTOR_HIST_ENCODER:
        log_root = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
        )
        model_file = get_load_path(
            log_root, load_run=args.load_run, checkpoint=args.checkpoint
        )
        model_name = model_file.split("/")[-1].split(".")[0]
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
            train_cfg.runner.load_run, "exported",
        )
        os.makedirs(path, exist_ok=True)
        torch.save(
            ppo_runner.alg.actor_critic.actor.state_dict(),
            os.path.join(path, model_name + "_actor.pt"),
        )
    if args.use_jit:
        path = os.path.join(
            log_pth, "traced",
            args.exptid + "_" + str(args.checkpoint) + "_jit.pt",
        )
        policy = torch.jit.load(path, map_location=ppo_runner.device)
    return policy, checkpoint, log_pth


def _policy_action(policy, obs, env, use_jit):
    if use_jit:
        return policy(torch.cat((
            obs[:, :env.cfg.env.num_proprio],
            obs[:, env.cfg.env.num_proprio + env.cfg.env.num_priv:],
        ), dim=1))
    return policy(obs.detach(), hist_encoding=True)


def _coordination_metrics(env, env_cfg, args, warmup_steps=None):
    if not args.eval_coordination:
        return None
    return CoordinationMetrics(
        env=env, cfg=env_cfg, num_envs=env.num_envs, device=env.device,
        dt=env.dt,
        warmup_steps=(args.eval_warmup_steps if warmup_steps is None else int(warmup_steps)),
        success_pos_thr=args.ee_success_pos_thr,
        success_ori_thr=args.ee_success_ori_thr,
        fall_height_thr=args.fall_height_thr,
        save_eval_traj=args.save_eval_traj,
    )


def _default_coverage_out_dir(log_pth, args, checkpoint):
    if args.eval_out_dir:
        base_dir = os.path.abspath(args.eval_out_dir)
    else:
        base_dir = os.path.join(
            log_pth, "hard_case_eval",
            f"{args.eval_coverage_mode}_ckpt{checkpoint}",
        )
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    return os.path.join(base_dir, run_id)


class _Progress:
    def __init__(self, total: int):
        self.total = int(total)
        self.completed = 0
        self.start = time.perf_counter()
        self._tqdm = None
        try:
            from tqdm import tqdm
            self._tqdm = tqdm(
                total=self.total, desc="Hard-case evaluation",
                unit="sample", dynamic_ncols=True,
            )
        except ImportError:
            print("tqdm is unavailable; using batch-level progress output.")

    def update(self, count: int, **postfix):
        count = int(count)
        self.completed += count
        elapsed = time.perf_counter() - self.start
        rate = self.completed / max(elapsed, 1.0e-9)
        eta = (self.total - self.completed) / max(rate, 1.0e-9)
        postfix = dict(postfix)
        postfix.update(
            elapsed=_format_duration(elapsed), ETA=_format_duration(eta),
            **{"samples/s": f"{rate:.2f}"},
        )
        if self._tqdm is not None:
            self._tqdm.update(count)
            self._tqdm.set_postfix(postfix, refresh=True)
        elif count:
            percentage = 100.0 * self.completed / max(self.total, 1)
            print(
                f"Coverage samples: {self.completed}/{self.total} ({percentage:.1f}%) "
                f"elapsed={postfix['elapsed']} ETA={postfix['ETA']} "
                f"samples/s={postfix['samples/s']}"
            )

    def write(self, message: str):
        self._tqdm.write(message) if self._tqdm is not None else print(message)

    def close(self):
        if self._tqdm is not None:
            self._tqdm.close()


def _build_scheduler(env, env_cfg, args, planned_samples):
    ee_nominal = list(env_cfg.goal_ee.ranges.init_pos_end) + [0.0, 0.0, 0.0]
    return EvaluationCoverageScheduler(
        command_ranges=env.command_ranges,
        goal_ee_ranges=env.goal_ee_ranges,
        use_5d_base_command=env.use_5d_base_command,
        num_samples=planned_samples,
        mode=args.eval_coverage_mode,
        seed=int(env_cfg.seed),
        ee_position_bins=args.eval_ee_position_bins,
        base_height_nominal=env_cfg.rewards.base_height_target,
        ee_nominal=ee_nominal,
    )


def _rejection_reason(code: int) -> str:
    return {
        0: "", 1: "outside_configured_range",
        2: "collision_box", 3: "underground",
    }.get(int(code), f"unknown_{int(code)}")


@torch.no_grad()
def _run_coverage_evaluation(
    *, env, env_cfg, policy, args, checkpoint, log_pth, init_timings,
):
    if args.record_video:
        raise ValueError(
            "Stage-A --eval_coverage is numerical-only; omit --record_video and use "
            "replay_hard_case.py for selected candidates."
        )
    if args.hard_case_mining and args.eval_coverage_mode != "joint":
        print(
            "[HardCaseMining] warning: joint mode is recommended; continuing with "
            f"mode={args.eval_coverage_mode}."
        )
    requested_samples = int(args.eval_num_samples)
    planned_samples = requested_samples
    if args.eval_probe_samples is not None:
        if args.eval_probe_samples <= 0:
            raise ValueError("--eval_probe_samples must be positive")
        planned_samples = min(requested_samples, int(args.eval_probe_samples))

    eval_init_start = time.perf_counter()
    scheduler = _build_scheduler(env, env_cfg, args, planned_samples)
    coverage = CoverageMetrics(scheduler, num_bins=args.eval_coverage_bins)
    out_dir = _default_coverage_out_dir(log_pth, args, checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Evaluation output directory: {out_dir}")

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    env.enable_evaluation_mode(command_override=True, ee_override=True, diagnostics=True)
    assert env.num_envs == int(env_cfg.env.num_envs), (env.num_envs, env_cfg.env.num_envs)
    print(f"Evaluation num_envs: {env.num_envs}")
    coordination = _coordination_metrics(
        env, env_cfg, args, warmup_steps=0
    )
    if coordination is not None:
        coordination.excluded_sample_warmup_steps = int(args.eval_sample_warmup_steps)
    miner = None
    if args.hard_case_mining:
        snapshot_metadata = {
            "seed": int(env_cfg.seed),
            "eval_deterministic": bool(args.eval_deterministic),
            "observe_gait_commands": bool(env_cfg.env.observe_gait_commands),
            "use_5d_base_command": bool(env.use_5d_base_command),
            "use_arm_base_message": bool(env.use_arm_base_message),
            "pfg_enabled": bool(env.use_pfg_reward),
            "manipulability_enabled": bool(
                getattr(env_cfg.rewards.manipulability, "enabled", False)
            ),
            "num_observations": int(env.cfg.env.num_observations),
            "terrain_height": list(env_cfg.terrain.height),
            "domain_rand": {
                name: bool(getattr(env_cfg.domain_rand, name))
                for name in (
                    "push_robots", "randomize_friction", "randomize_base_mass",
                    "randomize_base_com", "randomize_motor", "randomize_gripper_mass",
                )
                if hasattr(env_cfg.domain_rand, name)
            },
            "init_state": {
                name: float(getattr(env_cfg.init_state, name))
                for name in (
                    "origin_perturb_range", "init_vel_perturb_range", "rand_yaw_range",
                )
                if hasattr(env_cfg.init_state, name)
            },
        }
        miner = HardCaseMiner(
            num_envs=env.num_envs, device=env.device, dt=env.dt, out_dir=out_dir,
            pre_seconds=args.hard_case_pre_seconds,
            post_seconds=args.hard_case_post_seconds,
            min_steps=args.hard_case_min_steps,
            max_cases=args.hard_case_max_cases,
            max_video_candidates=args.hard_case_max_video_candidates,
            eta_raw_threshold=args.hard_case_eta_raw_thr,
            eta_ratio_threshold=args.hard_case_eta_ratio_thr,
            ee_request_threshold=args.hard_case_ee_request_thr,
            ee_error_threshold=args.hard_case_ee_error_thr,
            snapshot_interval_seconds=args.hard_case_snapshot_interval,
            snapshot_metadata=snapshot_metadata,
        )

    eval_initialization_sec = time.perf_counter() - eval_init_start
    init_timings["evaluation_initialization_sec"] = eval_initialization_sec
    print("[Timing]")
    print(f"Environment creation: {init_timings['environment_creation_sec']:.3f} s")
    print(f"Policy loading:        {init_timings['policy_loading_sec']:.3f} s")
    print(f"Eval initialization:  {eval_initialization_sec:.3f} s")

    total_batches = math.ceil(planned_samples / env.num_envs)
    sample_total_steps = int(args.eval_sample_warmup_steps) + int(args.eval_sample_steps)
    if sample_total_steps <= 0 or args.eval_sample_steps <= 0:
        raise ValueError("Evaluation sample step counts must be positive")
    progress = _Progress(planned_samples)
    evaluation_start = time.perf_counter()
    batch_stats = []
    global_eval_step = 0
    episode_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    for batch_index in range(total_batches):
        batch = scheduler.next_batch(env.num_envs)
        if batch is None:
            break
        batch_start = time.perf_counter()
        batch_size = len(batch)
        assigned_env_ids = all_env_ids[:batch_size]
        env.reset_idx(all_env_ids, start=False)
        if miner is not None:
            miner.reset_envs(all_env_ids)
        episode_ids[assigned_env_ids] += 1

        base_commands = batch.base_commands.to(env.device)
        ee_commands = batch.ee_commands.to(env.device)
        env.set_evaluation_commands(base_commands, assigned_env_ids)
        valid_local, rejection_codes = env.set_evaluation_ee_goals(
            ee_commands, assigned_env_ids,
            transition_steps=args.eval_sample_warmup_steps,
            total_steps=sample_total_steps,
        )
        rejected_local = (~valid_local).nonzero(as_tuple=False).flatten()
        if rejected_local.numel() > 0:
            rejected_sample_ids = batch.sample_ids[rejected_local.cpu()].tolist()
            reasons = [
                _rejection_reason(rejection_codes[index].item())
                for index in rejected_local
            ]
            coverage.mark_rejected(rejected_sample_ids, reasons)
            progress.update(
                len(rejected_sample_ids), batch=f"{batch_index + 1}/{total_batches}",
                active_envs=int(valid_local.sum().item()),
                hard_cases=(miner.detected_cases if miner else 0),
                saved_cases=(miner.saved_cases if miner else 0),
            )

        active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        active[assigned_env_ids] = valid_local
        initial_valid = active.clone()
        sample_ids_by_env = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
        sample_ids_by_env[assigned_env_ids] = batch.sample_ids.to(env.device)
        scheduled_base = torch.zeros_like(env.commands)
        scheduled_base[assigned_env_ids] = base_commands
        scheduled_ee = torch.zeros(env.num_envs, 6, device=env.device, dtype=ee_commands.dtype)
        scheduled_ee[assigned_env_ids] = ee_commands

        env._update_curr_ee_goal()
        env.compute_observations()
        obs = env.get_observations()
        if miner is not None:
            miner.capture_snapshot(
                env.get_evaluation_snapshot_state(), global_eval_step, active, force=True
            )

        sum_sq_error = torch.zeros(env.num_envs, device=env.device)
        measured_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        success_any = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        fall_any = torch.zeros_like(success_any)
        collision_any = torch.zeros_like(success_any)
        timeout_any = torch.zeros_like(success_any)
        min_eta_raw = torch.full_like(sum_sq_error, float("inf"))
        min_eta_ratio = torch.full_like(sum_sq_error, float("inf"))
        low_manip_steps = torch.zeros_like(measured_steps)
        alignment_sum = torch.zeros_like(sum_sq_error)
        achieved_ee_sphere = torch.full(
            (env.num_envs, 3), float("nan"), device=env.device
        )
        completed_this_batch = torch.zeros_like(active)
        steps_run = 0
        if coordination is not None:
            coordination.begin_sample_batch(initial_valid)

        for sample_step in range(sample_total_steps):
            active_before = active.clone()
            actions = _policy_action(policy, obs, env, args.use_jit)
            obs, _, _, _, _, _ = env.step(actions.detach())
            state = env.evaluation_step_state
            if state is None:
                raise RuntimeError("Environment did not publish evaluation_step_state")
            done = state["done"].bool() & active_before
            step_success = (
                (state["ee_pos_error"] <= args.ee_success_pos_thr)
                & (state["ee_orientation_error"] <= args.ee_success_ori_thr)
                & ~state["fall"].bool()
            )
            state_features = {
                "sample_id": sample_ids_by_env,
                "batch_id": torch.full_like(sample_ids_by_env, batch_index),
                "env_id": all_env_ids,
                "episode_id": episode_ids,
                "sim_step": torch.full_like(sample_ids_by_env, global_eval_step),
                "sample_step": torch.full_like(sample_ids_by_env, sample_step),
                "timestamp": torch.full(
                    (env.num_envs,), global_eval_step * env.dt, device=env.device
                ),
                **state,
                "success": step_success,
            }
            if miner is not None:
                miner.update(
                    features=state_features,
                    snapshot_state=env.get_evaluation_snapshot_state(),
                    active_mask=active_before,
                    sim_step=global_eval_step,
                    sample_ids=sample_ids_by_env,
                    batch_id=batch_index,
                    episode_ids=episode_ids,
                    scheduled_base_commands=scheduled_base,
                    scheduled_ee_commands=scheduled_ee,
                )
            if sample_step >= args.eval_sample_warmup_steps:
                measure_mask = active_before
                if coordination is not None:
                    coordination.update(
                        env,
                        actions.detach(),
                        obs,
                        active_mask=measure_mask,
                        fall_mask=state["fall"],
                        collision_mask=state["collision"],
                    )
                sum_sq_error[measure_mask] += torch.square(state["ee_pos_error"][measure_mask])
                measured_steps[measure_mask] += 1
                success_any |= step_success & measure_mask
                min_eta_raw[measure_mask] = torch.minimum(
                    min_eta_raw[measure_mask], state["eta_raw"][measure_mask]
                )
                min_eta_ratio[measure_mask] = torch.minimum(
                    min_eta_ratio[measure_mask], state["eta_ratio"][measure_mask]
                )
                low = state["eta_ratio"] < args.hard_case_eta_ratio_thr
                if args.hard_case_eta_raw_thr > 0.0:
                    low |= state["eta_raw"] < args.hard_case_eta_raw_thr
                low_manip_steps += (low & measure_mask).long()
                alignment_sum[measure_mask] += state["base_assist_alignment_raw"][measure_mask]
            achieved_ee_sphere[active_before] = state["achieved_ee_sphere"][active_before]
            fall_any |= state["fall"].bool() & active_before
            collision_any |= state["collision"].bool() & active_before
            timeout_any |= state["timeout"].bool() & active_before

            newly_completed = done & ~completed_this_batch
            if newly_completed.any():
                count = int(newly_completed.sum().item())
                completed_this_batch |= newly_completed
                active &= ~newly_completed
                if miner is not None:
                    miner.finish_envs(newly_completed.nonzero(as_tuple=False).flatten())
                progress.update(
                    count, batch=f"{batch_index + 1}/{total_batches}",
                    active_envs=int(active.sum().item()),
                    hard_cases=(miner.detected_cases if miner else 0),
                    saved_cases=(miner.saved_cases if miner else 0),
                )
            global_eval_step += 1
            steps_run += 1
            if not active.any():
                break

        if coordination is not None:
            coordination.end_sample_batch(
                initial_valid,
                success_mask=success_any,
                fall_mask=fall_any,
                collision_mask=collision_any,
            )

        # A trigger near the end of the fixed sample window still receives its
        # full post-trigger numerical window.  Only envs with an open candidate
        # continue; these extra steps are excluded from coverage/task metrics.
        if miner is not None:
            for extra_step in range(miner.post_steps):
                recording = miner.env_to_case >= 0
                if not recording.any():
                    break
                actions = _policy_action(policy, obs, env, args.use_jit)
                obs, _, _, _, _, _ = env.step(actions.detach())
                state = env.evaluation_step_state
                step_success = (
                    (state["ee_pos_error"] <= args.ee_success_pos_thr)
                    & (state["ee_orientation_error"] <= args.ee_success_ori_thr)
                    & ~state["fall"].bool()
                )
                state_features = {
                    "sample_id": sample_ids_by_env,
                    "batch_id": torch.full_like(sample_ids_by_env, batch_index),
                    "env_id": all_env_ids,
                    "episode_id": episode_ids,
                    "sim_step": torch.full_like(sample_ids_by_env, global_eval_step),
                    "sample_step": torch.full_like(
                        sample_ids_by_env, sample_total_steps + extra_step
                    ),
                    "timestamp": torch.full(
                        (env.num_envs,), global_eval_step * env.dt, device=env.device
                    ),
                    **state,
                    "success": step_success,
                }
                miner.update(
                    features=state_features,
                    snapshot_state=env.get_evaluation_snapshot_state(),
                    active_mask=recording,
                    sim_step=global_eval_step,
                    sample_ids=sample_ids_by_env,
                    batch_id=batch_index,
                    episode_ids=episode_ids,
                    scheduled_base_commands=scheduled_base,
                    scheduled_ee_commands=scheduled_ee,
                )
                done_recording = state["done"].bool() & recording
                if done_recording.any():
                    miner.finish_envs(done_recording.nonzero(as_tuple=False).flatten())
                global_eval_step += 1

        remaining_completed = initial_valid & ~completed_this_batch
        if remaining_completed.any():
            count = int(remaining_completed.sum().item())
            completed_this_batch |= remaining_completed
            if miner is not None:
                miner.finish_envs(remaining_completed.nonzero(as_tuple=False).flatten())
            progress.update(
                count, batch=f"{batch_index + 1}/{total_batches}", active_envs=0,
                hard_cases=(miner.detected_cases if miner else 0),
                saved_cases=(miner.saved_cases if miner else 0),
            )

        valid_env_ids = initial_valid.nonzero(as_tuple=False).flatten()
        batch_cpu = {
            "sample_id": sample_ids_by_env[valid_env_ids].cpu().tolist(),
            "steps": measured_steps[valid_env_ids].cpu().tolist(),
            "success": success_any[valid_env_ids].cpu().tolist(),
            "fall": fall_any[valid_env_ids].cpu().tolist(),
            "collision": collision_any[valid_env_ids].cpu().tolist(),
            "timeout": timeout_any[valid_env_ids].cpu().tolist(),
            "sum_sq_error": sum_sq_error[valid_env_ids].cpu().tolist(),
            "min_eta_raw": min_eta_raw[valid_env_ids].cpu().tolist(),
            "min_eta_ratio": min_eta_ratio[valid_env_ids].cpu().tolist(),
            "low_steps": low_manip_steps[valid_env_ids].cpu().tolist(),
            "alignment_sum": alignment_sum[valid_env_ids].cpu().tolist(),
            "achieved": achieved_ee_sphere[valid_env_ids].cpu().tolist(),
        }
        for index, sample_id in enumerate(batch_cpu["sample_id"]):
            measured = max(1, int(batch_cpu["steps"][index]))
            achieved = batch_cpu["achieved"][index]
            coverage.mark_executed(
                sample_id, batch_id=batch_index,
                env_id=int(valid_env_ids[index].item()),
                episode_id=int(episode_ids[valid_env_ids[index]].item()),
                evaluation_steps=int(batch_cpu["steps"][index]),
                success=bool(batch_cpu["success"][index]),
                fall=bool(batch_cpu["fall"][index]),
                collision=bool(batch_cpu["collision"][index]),
                timeout=bool(batch_cpu["timeout"][index]),
                termination_reason=(
                    "fall" if batch_cpu["fall"][index]
                    else "timeout" if batch_cpu["timeout"][index]
                    else "completed_window"
                ),
                ee_rmse=math.sqrt(float(batch_cpu["sum_sq_error"][index]) / measured),
                minimum_eta_raw=float(batch_cpu["min_eta_raw"][index]),
                minimum_eta_ratio=float(batch_cpu["min_eta_ratio"][index]),
                low_manip_duration_sec=float(batch_cpu["low_steps"][index]) * env.dt,
                mean_base_assist_alignment_raw=(
                    float(batch_cpu["alignment_sum"][index]) / measured
                ),
                **{
                    "achieved_ee/pos_l": achieved[0],
                    "achieved_ee/pos_p": achieved[1],
                    "achieved_ee/pos_y": achieved[2],
                },
            )

        batch_wall_time = time.perf_counter() - batch_start
        valid_count = int(initial_valid.sum().item())
        batch_stats.append({
            "batch_id": batch_index,
            "num_planned_samples": batch_size,
            "num_valid_samples": valid_count,
            "num_rejected_samples": int((~valid_local).sum().item()),
            "wall_time_sec": batch_wall_time,
            "samples_per_second": valid_count / max(batch_wall_time, 1.0e-9),
            "mean_episode_steps": (
                float(measured_steps[valid_env_ids].float().mean().item())
                if valid_count else 0.0
            ),
            "sim_steps": steps_run,
        })
        recent_batches = batch_stats[-5:]
        recent_valid = sum(item["num_valid_samples"] for item in recent_batches)
        recent_wall = sum(item["wall_time_sec"] for item in recent_batches)
        smoothed_rate = recent_valid / max(recent_wall, 1.0e-9)
        smoothed_eta = (
            max(0, planned_samples - progress.completed)
            / max(smoothed_rate, 1.0e-9)
        )
        progress.write(
            f"Batch {batch_index + 1}/{total_batches}: valid={valid_count}, "
            f"rejected={int((~valid_local).sum().item())}, wall={batch_wall_time:.2f}s, "
            f"throughput={valid_count / max(batch_wall_time, 1e-9):.2f} samples/s, "
            f"smoothed_ETA={_format_duration(smoothed_eta)}"
        )

    progress.close()
    evaluation_wall_time = time.perf_counter() - evaluation_start
    coverage_paths = coverage.save(out_dir)
    hard_case_paths = None
    if miner is not None:
        hard_case_paths = miner.save(sample_records=coverage.records)
        subset_path = hard_case_paths["subset_stats_json"]
        with open(subset_path, "r", encoding="utf-8") as file:
            subset_stats = json.load(file)
        executed_count = coverage.report()["executed_samples"]
        subset_stats.update(
            hard_case_rate_denominator=executed_count,
            hard_case_rate=(
                hard_case_paths["detected_cases"] / executed_count
                if executed_count else None
            ),
        )
        with open(subset_path, "w", encoding="utf-8") as file:
            json.dump(subset_stats, file, indent=2, sort_keys=True, allow_nan=True)

    coordination_paths = None
    if coordination is not None:
        coordination_summary = coordination.summarize()
        coordination_paths = coordination.save(os.path.join(out_dir, "coordination"))
        _print_coordination_summary(coordination_summary)

    coverage_report = coverage.report()
    processed = coverage_report["executed_samples"] + coverage_report["rejected_samples"]
    throughput = processed / max(evaluation_wall_time, 1.0e-9)
    recent_batches = batch_stats[-5:]
    smoothed_throughput = (
        sum(item["num_valid_samples"] for item in recent_batches)
        / max(sum(item["wall_time_sec"] for item in recent_batches), 1.0e-9)
        if recent_batches else 0.0
    )
    runtime_stats = {
        "num_envs": env.num_envs,
        "total_samples": planned_samples,
        "requested_full_samples": requested_samples,
        "processed_samples": processed,
        "executed_samples": coverage_report["executed_samples"],
        "rejected_samples": coverage_report["rejected_samples"],
        "total_batches": total_batches,
        **init_timings,
        "evaluation_wall_time_sec": evaluation_wall_time,
        "samples_per_second": throughput,
        "smoothed_samples_per_second_last_5_batches": smoothed_throughput,
        "hard_cases_detected": hard_case_paths["detected_cases"] if hard_case_paths else 0,
        "hard_cases_saved": hard_case_paths["saved_cases"] if hard_case_paths else 0,
        "snapshot_save_time_sec": (
            hard_case_paths["snapshot_save_time_sec"] if hard_case_paths else 0.0
        ),
        "trajectory_save_time_sec": (
            hard_case_paths["trajectory_save_time_sec"] if hard_case_paths else 0.0
        ),
        "video_save_time_sec": 0.0,
        "batches": batch_stats,
        "coordination_outputs": coordination_paths,
    }
    runtime_path = os.path.join(out_dir, "runtime_stats.json")
    with open(runtime_path, "w", encoding="utf-8") as file:
        json.dump(runtime_stats, file, indent=2, sort_keys=True, allow_nan=True)

    total_wall = (
        init_timings["environment_creation_sec"] + init_timings["policy_loading_sec"]
        + eval_initialization_sec + evaluation_wall_time
    )
    print("================ Evaluation Summary ================")
    print(f"Planned samples:        {planned_samples}")
    print(f"Completed samples:      {coverage_report['executed_samples']}")
    print(f"Rejected samples:       {coverage_report['rejected_samples']}")
    print(f"Parallel environments:  {env.num_envs}")
    print(f"Batches:                {total_batches}")
    print(f"Environment init:       {_format_duration(init_timings['environment_creation_sec'])}")
    print(f"Evaluation runtime:     {_format_duration(evaluation_wall_time)}")
    print(f"Total wall time:        {_format_duration(total_wall)}")
    print(f"Average throughput:     {throughput:.3f} samples/s")
    print(f"Hard cases detected:    {hard_case_paths['detected_cases'] if hard_case_paths else 0}")
    print(f"Hard cases saved:       {hard_case_paths['saved_cases'] if hard_case_paths else 0}")
    print(f"Snapshot IO:            {runtime_stats['snapshot_save_time_sec']:.3f} s")
    print(f"Trajectory IO:          {runtime_stats['trajectory_save_time_sec']:.3f} s")
    print("====================================================")
    print(f"Coverage report: {coverage_paths['json']}")
    print(f"Coverage samples: {coverage_paths['csv']}")
    print(f"Runtime stats: {runtime_path}")
    if hard_case_paths:
        print(f"Hard cases: {hard_case_paths['hard_cases_csv']}")
        print(f"Replay queue: {hard_case_paths['replay_candidates_json']}")
    if args.eval_probe_samples is not None and throughput > 0:
        print("[Probe estimate based on measured throughput; not an exact prediction]")
        for target in (4096, 8192, 16384):
            print(f"{target} samples ~= {target / throughput / 60.0:.1f} min")
    return runtime_stats


@torch.no_grad()
def _run_legacy_play(env, env_cfg, policy, args, checkpoint, log_pth, obs):
    Logger(env.dt)
    writers = {}
    if args.record_video:
        import imageio
        env.enable_viewer_sync = False
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "videos", args.exptid)
        os.makedirs(video_dir, exist_ok=True)
        for env_id in env._record_video_env_ids:
            writers[env_id] = imageio.get_writer(
                os.path.join(video_dir, f"{args.exptid}-{env_id}-{checkpoint}.mp4"),
                fps=25,
            )
    traj_length = (
        int(env.max_episode_length) if args.record_video
        else 1000 * int(env.max_episode_length)
    )
    if args.eval_coordination and args.eval_steps is not None:
        traj_length = int(args.eval_steps) + max(0, int(args.eval_warmup_steps))
    coordination = _coordination_metrics(env, env_cfg, args)
    for _ in range(traj_length):
        start_time = time.time()
        actions = _policy_action(policy, obs, env, args.use_jit)
        obs, _, _, _, _, _ = env.step(actions.detach())
        if coordination is not None:
            coordination.update(env, actions.detach(), obs)
        if args.record_video:
            images = env.render_record(mode="rgb_array")
            if images is not None:
                for env_id, image in images.items():
                    writers[env_id].append_data(image)
        time.sleep(max(0.02 - (time.time() - start_time), 0))
    for writer in writers.values():
        writer.close()
    if coordination is not None:
        summary = coordination.summarize()
        out_dir = args.eval_out_dir or os.path.join(log_pth, "coordination_eval")
        paths = coordination.save(out_dir)
        _print_coordination_summary(summary)
        print(f"Saved coordination evaluation: {paths}")


def play(args):
    log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    expected_num_envs, deterministic_changes = _prepare_play_cfg(env_cfg, args)
    if deterministic_changes:
        print("[Deterministic evaluation config]")
        for name, value in deterministic_changes.items():
            print(f"  {name} = {value}")

    environment_start = time.perf_counter()
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    environment_creation_sec = time.perf_counter() - environment_start
    assert env.num_envs == expected_num_envs, (
        f"Requested/configured {expected_num_envs} environments but Isaac Gym "
        f"created {env.num_envs}"
    )
    print(f"Evaluation num_envs: {env.num_envs}")
    obs = env.get_observations()

    policy_start = time.perf_counter()
    policy, checkpoint, log_pth = _load_policy(env, train_cfg, args, log_pth)
    init_timings = {
        "environment_creation_sec": environment_creation_sec,
        "policy_loading_sec": time.perf_counter() - policy_start,
    }
    if args.eval_coverage:
        return _run_coverage_evaluation(
            env=env, env_cfg=env_cfg, policy=policy, args=args,
            checkpoint=checkpoint, log_pth=log_pth, init_timings=init_timings,
        )
    if args.hard_case_mining:
        raise ValueError("--hard_case_mining requires --eval_coverage")
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    return _run_legacy_play(env, env_cfg, policy, args, checkpoint, log_pth, obs)


if __name__ == "__main__":
    EXPORT_POLICY = False
    SAVE_ACTOR_HIST_ENCODER = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    play(get_args())
