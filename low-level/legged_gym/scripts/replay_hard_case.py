# SPDX-License-Identifier: BSD-3-Clause
"""Replay one mined hard case from its pre-trigger simulator snapshot."""

from __future__ import annotations

import json
import os
import random
import time

import isaacgym  # noqa: F401 - must precede torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - task registration
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import task_registry


def _resolve_snapshot(args):
    if args.snapshot:
        snapshot_path = os.path.abspath(args.snapshot)
    else:
        root = args.hard_case_root or args.eval_out_dir
        if root is None or args.case_id is None:
            raise ValueError(
                "Provide --snapshot, or provide both --hard_case_root and --case_id"
            )
        snapshot_path = os.path.join(
            os.path.abspath(root), "cases", f"case_{int(args.case_id):06d}",
            "snapshot.pt",
        )
    if not os.path.isfile(snapshot_path):
        raise FileNotFoundError(snapshot_path)
    return snapshot_path


def _restore_rng_state(rng_state):
    if not rng_state:
        return
    if "torch_cpu" in rng_state:
        torch.set_rng_state(rng_state["torch_cpu"])
    if "torch_cuda_all" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda_all"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "python" in rng_state:
        random.setstate(rng_state["python"])


def _policy_action(policy, obs, env, use_jit):
    if use_jit:
        return policy(torch.cat((
            obs[:, :env.cfg.env.num_proprio],
            obs[:, env.cfg.env.num_proprio + env.cfg.env.num_priv:],
        ), dim=1))
    return policy(obs.detach(), hist_encoding=True)


def _load_policy(env, train_cfg, args, log_pth):
    train_cfg.runner.resume = True
    runner, _, checkpoint, log_pth = task_registry.make_alg_runner(
        log_root=log_pth, env=env, name=args.task, args=args,
        train_cfg=train_cfg, return_log_dir=True,
    )
    policy = runner.get_inference_policy(device=env.device, stochastic=args.stochastic)
    if args.use_jit:
        jit_path = os.path.join(
            log_pth, "traced",
            args.exptid + "_" + str(args.checkpoint) + "_jit.pt",
        )
        policy = torch.jit.load(jit_path, map_location=runner.device)
    return policy, checkpoint


def _validate_architecture(env, snapshot):
    state = snapshot.get("state", {})
    saved_obs = state.get("obs_buf")
    if saved_obs is None:
        raise ValueError("Snapshot has no obs_buf; it predates replay snapshot support")
    saved_obs_dim = int(saved_obs.shape[-1])
    current_obs_dim = int(env.obs_buf.shape[-1])
    if saved_obs_dim != current_obs_dim:
        raise ValueError(
            "Checkpoint/config observation architecture is incompatible with the case: "
            f"snapshot num_observations={saved_obs_dim}, current={current_obs_dim}. "
            "Observations will not be padded or truncated."
        )
    saved_commands = state.get("commands")
    if saved_commands is not None and int(saved_commands.shape[-1]) != int(env.commands.shape[-1]):
        raise ValueError(
            "Base-command architecture is incompatible with the snapshot: "
            f"saved={saved_commands.shape[-1]}, current={env.commands.shape[-1]}"
        )
    saved_message = state.get("m_a2b")
    if saved_message is not None and int(saved_message.shape[-1]) != 5:
        raise ValueError(f"Unexpected saved m_a2b dimension: {saved_message.shape[-1]}")
    source_config = snapshot.get("evaluation_config", {})
    expected_gait = source_config.get("observe_gait_commands")
    if expected_gait is not None and bool(expected_gait) != bool(env.cfg.env.observe_gait_commands):
        raise ValueError(
            "Snapshot/checkpoint gait-command architecture mismatch. "
            f"Snapshot observe_gait_commands={expected_gait}, current="
            f"{env.cfg.env.observe_gait_commands}. Pass --observe_gait_commands only "
            "for checkpoints that were trained with it."
        )
    expected_5d = source_config.get("use_5d_base_command")
    if expected_5d is not None and bool(expected_5d) != bool(env.use_5d_base_command):
        raise ValueError(
            "Snapshot/checkpoint base-command architecture mismatch: "
            f"snapshot use_5d_base_command={expected_5d}, current={env.use_5d_base_command}."
        )
    expected_message = source_config.get("use_arm_base_message")
    if expected_message is not None and bool(expected_message) != bool(env.use_arm_base_message):
        raise ValueError(
            "Snapshot/checkpoint arm-to-base message architecture mismatch: "
            f"snapshot use_arm_base_message={expected_message}, "
            f"current={env.use_arm_base_message}."
        )


def _replay_steps(args, snapshot_path, snapshot, env):
    if args.replay_steps is not None:
        if args.replay_steps <= 0:
            raise ValueError("--replay_steps must be positive")
        return int(args.replay_steps)
    metadata_path = os.path.join(os.path.dirname(snapshot_path), "metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        post_steps = int(round(
            float(metadata.get("post_trigger_seconds_requested", 3.0)) / env.dt
        ))
        trigger_step = int(metadata.get("trigger_step", snapshot["trigger_sim_step"]))
        snapshot_step = int(snapshot.get("snapshot_sim_step", trigger_step))
        return max(1, trigger_step - snapshot_step + post_steps)
    return max(1, int(round(4.0 / env.dt)))


@torch.no_grad()
def replay(args):
    snapshot_path = _resolve_snapshot(args)
    snapshot = torch.load(snapshot_path, map_location="cpu")
    case_dir = os.path.dirname(snapshot_path)
    metadata_path = os.path.join(case_dir, "metadata.json")
    metadata = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    case_id = int(metadata.get("case_id", args.case_id or 0))

    args.num_envs = 1
    log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.domain_rand.push_robots = False
    source_config = snapshot.get("evaluation_config", {})
    if "seed" in source_config and args.seed is None:
        env_cfg.seed = int(source_config["seed"])
    for name, value in source_config.get("domain_rand", {}).items():
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, bool(value))
    for name, value in source_config.get("init_state", {}).items():
        if hasattr(env_cfg.init_state, name):
            setattr(env_cfg.init_state, name, float(value))
    if "terrain_height" in source_config:
        env_cfg.terrain.height = list(source_config["terrain_height"])
    if args.record_video:
        env_cfg.env.record_video = True
        env_cfg.env.record_video_env_ids = [0]

    env_start = time.perf_counter()
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    environment_creation_sec = time.perf_counter() - env_start
    _validate_architecture(env, snapshot)

    policy_start = time.perf_counter()
    policy, checkpoint = _load_policy(env, train_cfg, args, log_pth)
    policy_loading_sec = time.perf_counter() - policy_start
    reset_result = env.reset()
    del reset_result
    env.enable_evaluation_mode(command_override=True, ee_override=True, diagnostics=True)
    obs = env.restore_evaluation_snapshot(snapshot, env_id=0)
    _restore_rng_state(snapshot.get("rng_state_at_trigger"))

    steps = _replay_steps(args, snapshot_path, snapshot, env)
    replay_dir = os.path.join(case_dir, "replay", args.replay_label)
    os.makedirs(replay_dir, exist_ok=True)
    writer = None
    video_path = None
    if args.record_video:
        import imageio
        video_path = os.path.join(replay_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=max(1, int(round(0.5 / env.dt))))

    trajectory = {}
    replay_start = time.perf_counter()
    video_save_time = 0.0
    try:
        from tqdm import tqdm
        progress = tqdm(
            range(steps), total=steps,
            desc=f"Replaying hard case {case_id}", unit="step", dynamic_ncols=True,
        )
    except ImportError:
        progress = range(steps)

    completed_steps = 0
    for step in progress:
        actions = _policy_action(policy, obs, env, args.use_jit)
        obs, _, _, _, _, _ = env.step(actions.detach())
        state = env.evaluation_step_state
        for name, value in state.items():
            trajectory.setdefault(name, []).append(value[0].detach().cpu().numpy())
        trajectory.setdefault("replay_step", []).append(np.asarray(step))
        completed_steps += 1
        if writer is not None:
            io_start = time.perf_counter()
            images = env.render_record(mode="rgb_array")
            if images is not None and 0 in images:
                writer.append_data(images[0])
            video_save_time += time.perf_counter() - io_start
        if bool(state["done"][0].item()):
            break
    if writer is not None:
        close_start = time.perf_counter()
        writer.close()
        video_save_time += time.perf_counter() - close_start

    replay_wall_time = time.perf_counter() - replay_start
    arrays = {name: np.stack(values) for name, values in trajectory.items()}
    trajectory_path = os.path.join(replay_dir, "trajectory.npz")
    trajectory_save_start = time.perf_counter()
    np.savez_compressed(trajectory_path, **arrays)
    trajectory_save_time = time.perf_counter() - trajectory_save_start
    runtime = {
        "case_id": case_id,
        "checkpoint": checkpoint,
        "replay_label": args.replay_label,
        "planned_steps": steps,
        "completed_steps": completed_steps,
        "environment_creation_sec": environment_creation_sec,
        "policy_loading_sec": policy_loading_sec,
        "stage_b_replay_wall_time_sec": replay_wall_time,
        "trajectory_save_time_sec": trajectory_save_time,
        "video_save_time_sec": video_save_time,
        "snapshot": snapshot_path,
        "trajectory": trajectory_path,
        "video": video_path,
        "determinism_note": snapshot.get("determinism_note", "not provided"),
    }
    runtime_path = os.path.join(replay_dir, "runtime_stats.json")
    with open(runtime_path, "w", encoding="utf-8") as file:
        json.dump(runtime, file, indent=2, sort_keys=True)
    print(f"Replay trajectory: {trajectory_path}")
    if video_path:
        print(f"Replay video: {video_path}")
    print(f"Replay runtime stats: {runtime_path}")
    return runtime


if __name__ == "__main__":
    replay(get_args())
