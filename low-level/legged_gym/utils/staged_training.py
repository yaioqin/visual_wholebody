"""Runtime support for iteration-based staged training."""

import math

import torch


_ORIENTATION_RANGE_NAMES = (
    "delta_orn_r",
    "delta_orn_p",
    "delta_orn_y",
)


def _read_stage(stage_cfg):
    stage = {"name": str(stage_cfg.name)}
    for field in _ORIENTATION_RANGE_NAMES:
        bounds = [float(value) for value in getattr(stage_cfg, field)]
        if (
            len(bounds) != 2
            or not all(math.isfinite(value) for value in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError(
                f"staged_training.{stage['name']}.{field} must be a finite "
                f"[min, max] range, got {bounds}"
            )
        stage[field] = bounds

    for field in ("base_height", "tracking_ee_world"):
        value = float(getattr(stage_cfg, field))
        if not math.isfinite(value):
            raise ValueError(
                f"staged_training.{stage['name']}.{field} must be finite, "
                f"got {value}"
            )
        stage[field] = value
    return stage


def apply_staged_training_parameters(env, stage_cfg, resample_orientation=False):
    """Apply a stage to both the config and the live environment caches."""
    stage = _read_stage(stage_cfg)

    for field in _ORIENTATION_RANGE_NAMES:
        bounds = list(stage[field])
        setattr(env.cfg.goal_ee.ranges, field, bounds)
        env.goal_ee_ranges[field] = bounds

    env.cfg.rewards.scales.base_height = stage["base_height"]
    env.cfg.rewards.arm_scales.tracking_ee_world = stage["tracking_ee_world"]

    if "base_height" not in env.reward_scales:
        raise RuntimeError(
            "Cannot change base_height at runtime because its reward function "
            "was not prepared"
        )
    if "tracking_ee_world" not in env.arm_reward_scales:
        raise RuntimeError(
            "Cannot change tracking_ee_world at runtime because its reward "
            "function was not prepared"
        )
    env.reward_scales["base_height"] = stage["base_height"]
    env.arm_reward_scales["tracking_ee_world"] = stage["tracking_ee_world"]
    env.active_staged_training_stage = stage["name"]

    # At the 37000 boundary, draw orientation commands from the enlarged range
    # immediately. Position trajectories and their timers remain uninterrupted.
    teleop_mode = bool(getattr(env.cfg.env, "teleop_mode", False))
    if resample_orientation and not teleop_mode:
        env_ids = torch.arange(env.num_envs, device=env.device)
        env._resample_ee_goal_orn_once(env_ids)

    values = ", ".join(
        [f"{field}={stage[field]}" for field in _ORIENTATION_RANGE_NAMES]
        + [
            f"base_height={stage['base_height']}",
            f"tracking_ee_world={stage['tracking_ee_world']}",
        ]
    )
    print(f"Staged training: applied {stage['name']} ({values})", flush=True)
    return stage


def learn_with_staged_training(
    runner,
    env_cfg,
    target_iteration,
    init_at_random_ep_len=True,
):
    """Train to an absolute iteration target and switch stages in-place."""
    schedule = getattr(env_cfg, "staged_training", None)
    if schedule is None or not bool(getattr(schedule, "enabled", False)):
        runner.learn(
            num_learning_iterations=int(target_iteration),
            init_at_random_ep_len=init_at_random_ep_len,
        )
        return

    switch_iteration = int(schedule.switch_iteration)
    target_iteration = int(target_iteration)
    current_iteration = int(runner.current_learning_iteration)
    if switch_iteration < 0:
        raise ValueError(
            "staged_training.switch_iteration must be non-negative, got "
            f"{switch_iteration}"
        )
    if target_iteration < 0:
        raise ValueError(
            f"target_iteration must be non-negative, got {target_iteration}"
        )
    # Validate both stages before launching a potentially long first segment.
    _read_stage(schedule.stage1)
    _read_stage(schedule.stage2)

    randomize_episode_lengths = bool(init_at_random_ep_len)
    active_stage_name = None

    while current_iteration < target_iteration:
        if current_iteration < switch_iteration:
            stage_cfg = schedule.stage1
            segment_end = min(switch_iteration, target_iteration)
        else:
            stage_cfg = schedule.stage2
            segment_end = target_iteration

        stage_name = str(stage_cfg.name)
        apply_staged_training_parameters(
            runner.env,
            stage_cfg,
            resample_orientation=(
                stage_name != active_stage_name
                and current_iteration >= switch_iteration
            ),
        )
        active_stage_name = stage_name

        segment_iterations = segment_end - current_iteration
        print(
            f"Staged training: global iterations {current_iteration} -> "
            f"{segment_end} use {stage_name}",
            flush=True,
        )
        runner.learn(
            num_learning_iterations=segment_iterations,
            init_at_random_ep_len=randomize_episode_lengths,
        )
        randomize_episode_lengths = False

        updated_iteration = int(runner.current_learning_iteration)
        if updated_iteration != segment_end:
            raise RuntimeError(
                "Runner iteration mismatch after staged training segment: "
                f"expected {segment_end}, got {updated_iteration}"
            )
        current_iteration = updated_iteration

    if current_iteration >= target_iteration:
        stage_cfg = (
            schedule.stage1
            if current_iteration < switch_iteration
            else schedule.stage2
        )
        if active_stage_name is None:
            apply_staged_training_parameters(
                runner.env,
                stage_cfg,
                resample_orientation=False,
            )
        if current_iteration > target_iteration:
            print(
                f"Staged training: checkpoint {current_iteration} is already "
                f"past target {target_iteration}; no updates were run",
                flush=True,
            )
