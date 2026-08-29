from pathlib import Path
import sys

LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils.coordination_metrics import CoordinationMetrics
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import task_registry


def make_env(args):
    if args.task == "widowGo1":
        args.task = "b1z1"
    args.num_envs = 4
    args.headless = True

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 4
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    if hasattr(env_cfg.terrain, "tot_cols"):
        env_cfg.terrain.tot_cols = 400
        env_cfg.terrain.tot_rows = 1600
        env_cfg.terrain.zScale = 0.0
        env_cfg.terrain.transform_x = -5.0
        env_cfg.terrain.transform_y = -20.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.action_delay = -1
    env_cfg.multi_agent.use_arm_delta_action = False
    env_cfg.multi_agent.allow_arm_policy_action = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env


def main():
    args = get_args(test=True)
    env = make_env(args)
    evaluator = CoordinationMetrics(env=env, cfg=env.cfg, warmup_steps=0)
    for _ in range(2):
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        obs, _, _, _, _, _ = env.step(actions)
        evaluator.update(env, actions, obs)

    assert not env.use_arm_delta_action
    assert env.allow_arm_policy_action
    assert not env.use_policy_arm_delta_action
    assert torch.any(env.actions[:, 12:18] != 0.0)
    assert env.arm_pos_targets.shape == (env.num_envs, 6)
    assert torch.any(env.torques[:, 12:18] != 0.0)
    summary = evaluator.summarize()
    assert summary["meta/eval_steps"] == 2
    assert summary["meta/arm_motion_source"] == "arm_pos_targets"
    assert summary["meta/arm_energy_source"] == "explicit_force_torque"
    print("DWBC direct arm-action smoke test passed")


if __name__ == "__main__":
    main()
