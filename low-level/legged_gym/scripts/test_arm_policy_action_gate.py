from pathlib import Path
import sys

LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
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
    env_cfg.domain_rand.push_robots = False
    env_cfg.multi_agent.use_arm_delta_action = True
    env_cfg.multi_agent.allow_arm_policy_action = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env


def main():
    args = get_args(test=True)
    env = make_env(args)
    actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
    env.step(actions)

    assert env.use_arm_delta_action
    assert not env.allow_arm_policy_action
    assert not env.use_policy_arm_delta_action
    assert torch.all(env.actions[:, 12:18] == 0.0)
    assert torch.all(env.last_arm_delta_action == 0.0)
    print("arm policy action gate smoke test passed")


if __name__ == "__main__":
    main()
