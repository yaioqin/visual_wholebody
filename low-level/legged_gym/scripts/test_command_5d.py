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
    env_cfg.commands.use_5d_base_command = True
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env


def main():
    args = get_args(test=True)
    env = make_env(args)
    env._resample_commands(torch.arange(env.num_envs, device=env.device))

    pitch_min, pitch_max = env.command_ranges["base_pitch"]
    height_min, height_max = env.command_ranges["base_height"]
    assert env.commands.shape[1] >= 5
    assert torch.all(env.commands[:, 3] >= pitch_min)
    assert torch.all(env.commands[:, 3] <= pitch_max)
    assert torch.all(env.commands[:, 4] >= height_min)
    assert torch.all(env.commands[:, 4] <= height_max)

    base_command_obs = env.commands[:, :5]
    assert base_command_obs.shape == (env.num_envs, 5)
    print("5D command smoke test passed")


if __name__ == "__main__":
    main()
