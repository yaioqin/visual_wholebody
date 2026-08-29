"""Two-iteration CPU smoke test for DWBC rollout, adaptation and PPO update."""

from pathlib import Path
import sys


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

import isaacgym  # noqa: F401 - must precede torch
import wandb

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import task_registry


def main():
    args = get_args(test=True)
    args.task = "b1z1"
    args.num_envs = 2
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 2
    env_cfg.env.action_delay = -1
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.tot_cols = 400
    env_cfg.terrain.tot_rows = 1600
    env_cfg.terrain.zScale = 0.0
    env_cfg.terrain.transform_x = -5.0
    env_cfg.terrain.transform_y = -20.0

    train_cfg.runner.resume = False
    train_cfg.runner.num_steps_per_env = 2
    train_cfg.runner.max_iterations = 2
    train_cfg.runner.save_interval = 100
    train_cfg.algorithm.num_learning_epochs = 1
    train_cfg.algorithm.num_mini_batches = 1

    log_root = Path("/tmp/dwbc_training_smoke")
    log_root.mkdir(parents=True, exist_ok=True)
    wandb.init(project="dwbc-smoke", mode="disabled")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _, _ = task_registry.make_alg_runner(
        env=env,
        args=args,
        train_cfg=train_cfg,
        log_root=str(log_root),
    )
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
    wandb.finish()

    assert runner.alg.counter == 2
    assert runner.alg.get_value_mixing_ratio() > 0.0
    assert runner.alg.storage.step == 0
    print("DWBC rollout, adaptation and PPO training smoke test passed")


if __name__ == "__main__":
    main()
