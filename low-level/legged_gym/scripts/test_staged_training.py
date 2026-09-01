from pathlib import Path
from types import SimpleNamespace
import sys

import torch


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.utils.staged_training import learn_with_staged_training


def _stage(name, ranges, base_height, tracking_ee_world):
    return SimpleNamespace(
        name=name,
        delta_orn_r=ranges[0],
        delta_orn_p=ranges[1],
        delta_orn_y=ranges[2],
        base_height=base_height,
        tracking_ee_world=tracking_ee_world,
    )


def _make_env():
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        goal_ee_ranges={},
        reward_scales={"base_height": -5.0},
        arm_reward_scales={"tracking_ee_world": 0.8},
    )
    env.cfg = SimpleNamespace(
        env=SimpleNamespace(teleop_mode=False),
        goal_ee=SimpleNamespace(ranges=SimpleNamespace()),
        rewards=SimpleNamespace(
            scales=SimpleNamespace(base_height=-5.0),
            arm_scales=SimpleNamespace(tracking_ee_world=0.8),
        ),
    )
    env.resampled_ranges = []

    def resample(env_ids):
        assert torch.equal(env_ids, torch.arange(env.num_envs))
        env.resampled_ranges.append(list(env.goal_ee_ranges["delta_orn_r"]))

    env._resample_ee_goal_orn_once = resample
    return env


def _make_runner(start_iteration):
    runner = SimpleNamespace(
        env=_make_env(),
        current_learning_iteration=start_iteration,
        calls=[],
    )

    def learn(num_learning_iterations, init_at_random_ep_len=False):
        runner.calls.append(
            (
                runner.current_learning_iteration,
                num_learning_iterations,
                init_at_random_ep_len,
            )
        )
        runner.current_learning_iteration += num_learning_iterations

    runner.learn = learn
    return runner


def _make_cfg():
    return SimpleNamespace(
        staged_training=SimpleNamespace(
            enabled=True,
            switch_iteration=37000,
            stage1=_stage(
                "original",
                ([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]),
                -5.0,
                0.8,
            ),
            stage2=_stage(
                "dq_net",
                ([-1.5, 1.5], [-1.2, 1.6], [-0.8, 0.8]),
                -4.0,
                1.5,
            ),
        )
    )


def main():
    fresh_runner = _make_runner(0)
    learn_with_staged_training(fresh_runner, _make_cfg(), 45000)
    assert fresh_runner.calls == [(0, 37000, True), (37000, 8000, False)]
    assert fresh_runner.current_learning_iteration == 45000
    assert fresh_runner.env.reward_scales["base_height"] == -4.0
    assert fresh_runner.env.arm_reward_scales["tracking_ee_world"] == 1.5
    assert fresh_runner.env.resampled_ranges == [[-1.5, 1.5]]

    resumed_runner = _make_runner(37000)
    learn_with_staged_training(resumed_runner, _make_cfg(), 45000)
    assert resumed_runner.calls == [(37000, 8000, True)]
    assert resumed_runner.current_learning_iteration == 45000
    assert resumed_runner.env.resampled_ranges == [[-1.5, 1.5]]

    mid_stage1_runner = _make_runner(20000)
    learn_with_staged_training(mid_stage1_runner, _make_cfg(), 45000)
    assert mid_stage1_runner.calls == [
        (20000, 17000, True),
        (37000, 8000, False),
    ]

    print("staged training smoke test passed")


if __name__ == "__main__":
    main()
