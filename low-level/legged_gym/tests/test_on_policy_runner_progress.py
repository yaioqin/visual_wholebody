import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch, sentinel

import torch

from legged_gym.on_policy_runner import OnPolicyRunner, RslOnPolicyRunner


class ProgressRecorder:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.n = kwargs["initial"]
        self.postfixes = []
        self.closed = False

    def update(self, count):
        self.n += count

    def set_postfix(self, **kwargs):
        self.postfixes.append(kwargs)

    def close(self):
        self.closed = True


def make_runner(distributed=False, rank=0, start_iteration=17):
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.distributed = distributed
    runner.rank = rank
    runner.world_size = 3 if distributed else 1
    runner.current_learning_iteration = start_iteration
    runner.log_dir = None
    runner.writer = None
    runner.device = "cpu"
    runner.env = SimpleNamespace(num_envs=2)
    runner.num_steps_per_env = 24
    runner.tot_timesteps = 0
    runner.tot_time = 0.0
    runner.alg = SimpleNamespace(
        learning_rate=0.001,
        actor_critic=SimpleNamespace(std=torch.ones(1, 18)),
    )
    return runner


def iteration_metrics(iteration):
    metrics = dict.fromkeys([
        "mean_value_loss", "mean_surrogate_loss", "mean_arm_torques_loss",
        "mean_hist_latent_loss", "mean_priv_reg_loss", "priv_reg_coef",
        "value_mixing_ratio", "torque_supervision_weight",
    ], 0.0)
    metrics.update({
        "it": iteration, "collection_time": 1.25, "learn_time": 0.25,
        "rewbuffer": [], "armrewbuffer": [], "lenbuffer": [],
        "donebuffer": [], "ep_infos": [],
    })
    return metrics


class OnPolicyRunnerProgressTest(unittest.TestCase):
    def test_resume_progress_and_logging_on_single_and_distributed_rank_zero(self):
        for distributed in (False, True):
            with self.subTest(distributed=distributed):
                runner = make_runner(distributed=distributed)
                bar = ProgressRecorder(initial=17, total=20)

                def simulate_learning(instance, iterations, random_episode_lengths):
                    self.assertIs(instance, runner)
                    self.assertEqual(iterations, 3)
                    self.assertTrue(random_episode_lengths)
                    for iteration in range(17, 20):
                        instance.log(iteration_metrics(iteration))
                    return sentinel.learn_result

                output = io.StringIO()
                with patch("legged_gym.on_policy_runner.tqdm", return_value=bar) as progress, \
                     patch.object(RslOnPolicyRunner, "learn", autospec=True, side_effect=simulate_learning), \
                     patch.object(RslOnPolicyRunner, "log", side_effect=lambda *args: print("upstream report")) as upstream_log, \
                     patch("legged_gym.on_policy_runner.dist.all_reduce") as reduce, \
                     patch("wandb.log") as wandb_log, \
                     contextlib.redirect_stdout(output):
                    result = runner.learn(3, init_at_random_ep_len=True)

                self.assertIs(result, sentinel.learn_result)
                options = progress.call_args.kwargs
                self.assertEqual(options["initial"], 17)
                self.assertEqual(options["total"], 20)
                self.assertEqual(options["desc"], "Training (3 GPUs)" if distributed else "Training")
                self.assertIn("ETA", options["bar_format"])
                self.assertEqual(bar.n, 20)
                self.assertEqual(len(bar.postfixes), 3)
                self.assertEqual(bar.postfixes[-1]["rollout"], "1.25s")
                self.assertEqual(bar.postfixes[-1]["update"], "0.25s")
                self.assertTrue(bar.closed)
                self.assertIsNone(runner._progress_bar)
                self.assertEqual(output.getvalue(), "")
                self.assertEqual(upstream_log.call_count, 0 if distributed else 3)
                self.assertEqual(wandb_log.call_count, 3 if distributed else 0)
                self.assertEqual(reduce.call_count, 6 if distributed else 0)

    def test_learning_error_closes_progress_and_propagates(self):
        runner = make_runner()
        bar = ProgressRecorder(initial=17, total=20)
        with patch("legged_gym.on_policy_runner.tqdm", return_value=bar), \
             patch.object(RslOnPolicyRunner, "learn", side_effect=RuntimeError("rollout failed")):
            with self.assertRaisesRegex(RuntimeError, "rollout failed"):
                runner.learn(3)
        self.assertTrue(bar.closed)
        self.assertIsNone(runner._progress_bar)

    def test_other_ranks_reduce_metrics_without_creating_progress(self):
        runner = make_runner(distributed=True, rank=1)

        def simulate_learning(instance, iterations, random_episode_lengths):
            instance.log(iteration_metrics(17))

        with patch("legged_gym.on_policy_runner.tqdm") as progress, \
             patch.object(RslOnPolicyRunner, "learn", autospec=True, side_effect=simulate_learning), \
             patch("legged_gym.on_policy_runner.dist.all_reduce") as reduce, \
             patch("wandb.log") as wandb_log:
            runner.learn(3)
        progress.assert_not_called()
        wandb_log.assert_not_called()
        self.assertEqual(reduce.call_count, 2)
        self.assertIsNone(runner._progress_bar)


if __name__ == "__main__":
    unittest.main()
