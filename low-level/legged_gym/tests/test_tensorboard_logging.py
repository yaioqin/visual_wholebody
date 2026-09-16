"""Exercise runner logging with in-memory writers, without native training dependencies."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class Tensor:
    def __init__(self, values):
        self.values = list(values) if isinstance(values, (list, tuple)) else [values]

    def __getitem__(self, key):
        return Tensor(self.values[key[-1]])

    def __truediv__(self, divisor):
        return Tensor([v / divisor for v in self.values])

    def mean(self):
        return Tensor(statistics.mean(self.values))

    def item(self):
        return self.values[0]

    def tolist(self):
        return self.values

    def reshape(self, *args):
        return self

    float = detach = cpu = reshape


class TensorBoardLoggingTest(unittest.TestCase):
    def setUp(self):
        # Load the complete production class while replacing imports requiring
        # PyTorch/RSL-RL. Its learn/log methods and super() calls run normally.
        path = Path(__file__).resolve().parents[1] / "on_policy_runner.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OnPolicyRunner")
        self.parent = type("Parent", (), {})
        self.parent.learn = lambda *args: None
        self.parent.log = lambda runner, locs, *args: setattr(
            runner, "tot_timesteps", runner.tot_timesteps + 48)
        self.dist = Mock()
        self.bar = Mock()
        self.bar.n = 37
        torch = SimpleNamespace(
            tensor=lambda values, **kwargs: Tensor(values),
            as_tensor=lambda values, **kwargs: Tensor(values),
            cat=lambda values: Tensor([v for tensor in values for v in tensor.values]),
        )
        namespace = dict(RslOnPolicyRunner=self.parent, torch=torch, dist=self.dist,
                         tqdm=Mock(return_value=self.bar), os=os, statistics=statistics,
                         redirect_stdout=redirect_stdout, StringIO=StringIO)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        self.runner = namespace["OnPolicyRunner"].__new__(namespace["OnPolicyRunner"])
        self.runner.__dict__.update(
            rank=0, world_size=1, distributed=False, log_dir="logs/test/run", writer=None,
            current_learning_iteration=37, device="cpu", num_steps_per_env=24,
            tot_timesteps=0, tot_time=0, env=SimpleNamespace(num_envs=2),
            alg=SimpleNamespace(learning_rate=0.001, actor_critic=SimpleNamespace(std=Tensor([1.] * 18))),
        )
        self.metrics = dict.fromkeys([
            "mean_value_loss", "mean_surrogate_loss", "mean_arm_torques_loss",
            "mean_hist_latent_loss", "mean_priv_reg_loss", "priv_reg_coef",
            "value_mixing_ratio", "torque_supervision_weight",
        ], 0.5)
        self.metrics.update(it=37, collection_time=1.5, learn_time=0.5,
                            rewbuffer=[2., 4.], armrewbuffer=[1., 2.], lenbuffer=[10., 20.],
                            donebuffer=[0.5], ep_infos=[{"rew_tracking": [2., 4.]}, {"rew_tracking": 6.}])
        self.writer = Mock()
        self.factory = Mock(return_value=self.writer)
        self.wandb = Mock()
        self.modules = patch.dict(sys.modules, {
            "torch.utils.tensorboard": SimpleNamespace(SummaryWriter=self.factory),
            "wandb": self.wandb,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def scalars(self):
        return {call.args[0]: call.args[1] for call in self.writer.add_scalar.call_args_list}

    def test_single_gpu_logs_rewards_losses_and_resume_step_without_wandb(self):
        self.parent.learn = lambda runner, *args: runner.log(self.metrics)
        with redirect_stdout(StringIO()):
            self.runner.learn(1)
        self.factory.assert_called_once_with(log_dir="logs/test/run/tensorboard", flush_secs=10)
        metrics = self.scalars()
        self.assertEqual(metrics["Train/mean_reward"], 3.)
        self.assertEqual(metrics["Episode_rew/rew_tracking"], 4.)
        self.assertEqual(metrics["Loss/value_function"], 0.5)
        self.assertEqual(metrics["Loss/learning_rate"], 0.001)
        self.assertEqual(metrics["Perf/total_fps"], 24.)
        self.assertEqual(metrics["Perf/total_timesteps"], 48)
        self.assertEqual(metrics["Policy/arm_mean_noise_std"], 1.)
        self.assertTrue(all(c.kwargs["global_step"] == 37 for c in self.writer.add_scalar.call_args_list))
        self.writer.add_histogram.assert_called_once()
        self.writer.close.assert_called_once()
        self.assertIsNone(self.runner.writer)

    def test_distributed_rank_zero_writes_same_reduced_scalars_as_wandb(self):
        self.runner.distributed = True
        self.runner.world_size = 3
        self.parent.learn = lambda runner, *args: runner.log(self.metrics)
        with redirect_stdout(StringIO()):
            self.runner.learn(1)
        self.assertEqual(self.scalars(), self.wandb.log.call_args.args[0])
        self.assertEqual(self.scalars()["Distributed/global_envs"], 6)
        self.assertEqual(self.scalars()["Rank0/Train/mean_reward"], 3.)
        self.assertEqual(self.dist.all_reduce.call_count, 2)

    def test_worker_rank_never_opens_writer(self):
        self.runner.distributed = True
        self.runner.world_size = 3
        self.runner.rank = 1
        self.parent.learn = lambda runner, *args: runner.log(self.metrics)
        self.runner.learn(1)
        self.factory.assert_not_called()
        self.writer.add_scalar.assert_not_called()
        self.wandb.log.assert_not_called()
        self.assertEqual(self.dist.all_reduce.call_count, 2)

    def test_exception_closes_writer_and_propagates(self):
        def fail(*args):
            raise RuntimeError("rollout failed")
        self.parent.learn = fail
        with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "rollout failed"):
            self.runner.learn(1)
        self.writer.close.assert_called_once()
        self.bar.close.assert_called_once()
        self.assertIsNone(self.runner.writer)


if __name__ == "__main__":
    unittest.main()
