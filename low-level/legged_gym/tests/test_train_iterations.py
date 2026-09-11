"""Check the training entry point's stopping condition without GPU simulation."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"


def load_train_script():
    utils = ModuleType("legged_gym.utils")
    utils.get_args = Mock()
    utils.task_registry = Mock()
    dependencies = {
        "numpy": ModuleType("numpy"),
        "isaacgym": ModuleType("isaacgym"),
        "torch": ModuleType("torch"),
        "wandb": Mock(),
        "legged_gym.envs": ModuleType("legged_gym.envs"),
        "legged_gym.utils": utils,
    }
    spec = importlib.util.spec_from_file_location("train_iterations_test", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class TrainIterationsTest(unittest.TestCase):
    def setUp(self):
        self.script = load_train_script()
        self.args = SimpleNamespace(
            task="b2z1",
            proj_name="b2z1-low",
            exptid="b2_z1_change_ee_pos_range",
            debug=False,
            resumeid="b2_z1",
            checkpoint=20000,
        )

    def run_training(self, current_iteration, target_iteration=45000):
        runner = Mock(current_learning_iteration=current_iteration)
        cfg = SimpleNamespace(runner=SimpleNamespace(max_iterations=target_iteration))
        self.script.task_registry.make_env.return_value = (Mock(), Mock())
        self.script.task_registry.make_alg_runner.return_value = (
            runner, cfg, current_iteration
        )
        with patch.object(self.script.os, "makedirs"), patch("builtins.print"):
            self.script.train(self.args)
        return runner

    def test_resume_from_20000_runs_only_25000_more_iterations(self):
        runner = self.run_training(20000)
        runner.learn.assert_called_once_with(
            num_learning_iterations=25000, init_at_random_ep_len=True
        )

    def test_fresh_training_runs_to_total_target(self):
        self.args.resumeid = None
        self.args.checkpoint = -1
        runner = self.run_training(0)
        runner.learn.assert_called_once_with(
            num_learning_iterations=45000, init_at_random_ep_len=True
        )

    def test_latest_checkpoint_uses_loaded_runner_iteration(self):
        self.args.checkpoint = -1
        runner = self.run_training(44000)
        runner.learn.assert_called_once_with(
            num_learning_iterations=1000, init_at_random_ep_len=True
        )

    def test_custom_total_target_is_respected(self):
        runner = self.run_training(20000, target_iteration=21000)
        runner.learn.assert_called_once_with(
            num_learning_iterations=1000, init_at_random_ep_len=True
        )

    def test_checkpoint_at_or_above_target_does_not_train(self):
        for current_iteration in (45000, 46000):
            with self.subTest(current_iteration=current_iteration):
                runner = self.run_training(current_iteration)
                runner.learn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
