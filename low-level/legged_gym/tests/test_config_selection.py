"""Check config selection and training wiring without Isaac Gym or CUDA."""

import ast
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.returns = None
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class ConfigSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "custom.py"
        self.path.write_text(
            "class B2Z1RoughCfg:\n    value = 'selected env'\n"
            "class B2Z1RoughCfgPPO:\n    seed = 73\n    value = 'selected PPO'\n"
        )
        default_env = type("B2Z1RoughCfg", (), {})()
        default_train = type("B2Z1RoughCfgPPO", (), {"seed": 1})()
        self.registry = SimpleNamespace(
            env_cfgs={"b2z1": default_env}, train_cfgs={"b2z1": default_train})
        self.get_cfgs = load_function("utils/task_registry.py", "get_cfgs", dict(
            os=os, runpy=runpy, LeggedRobotCfg=object, LeggedRobotCfgPPO=object,
        ), class_name="TaskRegistry")

    def test_default_configs_are_preserved(self):
        env, train = self.get_cfgs(self.registry, "b2z1")
        self.assertIs(env, self.registry.env_cfgs["b2z1"])
        self.assertIs(train, self.registry.train_cfgs["b2z1"])
        self.assertEqual(env.seed, 1)

    def test_absolute_and_relative_file_select_both_configs(self):
        for path in [str(self.path), os.path.relpath(self.path)]:
            env, train = self.get_cfgs(self.registry, "b2z1", path)
            self.assertEqual((env.value, train.value), ("selected env", "selected PPO"))
            self.assertEqual((env.seed, train.seed), (73, 73))
            self.assertIsNot(env, self.registry.env_cfgs["b2z1"])
        self.assertEqual(self.registry.train_cfgs["b2z1"].seed, 1)

    def test_invalid_file_fails_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, "Config file not found"):
            self.get_cfgs(self.registry, "b2z1", str(self.path) + ".missing")
        self.path.write_text("class B2Z1RoughCfg: pass\n")
        with self.assertRaisesRegex(ValueError, "B2Z1RoughCfgPPO"):
            self.get_cfgs(self.registry, "b2z1", str(self.path))

    def test_cli_accepts_config(self):
        import argparse

        def parse_arguments(description, custom_parameters):
            parser = argparse.ArgumentParser(description=description)
            for param in custom_parameters:
                options = dict(param)
                parser.add_argument(options.pop("name"), **options)
            args = parser.parse_args(["--task", "b2z1", "--exptid", "test", "--config", str(self.path)])
            args.compute_device_id = 2
            args.sim_device_type = "cuda"
            return args

        get_args = load_function("utils/helpers.py", "get_args", dict(
            gymutil=SimpleNamespace(parse_arguments=parse_arguments)))
        self.assertEqual(get_args().config, str(self.path))

    def test_training_uses_selected_configs_and_logs_source(self):
        env_cfg, train_cfg = self.get_cfgs(self.registry, "b2z1", str(self.path))
        train_cfg.runner = SimpleNamespace(max_iterations=8000)
        registry = Mock()
        registry.get_cfgs.return_value = (env_cfg, train_cfg)
        registry.make_env.return_value = (Mock(), env_cfg)
        runner = Mock()
        registry.make_alg_runner.return_value = (runner, train_cfg, 37000)
        wandb = Mock()
        train = load_function("scripts/train.py", "train", dict(
            os=SimpleNamespace(path=os.path, makedirs=Mock()),
            LEGGED_GYM_ROOT_DIR=str(ROOT.parent), LEGGED_GYM_ENVS_DIR=str(ROOT / "envs"),
            task_registry=registry, configure_distributed=Mock(), wandb=wandb,
            dist=SimpleNamespace(is_available=lambda: False),
        ))
        args = SimpleNamespace(config=str(self.path), task="b2z1", proj_name="test",
                               exptid="test", debug=False, distributed=False, rank=0)
        train(args)
        registry.get_cfgs.assert_called_once_with("b2z1", config_path=str(self.path))
        self.assertIs(registry.make_env.call_args.kwargs["env_cfg"], env_cfg)
        self.assertIs(registry.make_alg_runner.call_args.kwargs["train_cfg"], train_cfg)
        wandb.save.assert_any_call(str(self.path), policy="now")
        runner.learn.assert_called_once_with(num_learning_iterations=8000, init_at_random_ep_len=True)


if __name__ == "__main__":
    unittest.main()
