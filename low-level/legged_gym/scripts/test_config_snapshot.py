import importlib.util
import math
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.utils.config_snapshot import save_config_snapshot


def main():
    env_cfg = {
        "env": {"num_envs": 128, "observe_gait_commands": True},
        "array_value": np.array([1.0, 2.0]),
        "nan_value": float("nan"),
    }
    train_cfg = {
        "runner": {"max_iterations": 5000},
        "tensor_value": torch.tensor([3, 4]),
        "inf_value": float("inf"),
    }
    args = Namespace(task="b1z1", manipulability_enabled=True)
    with tempfile.TemporaryDirectory() as log_dir:
        config_path = save_config_snapshot(log_dir, env_cfg, train_cfg, args)
        spec = importlib.util.spec_from_file_location("saved_training_config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert Path(config_path).name == "config.py"
        assert module.env_cfg["env"]["observe_gait_commands"] is True
        assert module.env_cfg["array_value"] == [1.0, 2.0]
        assert math.isnan(module.env_cfg["nan_value"])
        assert module.train_cfg["tensor_value"] == [3, 4]
        assert math.isinf(module.train_cfg["inf_value"])
        assert module.args["task"] == "b1z1"
        assert module.args["manipulability_enabled"] is True

    print("config snapshot smoke test passed")


if __name__ == "__main__":
    main()
