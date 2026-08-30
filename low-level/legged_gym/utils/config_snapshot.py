"""Utilities for saving the effective training configuration."""

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from pprint import pformat

import numpy as np
import torch


def _to_python_value(value):
    """Convert common config values to literals that are safe in a Python file."""
    if isinstance(value, np.ndarray):
        return _to_python_value(value.tolist())
    if isinstance(value, np.generic):
        return _to_python_value(value.item())
    if torch.is_tensor(value):
        return _to_python_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {
            _to_python_value(key): _to_python_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_to_python_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_python_value(item) for item in value)
    if isinstance(value, set):
        return {_to_python_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_to_python_value(item) for item in value)
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    return value


def save_config_snapshot(log_dir, env_cfg, train_cfg, args):
    """Save the resolved B1Z1 configs and command-line args beside checkpoints."""
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, "config.py")
    values = {
        "env_cfg": _to_python_value(env_cfg),
        "train_cfg": _to_python_value(train_cfg),
        "args": _to_python_value(vars(args)),
    }

    source = (
        '"""Auto-generated effective configuration for this training run.\n\n'
        "env_cfg and train_cfg contain the final values loaded through\n"
        "b1z1_config_3D.py and b1z1_config.py after all overrides.\n"
        "args contains every parsed training argument.\n"
        '"""\n\n'
        "from math import inf, nan\n\n"
        f"env_cfg = {pformat(values['env_cfg'], sort_dicts=False, width=120)}\n\n"
        f"train_cfg = {pformat(values['train_cfg'], sort_dicts=False, width=120)}\n\n"
        f"args = {pformat(values['args'], sort_dicts=False, width=120)}\n\n"
        '__all__ = ["env_cfg", "train_cfg", "args"]\n'
    )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=log_dir,
            prefix=".config.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(source)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, config_path)
    except Exception:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

    return config_path
