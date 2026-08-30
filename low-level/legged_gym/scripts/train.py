# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime
import isaacgym

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
from legged_gym.utils.config_snapshot import save_config_snapshot
from legged_gym.utils.helpers import class_to_dict, get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.training import remaining_learning_iterations
import torch
import wandb

def train(args):
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    if args.debug:
        mode = "disabled"
        args.rows = 6
        args.cols = 2
        args.num_envs = 128
    else:
        mode = os.environ.get("WANDB_MODE")
    wandb_kwargs = {
        "project": args.proj_name,
        "name": args.exptid,
        "dir": LEGGED_GYM_ENVS_DIR +"/logs",
    }
    if mode:
        wandb_kwargs["mode"] = mode
    wandb.init(**wandb_kwargs)
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/b1z1_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/b1z1_config_3D.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/manip_loco.py", policy="now")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    config_path = save_config_snapshot(
        ppo_runner.log_dir,
        env_cfg=class_to_dict(env_cfg),
        train_cfg=class_to_dict(train_cfg),
        args=args,
    )
    print(f"Saved effective config to: {config_path}")
    current_iteration = ppo_runner.current_learning_iteration
    target_iteration = train_cfg.runner.max_iterations
    num_learning_iterations = remaining_learning_iterations(target_iteration, current_iteration)
    print(
        f"Training target: iteration {target_iteration}; "
        f"current checkpoint iteration: {current_iteration}; "
        f"remaining iterations: {num_learning_iterations}"
    )
    if num_learning_iterations == 0:
        print("Checkpoint has already reached the training target; skipping training.")
        return
    ppo_runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
