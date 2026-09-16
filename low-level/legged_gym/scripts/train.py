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
from legged_gym.utils import get_args, task_registry
import torch
import torch.distributed as dist
import wandb

from legged_gym.distributed import configure_distributed

def train(args):
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    if args.debug:
        mode = "disabled"
        args.rows = 6
        args.cols = 2
        args.num_envs = 128
    else:
        mode = "disabled" if getattr(args, "disable_wandb", False) else "online"
    config_path = getattr(args, "config", None)
    if config_path is not None:
        config_path = os.path.abspath(os.path.expanduser(config_path))
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, config_path=config_path)
    try:
        configure_distributed(args, default_seed=train_cfg.seed)
        if args.distributed:
            num_envs = env_cfg.env.num_envs if args.num_envs is None else args.num_envs
            batch_size = num_envs * train_cfg.runner.num_steps_per_env
            if num_envs <= 0 or batch_size % train_cfg.algorithm.num_mini_batches:
                raise ValueError("Per-rank rollout size must be positive and divisible by num_mini_batches.")
            print("Rank {}/{}: sim={}, learner={}, envs={} (global={}), seed={}".format(
                args.rank, args.world_size, args.sim_device, args.rl_device,
                num_envs, num_envs * args.world_size, args.seed))
        if args.rank == 0:
            os.makedirs(log_pth, exist_ok=True)
            wandb_dir = os.path.join(LEGGED_GYM_ENVS_DIR, "logs")
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(project=args.proj_name, name=args.exptid, mode=mode, dir=wandb_dir)
            wandb.save(config_path or LEGGED_GYM_ENVS_DIR + f"/manip_loco/{args.task}_config.py", policy="now")
            wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/manip_loco.py", policy="now")

        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        ppo_runner, train_cfg, _ = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg)
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if getattr(args, "rank", 0) == 0 and wandb.run is not None:
            wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
