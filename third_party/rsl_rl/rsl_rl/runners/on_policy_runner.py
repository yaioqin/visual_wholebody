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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, MultiAgentActorCritic
from rsl_rl.env import VecEnv

import wandb
from torchinfo import summary


def _format_seconds(seconds):
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _format_progress_bar(current, total, width=30):
    total = max(total, 1)
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {ratio * 100:5.1f}%"


def _accumulate_scalar_diagnostics(sums, diagnostics):
    if not diagnostics:
        return False
    for key, value in diagnostics.items():
        if isinstance(value, torch.Tensor):
            scalar = value.detach().float()
            scalar = scalar.mean() if scalar.numel() > 1 else scalar.reshape(())
        else:
            scalar = torch.tensor(float(value))
        if key in sums:
            sums[key] = sums[key] + scalar
        else:
            sums[key] = scalar
    return True


def _average_scalar_diagnostics(sums, count):
    if count <= 0:
        return {}
    return {key: value / count for key, value in sums.items()}


def _scalar_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        if actor_critic_class is None:
            raise ImportError(f"Policy class {self.cfg['policy_class_name']} is not available")
        if self.cfg["policy_class_name"] == "MultiAgentActorCritic":
            num_actor_obs = self.env.num_obs
            num_critic_obs_for_model = num_critic_obs
        else:
            num_actor_obs = self.env.cfg.env.num_proprio
            num_critic_obs_for_model = self.env.cfg.env.num_proprio
        actor_critic: ActorCritic = actor_critic_class( num_actor_obs,
                                                        num_critic_obs_for_model,
                                                        self.env.num_actions,
                                                        **self.policy_cfg, 
                                                        num_priv=env.cfg.env.num_priv,
                                                        num_hist=env.cfg.env.history_len, 
                                                        num_prop=env.cfg.env.num_proprio,
                                                        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        if hasattr(self.env, "allow_arm_policy_action") and not self.env.allow_arm_policy_action:
            self.alg.only_train_leg = True
            print("Arm policy action disabled: PPO surrogate updates leg actions only.")
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        summary(self.alg.actor_critic)

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        _, _ = self.env.reset()

        self.alg.set_arm_default_coeffs(self.env.p_gains[12:], self.env.d_gains[12:], self.env.default_dof_pos[-7:-2])
        
    def set_it(self, it):
        self.current_learning_iteration = it
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # init metrics
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_arm_torques_loss = 0.
        value_mixing_ratio = 0.
        torque_supervision_weight = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.

        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        armrewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        donebuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_arm_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        progress_start_time = time.time()
        tot_iter = self.current_learning_iteration + num_learning_iterations
        print(
            f"Training progress: {num_learning_iterations} iterations "
            f"(from {self.current_learning_iteration} to {tot_iter})"
        )
        for it in range(self.current_learning_iteration, tot_iter):
            # self.env.update_command_curriculum()

            start = time.time()
            hist_encoding = self.alg.supports_dagger and it % self.dagger_update_freq == 0
            low_level_diagnostic_sums = {}
            low_level_diagnostic_count = 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, hist_encoding)
                    obs, privileged_obs, rewards, arm_rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, arm_rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), arm_rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, arm_rewards, dones, infos)
                    if _accumulate_scalar_diagnostics(
                        low_level_diagnostic_sums,
                        getattr(self.env, "low_level_log_diagnostics", None),
                    ):
                        low_level_diagnostic_count += 1
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_arm_reward_sum += arm_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        armrewbuffer.extend(cur_arm_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        donebuffer.append(len(new_ids) / self.env.num_envs)
                        cur_reward_sum[new_ids] = 0
                        cur_arm_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # self.alg.storage.clear()
            
            # mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            if hist_encoding:
                mean_hist_latent_loss = self.alg.update_dagger()
            else:
                mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            low_level_command_tracking_diagnostics = _average_scalar_diagnostics(
                low_level_diagnostic_sums,
                low_level_diagnostic_count,
            )
            completed_iterations = it - (tot_iter - num_learning_iterations) + 1
            elapsed_time = stop - progress_start_time
            avg_iteration_time = elapsed_time / completed_iterations
            remaining_iterations = num_learning_iterations - completed_iterations
            remaining_training_time = avg_iteration_time * remaining_iterations
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), it)
            fps = int(self.num_steps_per_env * self.env.num_envs / (collection_time + learn_time))
            progress_bar = _format_progress_bar(completed_iterations, num_learning_iterations)
            print(
                f"Training progress {progress_bar} "
                f"{completed_iterations}/{num_learning_iterations} "
                f"(global iteration {it + 1}/{tot_iter}) | "
                f"fps: {fps} | iter: {collection_time + learn_time:.2f}s | "
                f"elapsed: {_format_seconds(elapsed_time)} | "
                f"remaining: {_format_seconds(remaining_training_time)}",
                flush=True,
            )
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), self.current_learning_iteration)
        print(f"Training progress complete: {num_learning_iterations}/{num_learning_iterations} iterations")

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb.log({'Episode/' + key: value}, step=locs['it'])
                if "rew" in key:
                    wandb_dict['Episode_rew/' + key] = value
                elif "metric" in key:
                    wandb_dict['Episode_metric/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        leg_mean_std = self.alg.actor_critic.std[:, :12].mean()
        arm_mean_std = self.alg.actor_critic.std[:, 12:].mean()
        std_numpy = self.alg.actor_critic.std.cpu().detach().numpy()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/arm_torques_loss'] = locs['mean_arm_torques_loss']
        wandb_dict['Loss/value_mixing_ratio'] = locs['value_mixing_ratio']
        wandb_dict['Loss/torque_supervision_weight'] = locs['torque_supervision_weight']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Policy/leg_mean_noise_std'] = leg_mean_std.item()
        wandb_dict['Policy/arm_mean_noise_std'] = arm_mean_std.item()
        wandb_dict['Policy/noise_std_dist'] = wandb.Histogram(std_numpy)
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if 'remaining_training_time' in locs:
            wandb_dict['Perf/elapsed_training_time_s'] = locs['elapsed_time']
            wandb_dict['Perf/remaining_training_time_s'] = locs['remaining_training_time']
            wandb_dict['Perf/avg_iteration_time_s'] = locs['avg_iteration_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_arm_reward'] = statistics.mean(locs['armrewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['Train/dones'] = statistics.mean(locs['donebuffer'])
            # wandb.log({'Train/mean_reward/time': statistics.mean(locs['rewbuffer'])}, step=self.tot_time)
            # wandb.log({'Train/mean_episode_length/time': statistics.mean(locs['lenbuffer'])}, step=self.tot_time)
        command_tracking_diagnostics = locs.get('low_level_command_tracking_diagnostics', {})
        for key, value in command_tracking_diagnostics.items():
            wandb_dict['CommandTracking/' + key] = value
        
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'History latent supervision loss:':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
                          f"""{'Privileged info regularizer loss:':>{pad}} {locs['mean_priv_reg_loss']:.4f}\n"""
                          f"""{'Privileged info regularizer lambda:':>{pad}} {locs['priv_reg_coef']:.4f}\n"""
                          f"""{'Leg mean action noise std:':>{pad}} {leg_mean_std.item():.2f}\n"""
                          f"""{'Arm mean action noise std:':>{pad}} {arm_mean_std.item():.2f}\n"""
                          f"""{'action noise std distribution:':>{pad}} {std_numpy.tolist()}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Dones:':>{pad}} {statistics.mean(locs['donebuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'History latent supervision loss:':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
                          f"""{'Leg mean action noise std:':>{pad}} {leg_mean_std.item():.2f}\n"""
                          f"""{'Arm mean action noise std:':>{pad}} {arm_mean_std.item():.2f}\n"""
                          f"""{'action noise std distribution:':>{pad}} {std_numpy.tolist()}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        if command_tracking_diagnostics:
            log_string += f"""{'5D command tracking:':>{pad}}\n"""
            for key in (
                "mean_abs_vx_error",
                "mean_abs_vy_error",
                "mean_abs_yaw_rate_error",
                "mean_abs_base_pitch_error",
                "mean_abs_base_height_error",
                "mean_base_height_command",
                "mean_base_pitch_command",
                "mean_root_z",
                "episode_length_mean",
                "reset_rate",
                "fall_rate",
                "torque_penalty",
                "action_rate_penalty",
            ):
                if key in command_tracking_diagnostics:
                    log_string += f"""{key + ':':>{pad}} {_scalar_to_float(command_tracking_diagnostics[key]):.4f}\n"""

        log_string += ep_string
        remaining_training_time = locs.get('remaining_training_time', None)
        remaining_line = ""
        if remaining_training_time is not None:
            remaining_line = f"""{'Remaining training time:':>{pad}} {_format_seconds(remaining_training_time)} ({remaining_training_time:.1f}s)\n"""
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{remaining_line}""")
        print(log_string)

    def save(self, path, it, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': it,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None, stochastic=False):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)

        if not stochastic:
            return self.alg.actor_critic.act_inference
        else:
            return self.alg.actor_critic.act
