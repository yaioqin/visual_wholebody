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

"""Synchronous PPO for independent, equal-sized simulator shards.

The update bodies follow the project's RSL-RL PPO implementation, including its
two value heads, arm torque supervision and separate history encoder optimizer.
Only initial state, advantage statistics, adaptive KL and gradients are shared.
Initialize torch.distributed before constructing DistributedPPO.
"""

from itertools import chain

import torch
import torch.distributed as dist
from torch import nn

from rsl_rl.algorithms import PPO


@torch.no_grad()
def broadcast_model(module, src=0):
    """Make parameters and registered buffers identical on every rank."""
    for tensor in chain(module.parameters(), module.buffers()):
        dist.broadcast(tensor, src=src)


@torch.no_grad()
def global_mean(values):
    """Mean over every element on every rank, allowing different local sizes."""
    statistics = torch.stack(
        (values.double().sum(), values.new_tensor(values.numel(), dtype=torch.float64))
    )
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    if statistics[1].item() == 0:
        raise ValueError("Cannot average an empty distributed tensor")
    return (statistics[0] / statistics[1]).to(values.dtype)


@torch.no_grad()
def normalize_global_advantages(advantages):
    """Match RolloutStorage's scalar mean/sample std across all rank shards.

    Both value heads participate in the same statistics, as in the single-GPU
    implementation. A second pass around the global mean avoids cancellation
    when advantages have a large offset or a small variance.
    """
    values = advantages.double()
    statistics = torch.stack(
        (values.sum(), values.new_tensor(values.numel()))
    )
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    if statistics[1].item() < 2:
        raise ValueError("Sample standard deviation needs at least two advantages")
    mean = statistics[0] / statistics[1]
    squared_deviations = (values - mean).square().sum()
    dist.all_reduce(squared_deviations, op=dist.ReduceOp.SUM)
    std = (squared_deviations / (statistics[1] - 1)).sqrt()
    return (advantages - mean.to(advantages.dtype)) / (std.to(advantages.dtype) + 1e-8)


class FlatGradientAverager:
    """Average dense gradients and their usage mask in one collective.

    Every rank must supply the same fixed parameter sequence. A missing local
    gradient contributes zero; a parameter unused on every rank keeps grad=None
    so Adam cannot move it through momentum from an earlier update.
    """

    def __init__(self, parameters):
        self.parameters = tuple(p for p in parameters if p.requires_grad)
        self.world_size = dist.get_world_size()
        self.sizes = tuple(p.numel() for p in self.parameters)
        self.numel = sum(self.sizes)
        if not self.parameters:
            self.buffer = None
            return
        reference = self.parameters[0]
        if any(
            p.device != reference.device or p.dtype != reference.dtype
            for p in self.parameters
        ):
            raise ValueError("Distributed PPO requires one parameter device and dtype")
        self.buffer = reference.new_zeros(self.numel + len(self.parameters))

    @torch.no_grad()
    def average(self):
        if self.buffer is None:
            return
        self.buffer.zero_()
        offset = 0
        usage = self.buffer[self.numel:]
        for index, (parameter, size) in enumerate(zip(self.parameters, self.sizes)):
            if parameter.grad is not None:
                self.buffer[offset:offset + size].copy_(parameter.grad.reshape(-1))
                usage[index] = 1
            offset += size
        dist.all_reduce(self.buffer, op=dist.ReduceOp.SUM)
        self.buffer[:self.numel].div_(self.world_size)
        # One host synchronization for all usage flags, not one per parameter.
        used_on_any_rank = usage.tolist()
        offset = 0
        for parameter, size, used in zip(self.parameters, self.sizes, used_on_any_rank):
            if used:
                gradient = self.buffer[offset:offset + size].view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = gradient.clone()
                else:
                    parameter.grad.copy_(gradient)
            else:
                parameter.grad = None
            offset += size


def average_gradients(parameters):
    """Average a fixed parameter sequence; use FlatGradientAverager to reuse memory."""
    FlatGradientAverager(parameters).average()


class DistributedPPO(PPO):
    """PPO with synchronized updates; constructor arguments are identical to PPO."""

    def __init__(self, *args, **kwargs):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before DistributedPPO")
        super().__init__(*args, **kwargs)
        self.world_size = dist.get_world_size()
        broadcast_model(self.actor_critic)
        self._policy_gradients = FlatGradientAverager(self.actor_critic.parameters())
        self._history_gradients = FlatGradientAverager(
            self.actor_critic.actor.history_encoder.parameters()
        )

    def compute_returns(self, last_critic_obs):
        super().compute_returns(last_critic_obs)
        self.storage.advantages = normalize_global_advantages(
            self.storage.returns - self.storage.values
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_arm_torques_loss = 0
        mean_priv_reg_loss = 0
        value_mixing_ratio = self.get_value_mixing_ratio()
        torque_supervision_weight = self.get_torque_supervision_weight() if self.torque_supervision else 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, target_arm_torques, current_arm_dof_pos, current_arm_dof_vel, hid_states_batch, masks_batch in generator:

            self.actor_critic.act(obs_batch, hist_encoding=False, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptation module update
            priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
            priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]
            # priv_reg_loss = torch.zeros(1, device=self.device)

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = global_mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate


            # Surrogate loss
            only_train_leg = False

            mixing_advantages_batch = torch.zeros_like(advantages_batch)
            if only_train_leg == True:
                mixing_advantages_batch[..., 0] = advantages_batch[..., 0]
                mixing_advantages_batch[..., 1] = advantages_batch[..., 1]
            else:
                mixing_advantages_batch[..., 0] = advantages_batch[..., 0] + value_mixing_ratio * advantages_batch[..., 1]
                mixing_advantages_batch[..., 1] = advantages_batch[..., 1] + value_mixing_ratio * advantages_batch[..., 0]
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = - mixing_advantages_batch * ratio
            surrogate_clipped = - mixing_advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            if only_train_leg == True:
                surrogate_loss = torch.max(surrogate, surrogate_clipped)[:, 0].mean()
            else:
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss \
                   + self.value_loss_coef * value_loss \
                   - self.entropy_coef * entropy_batch.mean() \
                   + priv_reg_coef * priv_reg_loss


            # adaptive arm gains
            if self.adaptive_arm_gains:
                actions_mean = self.actor_critic.act_inference(obs_batch)
                target_arm_dof_pos = actions_mean[:, 12: -6]
                delta_arm_p_gains = actions_mean[:, -6:]
            else:
                target_arm_dof_pos = self.actor_critic.act_inference(obs_batch)[:, -6:]
                delta_arm_p_gains = None

            # arm torque supervision
            if self.torque_supervision:
                arm_torques = self.arm_fk(delta_arm_p_gains, target_arm_dof_pos, current_arm_dof_pos, current_arm_dof_vel)
                arm_torques_loss = (arm_torques - target_arm_torques).pow(2).mean()
                torque_supervision_weight = self.get_torque_supervision_weight()
                loss += arm_torques_loss * torque_supervision_weight
                mean_arm_torques_loss += arm_torques_loss.item()


            # Gradient step
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._policy_gradients.average()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_arm_torques_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        self.storage.clear()

        self.update_counter()

        self.enforce_min_std()

        return mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef

    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, target_arm_torques, current_arm_dof_pos, current_arm_dof_vel, hid_states_batch, masks_batch in generator:
            with torch.inference_mode():
                self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

            # Adaptation module update
            with torch.inference_mode():
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad(set_to_none=True)
            hist_latent_loss.backward()
            self._history_gradients.average()
            nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()

            mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss
