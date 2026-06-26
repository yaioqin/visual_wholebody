from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from legged_gym.utils.action_codec import (
    HighLevelActionDims,
    flatten_high_level_action,
)


class MultiAgentHighLevelPolicy(nn.Module):
    """Multi-head high-level policy for base, arm chunk, grasp intent and h."""

    def __init__(
        self,
        obs_dim: int,
        chunk_horizon: int = 3,
        encoder_hidden_dims: Sequence[int] = (),
        backbone_hidden_dims: Sequence[int] = (256, 256),
        head_hidden_dims: Sequence[int] = (),
        activation: str = "elu",
    ):
        super().__init__()
        self.action_dims = HighLevelActionDims(chunk_horizon=chunk_horizon)
        act = get_activation(activation)

        encoder_layers, encoder_out_dim = build_mlp(obs_dim, encoder_hidden_dims, act)
        self.encoder = encoder_layers

        backbone_layers, backbone_out_dim = build_mlp(encoder_out_dim, backbone_hidden_dims, act)
        self.shared_backbone = backbone_layers

        self.base_head = build_head(
            backbone_out_dim, self.action_dims.base_command_dim, head_hidden_dims, act
        )
        self.arm_chunk_head = build_head(
            backbone_out_dim,
            self.action_dims.chunk_horizon * self.action_dims.ee_delta_dim,
            head_hidden_dims,
            act,
        )
        self.grasp_intent_head = build_head(
            backbone_out_dim, self.action_dims.grasp_intent_dim, head_hidden_dims, act
        )
        self.h_head = build_head(
            backbone_out_dim, self.action_dims.h_choice_dim, head_hidden_dims, act
        )

    @property
    def total_action_dim(self) -> int:
        return self.action_dims.total_action_dim

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = obs.shape[0]
        feat = self.encoder(obs)
        z = self.shared_backbone(feat)
        return {
            "base_command": self.base_head(z),
            "arm_action_chunk": self.arm_chunk_head(z).view(
                batch, self.action_dims.chunk_horizon, self.action_dims.ee_delta_dim
            ),
            "grasp_intent_logit": self.grasp_intent_head(z),
            "h_logits": self.h_head(z),
        }

    def flat_action(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.forward(obs)
        return flatten_high_level_action(
            {
                "base_command": out["base_command"],
                "arm_action_chunk": out["arm_action_chunk"],
                "grasp_intent": out["grasp_intent_logit"],
                "h_logits": out["h_logits"],
            }
        )


class MultiAgentActorCritic(nn.Module):
    """RSL-RL compatible wrapper around MultiAgentHighLevelPolicy.

    The actor emits a flat tensor so existing PPO storage, log-prob and entropy
    paths can stay unchanged. h logits and grasp intent logits are continuous
    Gaussian action dimensions here; the environment/wrapper should discretize
    h with argmax or categorical sampling before execution.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        activation="elu",
        init_std=1.0,
        chunk_horizon=3,
        encoder_hidden_dims=(),
        head_hidden_dims=(),
        value_dim=2,
        **kwargs,
    ):
        super().__init__()
        self.action_dims = HighLevelActionDims(chunk_horizon=chunk_horizon)
        if num_actions != self.action_dims.total_action_dim:
            raise ValueError(
                "MultiAgentActorCritic requires env.num_actions to equal "
                f"{self.action_dims.total_action_dim} for chunk_horizon={chunk_horizon}; "
                f"got {num_actions}."
            )

        self.actor = MultiAgentHighLevelPolicy(
            obs_dim=num_actor_obs,
            chunk_horizon=chunk_horizon,
            encoder_hidden_dims=encoder_hidden_dims,
            backbone_hidden_dims=actor_hidden_dims,
            head_hidden_dims=head_hidden_dims,
            activation=activation,
        )
        act = get_activation(activation)
        critic_layers, critic_out_dim = build_mlp(num_critic_obs, critic_hidden_dims, act)
        self.critic_backbone = critic_layers
        self.critic_head = nn.Linear(critic_out_dim, value_dim)
        self.value_dim = value_dim

        init_std_tensor = torch.as_tensor(init_std, dtype=torch.float32)
        if init_std_tensor.numel() == 1:
            init_std_tensor = init_std_tensor.repeat(1, self.action_dims.total_action_dim)
        elif init_std_tensor.numel() != self.action_dims.total_action_dim:
            init_std_tensor = init_std_tensor.mean().repeat(1, self.action_dims.total_action_dim)
        init_std_tensor = init_std_tensor.reshape(1, self.action_dims.total_action_dim)
        self.std = nn.Parameter(init_std_tensor)
        self.distribution = None
        Normal.set_default_validate_args = False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        entropy_sum = self.distribution.entropy().sum(dim=-1, keepdim=True)
        return entropy_sum.repeat(1, self.value_dim)

    def reset(self, dones=None):
        pass

    def update_distribution(self, observations, hist_encoding=False):
        mean = self.actor.flat_action(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        log_prob_sum = self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        return log_prob_sum.repeat(1, self.value_dim)

    def act_inference(self, observations, hist_encoding=False):
        return self.actor.flat_action(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic_head(self.critic_backbone(critic_observations))


def build_mlp(input_dim: int, hidden_dims: Iterable[int], activation: nn.Module):
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(clone_activation(activation))
        last_dim = hidden_dim
    if not layers:
        return nn.Identity(), input_dim
    return nn.Sequential(*layers), last_dim


def build_head(input_dim: int, output_dim: int, hidden_dims: Iterable[int], activation: nn.Module):
    body, body_out_dim = build_mlp(input_dim, hidden_dims, activation)
    if isinstance(body, nn.Identity):
        return nn.Linear(input_dim, output_dim)
    return nn.Sequential(body, nn.Linear(body_out_dim, output_dim))


def clone_activation(activation: nn.Module) -> nn.Module:
    return activation.__class__()


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "crelu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Invalid activation function: {act_name}")
