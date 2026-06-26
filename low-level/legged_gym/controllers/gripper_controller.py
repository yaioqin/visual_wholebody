from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GripperRuleConfig:
    rho_threshold: float = 0.75
    terminal_motion_threshold: float = 0.02


class RuleBasedGripperController:
    def __init__(self, config: GripperRuleConfig = GripperRuleConfig()):
        self.config = config

    def __call__(self, grasp_intent: torch.Tensor, m_a2g: torch.Tensor) -> torch.Tensor:
        if grasp_intent.dim() == 1:
            grasp_intent = grasp_intent.unsqueeze(-1)
        if m_a2g.shape[-1] != 2:
            raise ValueError("m_a2g must have shape [B, 2]")

        rho_align = m_a2g[:, 0:1]
        terminal_ee_motion = m_a2g[:, 1:2]
        close = (
            (grasp_intent > 0.5)
            & (rho_align > self.config.rho_threshold)
            & (terminal_ee_motion < self.config.terminal_motion_threshold)
        )
        return close.to(dtype=grasp_intent.dtype)


class GripperPolicyMLP(nn.Module):
    def __init__(self, obs_dim=3, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GripperInteractionAgent(nn.Module):
    def __init__(self, use_learnable_gripper=False, hidden_dim=64, rule_config=GripperRuleConfig()):
        super().__init__()
        self.use_learnable_gripper = use_learnable_gripper
        self.rule_controller = RuleBasedGripperController(rule_config)
        self.policy = GripperPolicyMLP(hidden_dim=hidden_dim) if use_learnable_gripper else None

    def forward(self, grasp_intent: torch.Tensor, m_a2g: torch.Tensor):
        if self.use_learnable_gripper:
            obs = torch.cat([grasp_intent, m_a2g], dim=-1)
            return self.policy(obs)
        return self.rule_controller(grasp_intent, m_a2g)

