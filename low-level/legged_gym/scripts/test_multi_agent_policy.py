from pathlib import Path
import sys

import torch


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.models.multi_agent_policy import MultiAgentHighLevelPolicy


def main():
    torch.manual_seed(1)
    batch = 4
    obs_dim = 73
    horizon = 3
    policy = MultiAgentHighLevelPolicy(
        obs_dim=obs_dim,
        chunk_horizon=horizon,
        encoder_hidden_dims=(64,),
        backbone_hidden_dims=(128,),
    )
    obs = torch.randn(batch, obs_dim)
    out = policy(obs)

    assert out["base_command"].shape == (batch, 5)
    assert out["arm_action_chunk"].shape == (batch, horizon, 6)
    assert out["grasp_intent_logit"].shape == (batch, 1)
    assert out["h_logits"].shape == (batch, 3)
    assert policy.flat_action(obs).shape == (batch, 5 + horizon * 6 + 1 + 3)
    print("multi-agent policy forward smoke test passed")


if __name__ == "__main__":
    main()

