from pathlib import Path
import sys

import torch


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.models.message_passing import MultiAgentMessageBuilder


def main():
    torch.manual_seed(1)
    batch = 4
    horizon = 1
    arm_action_chunk = torch.randn(batch, horizon, 6)
    arm_jacobian = torch.randn(batch, 6, 6)
    object_pos_wrist = torch.randn(batch, 3)

    m_a2b = MultiAgentMessageBuilder.compute_m_a2b(arm_action_chunk, arm_jacobian)
    m_a2g = MultiAgentMessageBuilder.compute_m_a2g(arm_action_chunk, object_pos_wrist)

    assert m_a2b.shape == (batch, 5), m_a2b.shape
    assert m_a2g.shape == (batch, 2), m_a2g.shape
    assert torch.isfinite(m_a2b).all()
    assert torch.isfinite(m_a2g).all()
    print("message passing smoke test passed")


if __name__ == "__main__":
    main()
