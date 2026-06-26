from pathlib import Path
import sys

import torch


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))


def main():
    torch.manual_seed(1)
    batch = 4
    actions = torch.randn(batch, 18)

    leg_action = actions[:, :12]
    arm_delta_action = actions[:, 12:18]
    arm_action_chunk = arm_delta_action.view(batch, 1, 6)
    gated_actions = actions.clone()
    allow_arm_policy_action = False
    if not allow_arm_policy_action:
        gated_actions[:, 12:18] = 0.0

    assert leg_action.shape == (batch, 12)
    assert arm_delta_action.shape == (batch, 6)
    assert arm_action_chunk.shape == (batch, 1, 6)
    assert torch.all(gated_actions[:, 12:18] == 0.0)
    assert torch.allclose(gated_actions[:, :12], actions[:, :12])
    print("low-level action split smoke test passed")


if __name__ == "__main__":
    main()
