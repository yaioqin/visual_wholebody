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
    arm_action = actions[:, 12:18]
    action_scale = torch.tensor(
        [0.4, 0.45, 0.45] * 4 + [2.1, 0.6, 0.6, 0.0, 0.0, 0.0]
    )
    joint_offsets = actions * action_scale

    assert leg_action.shape == (batch, 12)
    assert arm_action.shape == (batch, 6)
    assert torch.allclose(joint_offsets[:, 12:15], arm_action[:, :3] * action_scale[12:15])
    assert torch.all(joint_offsets[:, 15:18] == 0.0)
    assert torch.any(joint_offsets[:, 12:15] != 0.0)
    print("DWBC whole-body action split smoke test passed")


if __name__ == "__main__":
    main()
