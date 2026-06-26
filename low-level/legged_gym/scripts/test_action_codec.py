from pathlib import Path
import sys

import torch


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.utils.action_codec import flatten_high_level_action, unflatten_high_level_action


def main():
    torch.manual_seed(1)
    batch = 4
    horizon = 3
    action_dict = {
        "base_command": torch.randn(batch, 5),
        "arm_action_chunk": torch.randn(batch, horizon, 6),
        "grasp_intent": torch.rand(batch, 1),
        "h_logits": torch.randn(batch, 3),
    }

    flat = flatten_high_level_action(action_dict)
    recovered = unflatten_high_level_action(flat, horizon)

    assert flat.shape == (batch, 5 + horizon * 6 + 1 + 3)
    for key in action_dict:
        assert recovered[key].shape == action_dict[key].shape, key
        assert torch.allclose(recovered[key], action_dict[key]), key
    print("action codec smoke test passed")


if __name__ == "__main__":
    main()

