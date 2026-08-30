from pathlib import Path
import sys


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.utils.training import remaining_learning_iterations


def main():
    assert remaining_learning_iterations(450000, 0) == 450000
    assert remaining_learning_iterations(450000, 120000) == 330000
    assert remaining_learning_iterations(450000, 450000) == 0
    assert remaining_learning_iterations(450000, 460000) == 0
    print("training resume iteration test passed")


if __name__ == "__main__":
    main()
