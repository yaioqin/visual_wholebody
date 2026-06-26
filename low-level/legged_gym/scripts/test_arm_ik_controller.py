from pathlib import Path
import sys

import torch


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.controllers.arm_ik_controller import ArmIKController, ArmIKControllerConfig


def main():
    torch.manual_seed(1)
    batch = 4
    horizon = 3
    num_arm_joints = 6

    lower = -torch.ones(num_arm_joints)
    upper = torch.ones(num_arm_joints)
    controller = ArmIKController(
        num_arm_joints=num_arm_joints,
        joint_lower_limits=lower,
        joint_upper_limits=upper,
        config=ArmIKControllerConfig(max_ee_pos_delta=0.05, max_ee_rot_delta=0.2),
    )

    current_ee_pose = torch.zeros(batch, 6)
    current_joint_pos = torch.zeros(batch, num_arm_joints)
    action_chunk = torch.randn(batch, horizon, 6) * 0.1
    arm_jacobian = torch.eye(6).repeat(batch, 1, 1)
    h = torch.tensor([1, 2, 3, 3])

    joint_target = controller.solve(
        current_ee_pose=current_ee_pose,
        action_chunk=action_chunk,
        h=h,
        current_joint_pos=current_joint_pos,
        arm_jacobian=arm_jacobian,
    )

    assert joint_target.shape == (batch, num_arm_joints)
    assert torch.isfinite(joint_target).all()
    assert torch.all(joint_target <= upper.to(joint_target.device) + 1.0e-6)
    assert torch.all(joint_target >= lower.to(joint_target.device) - 1.0e-6)
    print("arm IK controller smoke test passed")


if __name__ == "__main__":
    main()

