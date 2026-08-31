from pathlib import Path
import math
import sys
import tempfile

import torch


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.utils.coordination_metrics import (  # noqa: E402
    CoordinationMetrics,
    _cart_to_sphere,
    quat_to_included_angles,
)


class DummyEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_envs = 5
        self.num_actions = 18
        self.num_dofs = 18
        self.dt = 0.02

        self.commands = torch.tensor([[1.0, 0.2, 0.3]]).repeat(self.num_envs, 1)
        self.base_lin_vel = torch.tensor([[0.9, 0.1, 0.0]]).repeat(self.num_envs, 1)
        self.base_ang_vel = torch.tensor([[0.0, 0.0, 0.1]]).repeat(self.num_envs, 1)

        self.root_states = torch.zeros(self.num_envs, 13)
        self.root_states[:, 2] = 0.5
        self.root_states[:, 6] = 1.0
        self.base_yaw_quat = self.root_states[:, 3:7].clone()

        # The first four actual positions form a tetrahedron of volume
        # 1/6000 m^3. The far fifth point is colliding and must be excluded.
        actual_ee_pos = torch.tensor(
            [
                [0.3, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.3, 0.1, 0.0],
                [0.3, 0.0, 0.1],
                [5.0, 5.0, 5.0],
            ]
        )
        # Targets are deliberately one metre away: workspace must use actual
        # reach positions and must not depend on tracking error.
        self.curr_ee_goal_cart = actual_ee_pos + torch.tensor([1.0, 0.0, 0.0])
        self.curr_ee_goal_cart_world = self.curr_ee_goal_cart.clone()
        self.curr_ee_goal_sphere = _cart_to_sphere(self.curr_ee_goal_cart)
        self.ee_pos = actual_ee_pos

        yaw = 0.2
        self.ee_orn = torch.tensor(
            [[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]]
        ).repeat(self.num_envs, 1)
        self.ee_goal_orn_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(
            self.num_envs, 1
        )

        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.termination_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.collision_buf = torch.tensor([False, False, False, False, True])
        self.torques = torch.zeros(self.num_envs, self.num_dofs)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dofs)
        self.dof_pos = torch.zeros(self.num_envs, self.num_dofs)

    def _get_ee_goal_spherical_center(self):
        return torch.zeros(self.num_envs, 3)


def main():
    yaw = 0.2
    yaw_quat = torch.tensor([[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]])
    included = quat_to_included_angles(yaw_quat)
    assert torch.allclose(included, torch.tensor([[yaw, 0.0, 0.0]]), atol=1.0e-6)

    env = DummyEnv()
    metrics = CoordinationMetrics(
        env,
        # Both position and orientation fail the 6D success criterion, while
        # workspace must still accept the four collision-free actual points.
        success_ori_thr=0.1,
        fall_height_thr=0.26,
    )
    metrics.update(env, torch.zeros(env.num_envs, env.num_actions))
    summary = metrics.summarize()

    assert abs(summary["roboduet/velocity/vx_mae_m_s"] - 0.1) < 1.0e-6
    assert abs(summary["roboduet/velocity/vy_mae_m_s"] - 0.1) < 1.0e-6
    assert abs(summary["roboduet/velocity/omega_z_mae_rad_s"] - 0.2) < 1.0e-6
    assert abs(summary["roboduet/orientation/alpha_mae_rad"] - yaw) < 1.0e-6
    assert abs(summary["roboduet/orientation/beta_mae_rad"]) < 1.0e-6
    assert abs(summary["roboduet/orientation/gamma_mae_rad"]) < 1.0e-6
    assert abs(summary["roboduet/orientation/zeta_geodesic_mae_rad"] - yaw) < 1.0e-6
    assert summary["ee/success_rate_step"] == 0.0
    assert summary["roboduet/meta/mode"] == "move"
    assert summary["roboduet/meta/workspace_criterion"] == "actual_ee_alive_collision_free"
    assert summary["roboduet/survival_rate_percent"] == 100.0
    assert summary["roboduet/workspace_valid_points"] == 4
    assert summary["roboduet/workspace_valid_point_rate"] == 0.8
    assert abs(summary["roboduet/workspace_m3"] - 1.0 / 6000.0) < 1.0e-9

    keys = list(summary)
    assert keys.index("roboduet/meta/mode") > keys.index("metrics/warnings")
    with tempfile.TemporaryDirectory() as out_dir:
        paths = metrics.save(out_dir)
        assert Path(paths["json"]).is_file()
        assert Path(paths["csv"]).is_file()

    print("RoboDuet Table III metrics smoke test passed")


if __name__ == "__main__":
    main()
