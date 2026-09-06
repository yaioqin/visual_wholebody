import tempfile
import unittest
import csv

import torch

from legged_gym.utils.eval_coverage import (
    CoverageMetrics,
    EE_COMMAND_NAMES,
    EvaluationCoverageScheduler,
    resolve_play_num_envs_value,
)
from legged_gym.utils.hard_case_mining import (
    GPUStateRingBuffer,
    HardCaseDetector,
    HardCaseMiner,
)
from legged_gym.utils.coordination_metrics import CoordinationMetrics


COMMAND_RANGES_3D = {
    "lin_vel_x": [-0.8, 0.8],
    "ang_vel_yaw": [-1.0, 1.0],
}
EE_RANGES = {
    "pos_l": [0.4, 0.95],
    "pos_p": [-1.2, 1.0],
    "pos_y": [-1.2, 1.2],
    "delta_orn_r": [-1.5, 1.5],
    "delta_orn_p": [-1.2, 1.6],
    "delta_orn_y": [-0.8, 0.8],
}


class EvaluationCoverageSchedulerTests(unittest.TestCase):
    def make_scheduler(self, **overrides):
        kwargs = dict(
            command_ranges=COMMAND_RANGES_3D,
            goal_ee_ranges=EE_RANGES,
            use_5d_base_command=False,
            num_samples=256,
            mode="joint",
            seed=7,
            ee_position_bins=9,
            ee_nominal=[0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        kwargs.update(overrides)
        return EvaluationCoverageScheduler(**kwargs)

    def test_samples_stay_in_ranges_and_include_boundaries(self):
        scheduler = self.make_scheduler(num_samples=1024)
        for dim, (name, command_index) in enumerate(scheduler.base_layout):
            values = scheduler.base_commands[:, command_index]
            self.assertGreaterEqual(
                float(values.min()), COMMAND_RANGES_3D[name][0] - 1.0e-6
            )
            self.assertLessEqual(
                float(values.max()), COMMAND_RANGES_3D[name][1] + 1.0e-6
            )
            self.assertTrue(torch.isclose(values, torch.tensor(COMMAND_RANGES_3D[name][0])).any())
            self.assertTrue(torch.isclose(values, torch.tensor(COMMAND_RANGES_3D[name][1])).any())
        for dim, name in enumerate(EE_COMMAND_NAMES):
            values = scheduler.ee_commands[:, dim]
            self.assertTrue(torch.isclose(values, torch.tensor(EE_RANGES[name][0])).any())
            self.assertTrue(torch.isclose(values, torch.tensor(EE_RANGES[name][1])).any())

    def test_sobol_schedule_is_deterministic_for_same_seed(self):
        first = self.make_scheduler(seed=11)
        second = self.make_scheduler(seed=11)
        self.assertTrue(torch.equal(first.base_commands, second.base_commands))
        self.assertTrue(torch.equal(first.ee_commands, second.ee_commands))

    def test_batches_assign_distinct_samples_and_exhaust_ids(self):
        scheduler = self.make_scheduler(num_samples=35, mode="base")
        ids = []
        first_commands = None
        while scheduler.remaining:
            batch = scheduler.next_batch(8)
            ids.extend(batch.sample_ids.tolist())
            if first_commands is None:
                first_commands = batch.base_commands
        self.assertEqual(ids, list(range(35)))
        self.assertGreater(torch.unique(first_commands, dim=0).shape[0], 1)

    def test_explicit_num_envs_is_never_replaced_or_capped(self):
        self.assertEqual(resolve_play_num_envs_value(6144, 32), 32)
        self.assertEqual(resolve_play_num_envs_value(6144, 512), 512)
        self.assertEqual(
            resolve_play_num_envs_value(6144, None, coverage_default=256), 256
        )

    def test_coverage_metrics_occupancy(self):
        scheduler = self.make_scheduler(num_samples=64, mode="base")
        metrics = CoverageMetrics(scheduler, num_bins=5)
        for sample_id in range(scheduler.num_samples):
            metrics.mark_executed(
                sample_id,
                success=True,
                **{
                    "achieved_ee/pos_l": 0.7,
                    "achieved_ee/pos_p": 0.0,
                    "achieved_ee/pos_y": 0.0,
                },
            )
        report = metrics.report()
        self.assertEqual(report["executed_samples"], 64)
        self.assertEqual(
            report["executed"]["base/lin_vel_x"]["coverage_percent"], 100.0
        )
        with tempfile.TemporaryDirectory() as out_dir:
            paths = metrics.save(out_dir)
            self.assertTrue(paths["json"].endswith("coverage_report.json"))


class HardCaseDetectorTests(unittest.TestCase):
    def test_zero_raw_threshold_disables_raw_criterion(self):
        detector = HardCaseDetector(
            1,
            torch.device("cpu"),
            eta_raw_threshold=0.0,
            eta_ratio_threshold=0.2,
            ee_request_threshold=0.02,
            ee_error_threshold=0.05,
            min_steps=1,
        )
        trigger, conditions = detector.update(
            eta_raw=torch.tensor([-1.0]),
            eta_ratio=torch.tensor([0.8]),
            ee_request_amplitude=torch.tensor([0.1]),
            ee_pos_error=torch.tensor([0.2]),
        )
        self.assertFalse(trigger.item())
        self.assertFalse(conditions["low_manip_raw"].item())

    def test_temporal_persistence_and_independent_envs(self):
        detector = HardCaseDetector(
            3,
            torch.device("cpu"),
            eta_raw_threshold=0.05,
            eta_ratio_threshold=0.2,
            ee_request_threshold=0.02,
            ee_error_threshold=0.05,
            min_steps=3,
        )
        eta_raw = torch.tensor([0.01, 0.01, 1.0])
        eta_ratio = torch.tensor([0.1, 0.1, 1.0])
        request = torch.tensor([0.1, 0.1, 0.1])
        error = torch.tensor([0.1, 0.01, 0.1])
        for step in range(2):
            trigger, _ = detector.update(
                eta_raw=eta_raw,
                eta_ratio=eta_ratio,
                ee_request_amplitude=request,
                ee_pos_error=error,
                sim_step=step,
            )
            self.assertFalse(trigger.any())
        trigger, _ = detector.update(
            eta_raw=eta_raw,
            eta_ratio=eta_ratio,
            ee_request_amplitude=request,
            ee_pos_error=error,
            sim_step=2,
        )
        self.assertEqual(trigger.tolist(), [True, False, False])

    def test_ring_buffer_returns_correct_pretrigger_sequence(self):
        ring = GPUStateRingBuffer(2, capacity=3, feature_dim=1, device="cpu")
        for step in range(5):
            ring.append(torch.tensor([[float(step)], [float(step + 10)]]))
        self.assertEqual(ring.ordered(0).flatten().tolist(), [2.0, 3.0, 4.0])
        self.assertEqual(ring.ordered(1).flatten().tolist(), [12.0, 13.0, 14.0])

    def test_miner_saves_triggered_case_and_snapshot(self):
        with tempfile.TemporaryDirectory() as out_dir:
            miner = HardCaseMiner(
                num_envs=1,
                device="cpu",
                dt=0.1,
                out_dir=out_dir,
                pre_seconds=0.2,
                post_seconds=0.1,
                min_steps=2,
                max_cases=2,
                eta_raw_threshold=0.05,
                eta_ratio_threshold=0.2,
                ee_request_threshold=0.02,
                ee_error_threshold=0.05,
                snapshot_interval_seconds=0.1,
            )
            snapshot = {"root": torch.zeros(1, 2)}
            active = torch.ones(1, dtype=torch.bool)
            miner.capture_snapshot(snapshot, 0, active, force=True)
            for step in range(3):
                features = {
                    "eta_raw": torch.tensor([0.01]),
                    "eta_ratio": torch.tensor([0.1]),
                    "ma2b_ee_request_amplitude_current": torch.tensor([0.1]),
                    "ee_pos_error": torch.tensor([0.2]),
                    "manip_log_eta": torch.tensor([-4.0]),
                    "manip_log_eta_ref": torch.tensor([-1.0]),
                    "directional_manipulability_xy": torch.tensor([0.1]),
                    "base_assist_alignment_raw": torch.tensor([-0.1]),
                }
                miner.update(
                    features=features,
                    snapshot_state=snapshot,
                    active_mask=active,
                    sim_step=step,
                    sample_ids=torch.tensor([5]),
                    batch_id=0,
                    episode_ids=torch.tensor([1]),
                    scheduled_base_commands=torch.zeros(1, 3),
                    scheduled_ee_commands=torch.zeros(1, 6),
                )
            outputs = miner.save(
                sample_records=[
                    {
                        "sample_id": 5,
                        "status": "executed",
                        "success": False,
                        "fall": False,
                        "collision": False,
                        "timeout": False,
                        "ee_rmse": 0.2,
                    }
                ]
            )
            self.assertEqual(outputs["detected_cases"], 1)
            self.assertEqual(outputs["saved_cases"], 1)
            self.assertTrue(outputs["hard_cases_csv"].endswith("hard_cases.csv"))

    def test_miner_keeps_global_top_k_not_first_k(self):
        with tempfile.TemporaryDirectory() as out_dir:
            miner = HardCaseMiner(
                num_envs=1,
                device="cpu",
                dt=0.1,
                out_dir=out_dir,
                pre_seconds=0.1,
                post_seconds=0.0,
                min_steps=1,
                max_cases=2,
                eta_raw_threshold=0.0,
                eta_ratio_threshold=0.2,
                ee_request_threshold=0.02,
                ee_error_threshold=0.05,
                snapshot_interval_seconds=0.1,
            )
            active = torch.ones(1, dtype=torch.bool)
            snapshot = {"root": torch.zeros(1, 2)}
            for sample_id, ratio, error in ((10, 0.19, 0.1), (20, 0.10, 0.2), (30, 0.01, 0.5)):
                miner.reset_envs(torch.tensor([0]))
                miner.capture_snapshot(snapshot, sample_id, active, force=True)
                features = {
                    "eta_raw": torch.tensor([0.01]),
                    "eta_ratio": torch.tensor([ratio]),
                    "ma2b_ee_request_amplitude_current": torch.tensor([0.1]),
                    "ee_pos_error": torch.tensor([error]),
                    "manip_log_eta": torch.tensor([-4.0]),
                    "manip_log_eta_ref": torch.tensor([-1.0]),
                    "directional_manipulability_xy": torch.tensor([0.1]),
                    "base_assist_alignment_raw": torch.tensor([-0.1]),
                }
                miner.update(
                    features=features,
                    snapshot_state=snapshot,
                    active_mask=active,
                    sim_step=sample_id,
                    sample_ids=torch.tensor([sample_id]),
                    batch_id=sample_id,
                    episode_ids=torch.tensor([sample_id]),
                    scheduled_base_commands=torch.zeros(1, 3),
                    scheduled_ee_commands=torch.zeros(1, 6),
                )
            outputs = miner.save()
            self.assertEqual(outputs["detected_cases"], 3)
            self.assertEqual(outputs["completed_candidates"], 3)
            self.assertEqual(outputs["saved_cases"], 2)
            with open(outputs["hard_cases_csv"], newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual({int(row["sample_id"]) for row in rows}, {20, 30})


class _CoordinationFakeEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.num_actions = 18
        self.num_dofs = 18
        self.dt = 0.1
        self.dof_names = [f"joint_{index}" for index in range(18)]
        self.base_lin_vel = torch.zeros(2, 3)
        self.base_ang_vel = torch.zeros(2, 3)
        self.commands = torch.zeros(2, 3)
        self.root_states = torch.zeros(2, 13)
        self.root_states[:, 0] = torch.tensor([10.0, 20.0])
        self.root_states[:, 2] = 0.5
        self.root_states[:, 6] = 1.0
        self.base_quat = self.root_states[:, 3:7]
        self.ee_pos = torch.tensor([[11.0, 0.0, 0.5], [22.0, 0.0, 0.5]])
        self.curr_ee_goal_cart_world = self.ee_pos.clone()
        self.ee_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
        self.ee_goal_quat = self.ee_quat.clone()
        self.reset_buf = torch.zeros(2, dtype=torch.bool)
        self.time_out_buf = torch.zeros(2, dtype=torch.bool)
        self.termination_buf = torch.zeros(2, dtype=torch.bool)
        self.torques = torch.zeros(2, 18)
        self.dof_vel = torch.zeros(2, 18)
        self.dof_pos = torch.zeros(2, 18)
        self.arm_pos_targets = torch.zeros(2, 6)


class CoordinationSampleTests(unittest.TestCase):
    def test_sample_mask_collision_split_and_base_local_workspace(self):
        env = _CoordinationFakeEnv()
        metrics = CoordinationMetrics(env, num_envs=2, device="cpu", dt=0.1)
        active = torch.tensor([True, False])
        metrics.begin_sample_batch(active)
        metrics.update(
            env,
            torch.zeros(2, 18),
            active_mask=active,
            fall_mask=torch.tensor([False, True]),
            collision_mask=torch.tensor([True, False]),
        )
        metrics.end_sample_batch(
            active,
            success_mask=torch.tensor([True, True]),
            fall_mask=torch.tensor([False, True]),
            collision_mask=torch.tensor([True, False]),
        )
        summary = metrics.summarize()
        self.assertEqual(summary["meta/total_samples"], 1.0)
        self.assertEqual(summary["meta/aggregation_unit"], "coverage_sample")
        self.assertEqual(summary["stability/fall_rate_step"], 0.0)
        self.assertEqual(summary["stability/collision_rate_step"], 1.0)
        self.assertEqual(summary["stability/sample_collision_rate"], 1.0)
        self.assertTrue(torch.allclose(metrics.success_points[0][0], torch.tensor([1.0, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
