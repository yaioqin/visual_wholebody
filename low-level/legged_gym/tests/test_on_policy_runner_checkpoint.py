import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from legged_gym.on_policy_runner import OnPolicyRunner


class CurriculumEnv:
    def __init__(self, global_steps=0):
        self.global_steps = global_steps
        self.loaded_at_steps = []

    def on_checkpoint_loaded(self):
        self.loaded_at_steps.append(self.global_steps)


def make_runner(env):
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    actor_critic = torch.nn.Linear(2, 1)
    runner.alg = SimpleNamespace(
        actor_critic=actor_critic,
        optimizer=torch.optim.Adam(actor_critic.parameters()),
    )
    runner.env = env
    runner.device = "cpu"
    runner.num_steps_per_env = 24
    runner.current_learning_iteration = 0
    return runner


class OnPolicyRunnerCheckpointTest(unittest.TestCase):
    def test_global_steps_round_trip_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            source = make_runner(CurriculumEnv(global_steps=123_457))
            source.save(path, it=17, infos={"tag": "round-trip"})

            payload = torch.load(path, map_location="cpu")
            self.assertEqual(payload["env_global_steps"], 123_457)

            target = make_runner(CurriculumEnv())
            infos = target.load(path)

            self.assertEqual(target.current_learning_iteration, 17)
            self.assertEqual(target.env.global_steps, 123_457)
            self.assertEqual(target.env.loaded_at_steps, [123_457])
            self.assertEqual(infos, {"tag": "round-trip"})

    def test_legacy_checkpoint_recovers_curriculum_phase_from_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_model.pt"
            source = make_runner(CurriculumEnv())
            torch.save(
                {
                    "model_state_dict": source.alg.actor_critic.state_dict(),
                    "optimizer_state_dict": source.alg.optimizer.state_dict(),
                    "iter": 73,
                    "infos": None,
                },
                path,
            )

            target = make_runner(CurriculumEnv())
            target.load(path)

            self.assertEqual(target.env.global_steps, 73 * 24)
            self.assertEqual(target.env.loaded_at_steps, [73 * 24])

    def test_environment_without_global_steps_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generic_model.pt"
            source = make_runner(SimpleNamespace())
            source.save(path, it=5)

            payload = torch.load(path, map_location="cpu")
            self.assertNotIn("env_global_steps", payload)

            target = make_runner(SimpleNamespace())
            target.load(path)
            self.assertFalse(hasattr(target.env, "global_steps"))


if __name__ == "__main__":
    unittest.main()
