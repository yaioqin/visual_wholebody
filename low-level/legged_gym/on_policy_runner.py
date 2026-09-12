"""Project-specific extensions to the RSL-RL on-policy runner."""

import torch

from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner


class OnPolicyRunner(RslOnPolicyRunner):
    """Persist environment progress that affects scheduled training behavior."""

    ENV_GLOBAL_STEPS_KEY = "env_global_steps"

    def save(self, path, it, infos=None):
        checkpoint = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": it,
            "infos": infos,
        }
        if hasattr(self.env, "global_steps"):
            checkpoint[self.ENV_GLOBAL_STEPS_KEY] = int(self.env.global_steps)
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_learning_iteration = checkpoint["iter"]

        if hasattr(self.env, "global_steps"):
            # Checkpoints created before global_steps was persisted still need
            # to resume in the correct curriculum phase.  One runner iteration
            # contains num_steps_per_env vector-environment control steps.
            fallback_steps = max(0, int(self.current_learning_iteration)) * int(
                self.num_steps_per_env
            )
            self.env.global_steps = int(
                checkpoint.get(self.ENV_GLOBAL_STEPS_KEY, fallback_steps)
            )

            # The runner constructor resets the environment before load(), so
            # its current commands and observations were generated at step 0.
            # Give curriculum-aware environments a chance to refresh them.
            on_checkpoint_loaded = getattr(self.env, "on_checkpoint_loaded", None)
            if callable(on_checkpoint_loaded):
                on_checkpoint_loaded()

        return checkpoint["infos"]
