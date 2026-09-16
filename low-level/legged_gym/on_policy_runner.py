"""Project-specific extensions to the RSL-RL on-policy runner."""

from contextlib import redirect_stdout
from io import StringIO
import os
import statistics

import torch
import torch.distributed as dist
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner


class OnPolicyRunner(RslOnPolicyRunner):
    """Persist environment progress that affects scheduled training behavior."""

    ENV_GLOBAL_STEPS_KEY = "env_global_steps"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", distributed=False):
        self.distributed = distributed
        self.rank = dist.get_rank() if distributed else 0
        self.world_size = dist.get_world_size() if distributed else 1
        if distributed:
            if train_cfg["runner"]["algorithm_class_name"] != "PPO":
                raise ValueError("Distributed training requires PPO.")
            if train_cfg["runner"]["policy_class_name"] != "ActorCritic":
                raise ValueError("Distributed training currently supports the feed-forward ActorCritic.")
            if log_dir is None:
                raise ValueError("Distributed training requires a shared log directory.")
        super().__init__(env, train_cfg, log_dir, device)
        if distributed:
            from legged_gym.distributed_ppo import DistributedPPO

            # Keep the upstream runner's environment setup and rollout storage.
            # Only replace the learner; the single-GPU path remains upstream PPO.
            local_alg = self.alg
            self.alg = DistributedPPO(local_alg.actor_critic, device=device, **self.alg_cfg)
            self.alg.storage = local_alg.storage
            self.alg.set_arm_default_coeffs(
                local_alg.default_arm_p_gains,
                local_alg.default_arm_d_gains,
                local_alg.default_arm_dof_pos,
            )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._progress_bar = None
        if self.rank == 0:
            start_iteration = self.current_learning_iteration
            description = "Training"
            if self.distributed:
                description += " ({} GPUs)".format(self.world_size)
            self._progress_bar = tqdm(
                total=start_iteration + num_learning_iterations,
                initial=start_iteration,
                desc=description,
                unit="iter",
                dynamic_ncols=True,
                mininterval=1.0,
                smoothing=0.1,
                bar_format=("{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed} elapsed, ETA {remaining}, {rate_fmt}{postfix}]"),
            )
        try:
            if self.rank == 0 and self.log_dir is not None:
                from torch.utils.tensorboard import SummaryWriter

                tensorboard_dir = os.path.join(self.log_dir, "tensorboard")
                self.writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=10)
                print("TensorBoard logs: {}".format(tensorboard_dir))
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            if self._progress_bar is not None:
                self._progress_bar.close()
                self._progress_bar = None
            if self.writer is not None:
                self.writer.close()
                self.writer = None

    def _update_progress(self, iteration, collection_time, learn_time):
        progress = getattr(self, "_progress_bar", None)
        if progress is None:
            return
        progress.set_postfix(
            rollout="{:.2f}s".format(collection_time),
            update="{:.2f}s".format(learn_time),
            refresh=False,
        )
        # tqdm measures wall time between updates, including intervening
        # logging/checkpoint work, instead of estimating ETA from GPU timings.
        progress.update(max(0, iteration + 1 - progress.n))

    def log(self, locs, width=80, pad=35):
        if not self.distributed:
            # Preserve every upstream WandB metric without printing its long
            # per-iteration report over the progress bar.
            with redirect_stdout(StringIO()):
                super().log(locs, width, pad)
            if self.writer is not None:
                self._log_single_gpu_tensorboard(locs)
            self._update_progress(locs["it"], locs["collection_time"], locs["learn_time"])
            return

        import statistics
        import wandb

        # Every worker enters these collectives, but only rank zero writes logs.
        loss_names = ["mean_value_loss", "mean_surrogate_loss", "mean_arm_torques_loss",
                      "mean_hist_latent_loss", "mean_priv_reg_loss"]
        losses = torch.tensor([locs[key] for key in loss_names], device=self.device)
        dist.all_reduce(losses)
        losses /= self.world_size
        durations = torch.tensor([
            locs["collection_time"], locs["learn_time"],
            locs["collection_time"] + locs["learn_time"],
        ], device=self.device)
        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
        if self.rank != 0:
            return

        collection_time, learn_time, iteration_time = durations.tolist()
        global_envs = self.env.num_envs * self.world_size
        self.tot_timesteps += self.num_steps_per_env * global_envs
        self.tot_time += iteration_time
        fps = self.num_steps_per_env * global_envs / max(iteration_time, 1e-9)
        metric_names = ["value_function", "surrogate", "arm_torques_loss",
                        "hist_latent_loss", "priv_reg_loss"]
        metrics = {"Loss/" + name: value for name, value in zip(metric_names, losses.tolist())}
        metrics.update({
            "Loss/learning_rate": self.alg.learning_rate,
            "Loss/priv_ref_lambda": locs["priv_reg_coef"],
            "Loss/value_mixing_ratio": locs["value_mixing_ratio"],
            "Loss/torque_supervision_weight": locs["torque_supervision_weight"],
            "Perf/total_fps": fps,
            "Perf/collection time": collection_time,
            "Perf/learning_time": learn_time,
            "Perf/iteration_time": iteration_time,
            "Perf/total_timesteps": self.tot_timesteps,
            "Distributed/world_size": self.world_size,
            "Distributed/global_envs": global_envs,
            "Policy/leg_mean_noise_std": self.alg.actor_critic.std[:, :12].mean().item(),
            "Policy/arm_mean_noise_std": self.alg.actor_critic.std[:, 12:].mean().item(),
        })
        # Episode metrics describe rank zero's environments, with an explicit
        # prefix. Losses and throughput above cover all ranks.
        for key, buffer_name in [("mean_reward", "rewbuffer"), ("mean_arm_reward", "armrewbuffer"),
                                 ("mean_episode_length", "lenbuffer"), ("dones", "donebuffer")]:
            if locs[buffer_name]:
                metrics["Rank0/Train/" + key] = statistics.mean(locs[buffer_name])
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                values = [torch.as_tensor(info[key], device=self.device).reshape(-1)
                          for info in locs["ep_infos"] if key in info]
                metrics["Rank0/Episode/" + key] = torch.cat(values).float().mean().item()
        wandb.log(metrics, step=locs["it"])
        self._write_tensorboard(metrics, locs["it"])
        self._update_progress(locs["it"], collection_time, learn_time)

    def _log_single_gpu_tensorboard(self, locs):
        loss_keys = {
            "value_function": "mean_value_loss", "surrogate": "mean_surrogate_loss",
            "arm_torques_loss": "mean_arm_torques_loss", "hist_latent_loss": "mean_hist_latent_loss",
            "priv_reg_loss": "mean_priv_reg_loss", "priv_ref_lambda": "priv_reg_coef",
            "value_mixing_ratio": "value_mixing_ratio",
            "torque_supervision_weight": "torque_supervision_weight",
        }
        metrics = {"Loss/" + name: locs[key] for name, key in loss_keys.items()}
        iteration_time = locs["collection_time"] + locs["learn_time"]
        metrics.update({
            "Loss/learning_rate": self.alg.learning_rate,
            "Perf/total_fps": self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-9),
            "Perf/collection time": locs["collection_time"],
            "Perf/learning_time": locs["learn_time"],
            "Perf/iteration_time": iteration_time,
            "Perf/total_timesteps": self.tot_timesteps,
            "Policy/leg_mean_noise_std": self.alg.actor_critic.std[:, :12].mean().item(),
            "Policy/arm_mean_noise_std": self.alg.actor_critic.std[:, 12:].mean().item(),
        })
        for name, buffer in [("mean_reward", "rewbuffer"), ("mean_arm_reward", "armrewbuffer"),
                             ("mean_episode_length", "lenbuffer"), ("dones", "donebuffer")]:
            if locs[buffer]:
                metrics["Train/" + name] = statistics.mean(locs[buffer])
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                values = [torch.as_tensor(info[key], device=self.device).reshape(-1)
                          for info in locs["ep_infos"] if key in info]
                prefix = "Episode_rew/" if "rew" in key else (
                    "Episode_metric/" if "metric" in key else "Episode/")
                metrics[prefix + key] = torch.cat(values).float().mean().item()
        self._write_tensorboard(metrics, locs["it"])

    def _write_tensorboard(self, metrics, iteration):
        if self.writer is None or self.rank != 0:
            return
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, global_step=iteration)
        self.writer.add_histogram("Policy/noise_std_dist",
                                  self.alg.actor_critic.std.detach().cpu(), global_step=iteration)

    def save(self, path, it, infos=None):
        if getattr(self, "rank", 0) != 0:
            return
        checkpoint = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": it,
            "infos": infos,
        }
        if hasattr(self.env, "global_steps"):
            checkpoint[self.ENV_GLOBAL_STEPS_KEY] = int(self.env.global_steps)
        if hasattr(self.alg, "hist_encoder_optimizer"):
            checkpoint["hist_encoder_optimizer_state_dict"] = self.alg.hist_encoder_optimizer.state_dict()
        if hasattr(self.alg, "counter"):
            checkpoint["algorithm_counter"] = self.alg.counter
        if hasattr(self.alg, "learning_rate"):
            checkpoint["learning_rate"] = self.alg.learning_rate
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if hasattr(self.alg, "hist_encoder_optimizer") and "hist_encoder_optimizer_state_dict" in checkpoint:
                self.alg.hist_encoder_optimizer.load_state_dict(checkpoint["hist_encoder_optimizer_state_dict"])
            if hasattr(self.alg, "learning_rate"):
                self.alg.learning_rate = checkpoint.get("learning_rate", self.alg.optimizer.param_groups[0]["lr"])
        self.current_learning_iteration = checkpoint["iter"]
        if hasattr(self.alg, "counter"):
            self.alg.counter = checkpoint.get("algorithm_counter", self.current_learning_iteration)

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
