"""Device assignment must happen before Isaac Gym creates its CUDA context."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from legged_gym.distributed import configure_distributed


def make_args(**overrides):
    values = dict(distributed=False, horovod=False, headless=True, record_video=False,
                  task="b2z1", seed=None, sim_device="cuda:1", rl_device="cuda:1")
    values.update(overrides)
    return SimpleNamespace(**values)


class DistributedLaunchTest(unittest.TestCase):
    def test_single_gpu_keeps_device_and_seed(self):
        args = make_args()
        with patch.dict(os.environ, {}, clear=True), patch("legged_gym.distributed.dist.init_process_group") as init:
            configure_distributed(args, default_seed=1)
        init.assert_not_called()
        self.assertEqual((args.sim_device, args.rl_device, args.seed), ("cuda:1", "cuda:1", None))
        self.assertFalse(args.distributed)

    def test_torchrun_assigns_each_gpu_and_distinct_seed(self):
        for rank in range(3):
            args = make_args()
            launch_env = dict(WORLD_SIZE="3", LOCAL_WORLD_SIZE="3", RANK=str(rank), LOCAL_RANK=str(rank))
            with patch.dict(os.environ, launch_env, clear=True), \
                    patch("legged_gym.distributed.dist.is_available", return_value=True), \
                    patch("legged_gym.distributed.dist.is_nccl_available", return_value=True), \
                    patch("legged_gym.distributed.torch.cuda.is_available", return_value=True), \
                    patch("legged_gym.distributed.torch.cuda.device_count", return_value=3), \
                    patch("legged_gym.distributed.torch.cuda.set_device") as set_device, \
                    patch("legged_gym.distributed.dist.init_process_group") as init:
                configure_distributed(args, default_seed=10)
            set_device.assert_called_once_with(rank)
            init.assert_called_once()
            self.assertEqual((args.sim_device, args.rl_device), ("cuda:{}".format(rank),) * 2)
            self.assertEqual((args.compute_device_id, args.sim_device_id), (rank, rank))
            self.assertEqual(args.seed, 10 + rank)
            self.assertEqual((args.rank, args.world_size), (rank, 3))

    def test_distributed_flag_requires_multiple_workers(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "torchrun"):
                configure_distributed(make_args(distributed=True), default_seed=1)


if __name__ == "__main__":
    unittest.main()
