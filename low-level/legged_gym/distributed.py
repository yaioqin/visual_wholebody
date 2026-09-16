"""Single-node torchrun setup, before constructing any Isaac Gym environments."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def configure_distributed(args, default_seed):
    """Bind one simulation and learner to each local GPU; num_envs is per rank."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if getattr(args, "horovod", False):
        raise ValueError("--horovod is not implemented; use torchrun with --distributed.")
    requested = getattr(args, "distributed", False)
    args.distributed = world_size > 1
    args.rank = 0
    args.world_size = world_size
    if not args.distributed:
        if requested:
            raise ValueError("Launch --distributed with torchrun --nproc_per_node=3.")
        return

    if not args.headless or args.record_video:
        raise ValueError("Distributed training requires --headless without --record_video.")
    if args.task == "b1z1_pick":
        raise ValueError("Distributed training is supported by the low-level PPO runner only.")
    local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    if int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) != world_size:
        raise ValueError("This launcher supports a single node only (--nnodes=1).")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("Distributed GPU training requires PyTorch with NCCL support.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("Each torchrun worker needs its own visible CUDA GPU.")

    torch.cuda.set_device(local_rank)
    args.local_rank = local_rank
    args.sim_device_type = "cuda"
    args.compute_device_id = local_rank
    args.sim_device_id = local_rank
    args.sim_device = "cuda:{}".format(local_rank)
    args.rl_device = args.sim_device
    args.use_gpu = True
    args.use_gpu_pipeline = True
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=30))

    # The environment seed differs across ranks; DistributedPPO broadcasts the
    # initial model so all learners nevertheless start from identical weights.
    seed = default_seed if args.seed is None else args.seed
    if seed == -1:
        seed_tensor = torch.randint(0, 10000, (1,), device=args.rl_device)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    args.seed = seed + args.rank
