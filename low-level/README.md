# Training a universal low-level policy

## Code structure
`legged_gym/envs` contains environment-related codes.

`legged_gym/scripts` contains train and test scripts.

## Train

The environment related code is `legged_gym/legged_gym/envs/manip_loco/manip_loco.py`, and the related config for b1z1 hardware is in `legged_gym/legged_gym/envs/b1z1/b1z1_config.py`.

```bash
conda activate 
cd legged_gym/scripts
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python train.py --headless --exptid SOME_YOUR_DESCRIPTION --proj_name b1z1-low --task b1z1 --sim_device cuda:1 --rl_device cuda:1 --observe_gait_commands
```
- `--debug` disables wandb and set a small number of envs for faster execution.
- `--disable_wandb` disables W&B logging without changing the environment count or other training settings. Local checkpoints and TensorBoard metrics are still saved.
- `--headless` disables rendering, typically used when you train model.
- `--proj_name` the folder containing all your logs and wandb project name. `manip-loco` is default.
- `--observe_gait_commands` is for tracking specific gait commands and learning the trotting behavior.

Check `legged_gym/legged_gym/utils/helpers.py` for all command line args.

### Local TensorBoard logs

Training automatically writes TensorBoard events to
`logs/<proj_name>/<exptid>/tensorboard`, including with `--disable_wandb`.
It records losses, learning rate, rewards, episode metrics, policy noise and
throughput at each training iteration. Distributed training writes only on rank
zero; losses and throughput use the reduced values, while episode metrics use
the `Rank0/` prefix. Events flush every 10 seconds and the writer closes when
training finishes or raises an exception. Resumed training keeps checkpoint
iteration numbers on the horizontal axis. Use a new `--exptid` for a fresh run
to avoid mixing curves with an earlier run.

In your training environment, install the dependency if needed and start the UI
from this `low-level` directory:

```bash
python -m pip install tensorboard
tensorboard --logdir logs/b2z1-low --port 6006
```

Open `http://localhost:6006` on that machine. For a remote training server,
forward the port with `ssh -L 6006:localhost:6006 <user>@<server>`, then open
the same URL locally. Existing runs made with W&B disabled have no historical
metric events to recover; TensorBoard records iterations after this change.

### Select a config file

Use `--config <path.py>` with `train.py` or `play.py` to load both the environment
and PPO configuration from that file. Paths are absolute or relative to the
current working directory. For `--task b2z1`, the file must provide
`B2Z1RoughCfg` and `B2Z1RoughCfgPPO`. Without `--config`, the registered task
defaults are used. Command-line overrides such as `--num_envs` and
`--max_iterations` still apply. Training prints the selected path and saves the
selected source file to W&B.

Single-GPU resume on GPU 2, from this `low-level` directory:

```bash
python legged_gym/scripts/train.py \
  --headless \
  --task b2z1 \
  --config legged_gym/envs/manip_loco/b2z1_config_LU.py \
  --proj_name b2z1-low \
  --exptid b2z1_Lu \
  --disable_wandb \
  --observe_gait_commands \
  --sim_device cuda:2 --rl_device cuda:2
```

This resumes `logs/b2z1-low/b2z1_Lu/model_37000.pt`. Add
`--resumeid <old_exptid>` to load the checkpoint from another run.

### Three-GPU synchronous training

From this `low-level` directory, in the same environment used for single-GPU
training, launch one simulator and PPO learner per GPU with
[PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
  legged_gym/scripts/train.py \
  --headless \
  --task b2z1 \
  --config legged_gym/envs/manip_loco/b2z1_config_LU_gait_fix.py \
  --proj_name b2z1-low \
  --exptid b2z1_Lu \
  --distributed \
  --num_envs 2048 \
  --disable_wandb \
  --observe_gait_commands 
  \
  --resume \
  --checkpoint 37000 \
  --max_iterations 8000

```

`--num_envs` is **per process**: `3 × 2048 = 6144` environments, matching the
default single-GPU B2Z1 batch. With 24 rollout steps, both configurations collect
147456 transitions per iteration. Each worker's simulation and learner are bound
to its `LOCAL_RANK`; device arguments are overridden in distributed mode. Change
`CUDA_VISIBLE_DEVICES` to select the three physical GPUs. `--horovod` is not used.
`python -m torch.distributed.run` is equivalent to `torchrun` and also works when
the environment does not install the `torchrun` executable.

Every GPU collects its own trajectories and computes gradients. PPO and the
separate DAgger history encoder update both average gradients across all GPUs
**before gradient clipping and each optimizer step**. Model initialization,
advantage statistics and adaptive KL statistics are also synchronized. This is
synchronous data parallel training with a complete model replica on each GPU.
The implementation supports the low-level feed-forward PPO runner on one node.

Only rank zero creates a WandB run and writes checkpoints. Checkpoints retain the
single-GPU model format and also save the history encoder optimizer and algorithm
schedule counter; existing checkpoints remain loadable. Use the usual
`--resumeid <old_exptid> --checkpoint <iteration>` arguments to resume. Environment
curriculum steps are not multiplied by the GPU count.

Compare warmed-up `Perf/iteration_time` and time to the same policy quality with
the single-GPU baseline. This metric covers rollout and update; include logging
and checkpoint overhead when comparing sustained wall-clock training time.
`Perf/total_fps` counts all environments using the slowest
worker's iteration time; losses are averaged across workers. Episode/reward
metrics under `Rank0/` describe rank zero's environments. Communication, CPU
simulation work and per-GPU terrain allocation limit scaling, so three GPUs do
not guarantee three times the speed. Keep the original learning rate, rollout
length and iteration count for the first comparison. Do not divide iterations
by three: curriculum and PPO/DAgger schedules still advance per iteration.
`--debug` forces 128 environments **per process**, so it is only a smoke test,
not a comparable speed benchmark.

Both single-GPU and distributed runs display a progress bar with completed/total
iterations, percentage, elapsed time, **ETA**, iteration rate, and rollout/update
times. Only rank zero displays it in distributed mode. ETA is estimated from
observed iteration wall-clock times and becomes available after training starts;
it includes intervening logging/checkpoint work but excludes simulator startup.
On resume the bar starts at the saved iteration and its total includes the
additional iterations requested by `--max_iterations`, matching the runner's
existing resume behavior.

CPU tests exercise three real Gloo workers against a combined-batch reference
without Isaac Gym or CUDA:

```bash
python -m unittest legged_gym.tests.test_distributed_training \
  legged_gym.tests.test_distributed_launch \
  legged_gym.tests.test_on_policy_runner_checkpoint \
  legged_gym.tests.test_on_policy_runner_progress -v
```

## Play
Only need to specify `--exptid`. The parser will automatically find corresponding runs.
```bash
cd legged_gym/scripts
python play.py --exptid SOME_YOUR_DESCRIPTION --task b1z1 --proj_name b1z1-low --checkpoint 64000 --observe_gait_commands

python play.py --exptid b2z1_gait_fix_01_3gpu_newurdf  --task b2z1 --proj_name b2z1-low --checkpoint 37000 --observe_gait_commands --record_video

python play.py --exptid b2z1_Lu  --task b2z1 --proj_name b2z1-low --checkpoint 2000 --observe_gait_commands --record_video --sim_device cpu --rl_device cpu --config ../../legged_gym/envs/manip_loco/b2z1_config_LU.py 

```
Use `--sim_device cpu --rl_device cpu` in case not enough GPU memory.

`--record_video` also works with `--headless` (the default). Recording keeps the
graphics device enabled without opening a viewer and follows the robot with a
camera sensor. A working NVIDIA graphics/Vulkan driver is required on the server.
Videos are saved to `low-level/logs/videos/<run_name>/`, regardless of the working
directory; the full output filename is printed when recording starts.
Recording lasts **60 seconds** by default, across episode resets. Use
`--video_duration 60` to set the duration explicitly (or another value in seconds).
The video includes a yellow current end-effector target, a red planned target
trajectory, and a cyan trajectory endpoint. These are projected into the image
using the recording camera matrices and remain visible through occluding objects.
The trajectory uses the controller's spherical interpolation, not a straight
line between endpoints. The overlay also works without a viewer window.

## Suggestions
To choose a good low-level policy that can be further used for training the high-level policy, we suggest you deploy the low-level policy first, and see if it goes well before training a high-level policy.
