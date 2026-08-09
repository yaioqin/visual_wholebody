# Training a universal low-level policy

## Code structure
`legged_gym/envs` contains environment-related codes.

`legged_gym/scripts` contains train and test scripts.

## Train

The environment related code is `legged_gym/legged_gym/envs/manip_loco/manip_loco.py`, and the Aliengo+Z1 config entry is `legged_gym/legged_gym/envs/manip_loco/aliengo_z1_config.py`.

```bash
cd legged_gym/scripts
conda activate b1z1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
python train.py --headless --exptid aliengo_z1_source --proj_name aliengo-z1-low --task aliengo_z1 --sim_device cuda:0 --rl_device cuda:0 --observe_gait_commands
```
- `--debug` disables wandb and set a small number of envs for faster execution.
- `--headless` disables rendering, typically used when you train model.
- `--proj_name` the folder containing all your logs and wandb project name. `manip-loco` is default.
- `--observe_gait_commands` is for tracking specific gait commands and learning the trotting behavior.

Check `legged_gym/legged_gym/utils/helpers.py` for all command line args.

## Visualize
To inspect whether the Aliengo+Z1 URDF, meshes, and joint order load correctly:
```bash
cd legged_gym/scripts
conda activate b1z1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
python visualize_aliengo_z1_standby.py --sim_device cuda:0
```
By default this renders the static standby pose without advancing physics. Add `--simulate` if you also want to check the pose under PhysX and PD targets.


## Play
Only need to specify `--exptid`. The parser will automatically find corresponding runs.
```bash
cd legged_gym/scripts
python play.py --exptid SOME_YOUR_DESCRIPTION --task aliengo_z1 --proj_name aliengo-z1-low --checkpoint 64000 --observe_gait_commands
python play.py --exptid aliengo_z1_source --proj_name aliengo-z1-low --checkpoint 64000 --observe_gait_commands

```
Use `--sim_device cpu --rl_device cpu` in case not enough GPU memory.

For GPU video recording on a server without Xorg, including NVIDIA EGL setup,
camera-follow replay, rough terrain, and VBC current/target pose markers, see
[`VIDEO_REPLAY_EGL.md`](VIDEO_REPLAY_EGL.md). The reusable entrypoint is
`legged_gym/scripts/play_aliengo_z1_video.py`.

更多运行指令请看run_cmd.txt
## Suggestions
To choose a good low-level policy that can be further used for training the high-level policy, we suggest you deploy the low-level policy first, and see if it goes well before training a high-level policy.
