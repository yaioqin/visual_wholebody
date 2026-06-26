# Training a universal low-level policy

## Code structure
`legged_gym/envs` contains environment-related codes.

`legged_gym/scripts` contains train and test scripts.

## Train

The environment related code is `legged_gym/legged_gym/envs/manip_loco/manip_loco.py`, and the related config for b1z1 hardware is in `legged_gym/legged_gym/envs/b1z1/b1z1_config.py`.

```bash
cd legged_gym/scripts
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python train.py --headless --exptid MARL_b1_z1 --proj_name b1z1-low --task b1z1 --sim_device cuda:1 --rl_device cuda:1 --observe_gait_commands
```
- `--debug` disables wandb and set a small number of envs for faster execution.
- `--headless` disables rendering, typically used when you train model.
- `--proj_name` the folder containing all your logs and wandb project name. `manip-loco` is default.
- `--observe_gait_commands` is for tracking specific gait commands and learning the trotting behavior.
- `--allow_arm_policy_action` allows the policy's last 6 action dimensions to drive the arm delta IK controller. Without it, the environment zeros those 6 dimensions and PPO updates the leg surrogate only.

To train a whole-body policy with policy-controlled arm motion, start a new run with the arm gate enabled:
```bash
python train.py --headless --exptid MARL_b1_z1_5D_wholebody --proj_name b1z1-low --task b1z1 --sim_device cuda:1 --rl_device cuda:1 --observe_gait_commands --allow_arm_policy_action
```

Check `legged_gym/legged_gym/utils/helpers.py` for all command line args.

## Play
Only need to specify `--exptid`. The parser will automatically find corresponding runs.
```bash
cd legged_gym/scripts
python play.py --exptid MARL_b1_z1_5D_base_command --task b1z1 --proj_name b1z1-low --checkpoint 45000 --sim_device cuda:1 --rl_device cuda:1 --observe_gait_commands
```
Use `--sim_device cpu --rl_device cpu` in case not enough GPU memory.
Use `--allow_arm_policy_action` for checkpoints that were trained with that flag. Base-only checkpoints trained without the flag did not learn meaningful arm actions, so enabling the flag only at play time is not expected to produce a useful arm motion.

## Suggestions
To choose a good low-level policy that can be further used for training the high-level policy, we suggest you deploy the low-level policy first, and see if it goes well before training a high-level policy.
