# Deep Whole-Body Control for B1 + Z1

This branch implements the unified low-level controller from *Deep Whole-Body
Control: Learning a Unified Policy for Manipulation and Locomotion* on the
B1 + Z1 model.

The implementation follows `document/2210.10044v1.pdf` and the public
`~/lab/Deep-Whole-Body-Control` code:

- one policy emits 12 leg and 6 arm joint-position offsets;
- both action groups use joint-space PD at 50 Hz over a 200 Hz simulation;
- separate locomotion/manipulation returns feed two value and log-prob heads;
- Advantage Mixing linearly reaches full cross-task credit over 3000 updates;
- a privileged encoder and ten-step history encoder are jointly trained using
  the regularized online-adaptation objective;
- rewards and their scales match the public `widowGo1` configuration.

Robot-specific differences are limited to the B1/Z1 URDF, safe default joint
pose, one passive gripper joint, and the Z1 mount offset used by the spherical
end-effector command frame.

## Setup

Install Isaac Gym, this package, and the DWBC-compatible RSL-RL fork. In this
workspace the fork is at `../third_party/rsl_rl`:

```bash
pip install -e ../third_party/rsl_rl
pip install -e .
```

The standard PyPI RSL-RL package is not sufficient because DWBC requires two
reward/value streams, Advantage Mixing, and the history adaptation module.

## Train

```bash
cd legged_gym/scripts
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python train.py \
  --headless \
  --task b1z1 \
  --exptid dwbc_b1z1 \
  --proj_name b1z1-low \
  --sim_device cuda:0 \
  --rl_device cuda:0
```

Use `--debug --num_envs 128 --max_iterations 2` for a short smoke run. DWBC
always learns all 18 actions; no arm-action gate, IK flag, 5-D base command, or
gait-command flag is required.

## Evaluate

The existing `play.py` return signature and coordination evaluator remain
supported:

```bash
python play.py \
  --task b1z1 \
  --exptid dwbc_b1z1 \
  --proj_name b1z1-low \
  --checkpoint 40000 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --eval_coordination \
  --eval_steps 2000
```

`--flat_terrain` is supported for evaluation. Checkpoints from the earlier
IK/gated-arm experiments have different observation and action semantics and
must be evaluated on their original branch; the evaluation APIs themselves are
kept compatible here.

Implementation/verification progress is recorded in
`DWBC_IMPLEMENTATION_STATUS.md`.
