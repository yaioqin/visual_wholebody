# DWBC implementation status

## Goal

Implement the Deep Whole-Body Control algorithm from
`document/2210.10044v1.pdf` and the public
`~/lab/Deep-Whole-Body-Control` code on the B1 + Z1 low-level environment,
while preserving the existing training/play/evaluation interfaces.

## Completed

- Compared the paper, public `widowGo1` environment/config and the current
  B1+Z1 environment.
- Identified that the sibling `third_party/rsl_rl` already contains DWBC's
  dual action/value heads, Advantage Mixing and regularized history adaptation.
- Replaced the experiment wrapper config with DWBC observation, command,
  control, domain-randomization, reward and PPO parameters.
- Replaced the old gated/IK arm path with one direct 18-dimensional policy:
  12 leg and 6 arm joint-position offsets all pass through the same effort-mode
  joint-space PD controller. The configured two-step action delay is preserved.
- Ported the 72-dimensional B1/Z1 proprioceptive layout, 24 privileged values,
  ten-step history, spherical EE commands, command-conditioned termination and
  reference Perlin terrain dimensions.
- Kept separate locomotion and manipulation reward buffers. The only non-zero
  terms, their formulas, scales and the final `/100` normalization match the
  public implementation. In particular, the forward-velocity L1 term was
  restored without the old command-magnitude normalization.
- Configured the dual-head actor/critic, Advantage Mixing schedule
  `[1, 0, 3000]`, privileged encoder `[64, 20]`, history adaptation frequency
  20, and regularization schedule `[0, 0.1, 3000, 7000]`.
- Preserved the six-value environment step return, public EE-goal accessor,
  action/target buffers and coordination evaluator inputs used by `play.py`.
  Evaluation now reports the arm energy from the explicit PD effort torques.
- Updated training/evaluation documentation, default task, compatibility YAML
  and focused contract/smoke tests.

## Verification completed

- Python compilation and `git diff --check`.
- Configuration, reward-formula, observation-size and whole-body PD contract
  checks.
- Real Isaac Gym CPU simulation with 18-dimensional random actions, including
  non-zero arm torques and coordination-metric summarization.
- Two complete CPU training iterations: the first exercised online history
  adaptation and the second exercised the dual-return PPO/Advantage Mixing
  update.

## Remaining / operational notes

- No implementation work is known to remain.
- A full 5000-environment, 40000-iteration training run was not started; that is
  the intended production training job and requires a CUDA-capable machine.
- Earlier IK/gated-arm checkpoints have incompatible observation/action
  semantics and must stay on their original branch. The evaluation interfaces,
  rather than those old model semantics, are what remain compatible here.
- `document/2210.10044v1.pdf` was supplied by the user and was not modified.
