# Multi-Agent Low-Level Training Plan

This note documents the staged multi-agent structure added alongside the
existing low-level training flow. The default B1Z1 low-level training path stays
on the original `ActorCritic` unless the config is explicitly switched.

## Code Map

- Existing low-level actor-critic: `third_party/rsl_rl/rsl_rl/modules/actor_critic.py`
- Existing PPO runner: `third_party/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Existing PPO update: `third_party/rsl_rl/rsl_rl/algorithms/ppo.py`
- Existing low-level env: `low-level/legged_gym/envs/manip_loco/manip_loco.py`
- Existing B1Z1 config: `low-level/legged_gym/envs/manip_loco/b1z1_config.py`
- Existing reward container: `low-level/legged_gym/envs/rewards/maniploco_rewards.py`
- New message passing: `low-level/legged_gym/models/message_passing.py`
- New structured action codec: `low-level/legged_gym/utils/action_codec.py`
- New high-level multi-head policy: `low-level/legged_gym/models/multi_agent_policy.py`
- New arm IK interface/controller: `low-level/legged_gym/controllers/arm_ik_controller.py`
- New gripper controller/policy: `low-level/legged_gym/controllers/gripper_controller.py`

## Existing State

The current low-level policy already has separate leg and arm actor heads. The
environment action dimension is 18: first 12 leg position targets and last 6
arm-related action dimensions. In `ManipLoco.step`, the last 6 action dims are
currently zeroed and the arm target is generated from the environment's current
end-effector goal through damped least-squares IK.

Current observations are built in `ManipLoco.compute_observations` from body
orientation, base angular velocity, joint position/velocity, last leg action,
foot contact, 3D base command, end-effector goal position and orientation. The
new `use_arm_base_message` switch appends `m_a2b` to the current proprioceptive
observation without changing the default path.

## Structured High-Level Action

For chunk horizon `H`, the action dim is:

```text
total_action_dim = base_command_dim + H * ee_delta_dim + grasp_intent_dim + h_choice_dim
                 = 5 + H * 6 + 1 + 3
```

With the default `H = 3`, `total_action_dim = 27`.

The structured fields are:

```python
{
    "base_command": Tensor[B, 5],
    "arm_action_chunk": Tensor[B, H, 6],
    "grasp_intent": Tensor[B, 1],
    "h_logits": Tensor[B, 3],
}
```

Use `flatten_high_level_action` and `unflatten_high_level_action` to cross the
existing flat PPO boundary.

## Messages

`m_a2b` has shape `[B, 5]`:

```text
[mean future ee delta position (3), future ee action amplitude (1), manipulability (1)]
```

`m_a2g` has shape `[B, 2]`:

```text
[rho_align, terminal_ee_motion]
```

Manipulability uses the determinant form by default:

```text
sqrt(clamp(det(J J^T + eps I), min=0))
```

When determinant output is non-finite or negative, the implementation falls back
to:

```text
sqrt(prod(clamp(singular_values(J), min=eps)))
```

## Stages

1. Base locomotion PPO:
   - Input: existing robot state + base command + optional `m_a2b`
   - Output: leg joint targets through the existing PD controller
   - Enable `use_arm_base_message` and optionally `use_assist_reward`

2. Arm IK controller:
   - No neural training
   - Feed current EE pose, action chunk and `h`
   - Use project kinematics/Jacobian when available
   - Current controller has a DLS path if a batched Jacobian is supplied

3. Gripper:
   - Default: rule-based close/open
   - Optional: `GripperPolicyMLP` trained by `train_gripper_bc.py`

4. High-level policy:
   - BC pretrain with base MSE, arm SmoothL1, grasp BCEWithLogits, h CE
   - PPO fine-tune with base, IK and gripper frozen

5. Joint finetune:
   - Not default
   - Prefer high-level + gripper first
   - Only unfreeze base last layers if needed

## TODOs

- Wire a real B1+Z1 FK/Jacobian backend into `KinematicsInterface`.
- Replace the current object-to-wrist placeholder with a real wrist camera
  transform when camera observations are available.
- Add an environment wrapper that consumes the full 27D high-level action,
  discretizes `h`, runs the fixed low-level base policy, arm IK and gripper
  agent, then steps the simulator.
- Decide whether high-level PPO should keep Gaussian h-logit dimensions or move
  to a mixed continuous/categorical distribution.

