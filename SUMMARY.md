# B2-Z1 Reachable 训练配置总结

本文只总结当前代码中的两种训练方式：

- `b2_z1_reachable_workspace`
- `b2_z1_reachable_balanced`

对应配置文件为 [`low-level/legged_gym/envs/manip_loco/b2z1_config.py`](low-level/legged_gym/envs/manip_loco/b2z1_config.py)，环境注册位于 [`low-level/legged_gym/envs/__init__.py`](low-level/legged_gym/envs/__init__.py)。

## 1. 共同基础

### B2-Z1 机器人配置

两种任务都继承 B2-Z1 的基础配置，而不是直接使用 B1 的机器人参数：

- URDF：`resources/robots/b2_z1/urdf/b2_z1.urdf`
- 初始机身高度：`0.55`
- locomotion action：前 12 个维度，对应四条腿
- arm action：后 6 个维度
- B2 关节 action scale：腿部 `0.35`，机械臂 `0.25`
- 机械臂 base offset：`[0.0, 0.0, 0.09]`
- `observe_gait_commands=True`
- policy 输出使用 tanh，以限制 action 范围

B2 的基础 locomotion reward 主要包括：

- `walking_dof = 1.0`
- `tracking_lin_vel_max = 2.5`
- `tracking_ang_vel = 0.5`
- `tracking_contacts_shaped_force = -0.2`
- `tracking_contacts_shaped_vel = -0.2`
- `torques = -1.0e-5`

这些是继承层的默认值；两个 reachable 任务只覆盖其中与实验目的相关的项目。

### EE 的定义

两种任务都使用：

```python
grasp_offset = 0.086
use_grasp_point_for_ee = True
```

这表示 EE tracking 使用 grasp point，而不是简单使用末端 link 原点。`0.086` 是沿末端方向的 grasp offset，用于让 tracking 点更接近实际抓取位置。

### 动作和全身控制路径

当前 `ManipLoco.step()` 的实际控制路径如下：

1. policy 输出 18 维 action。
2. 前 12 维保留为腿部 action。
3. 后 6 维 arm action 在环境 step 中被 mask 为 0。
4. 机械臂根据当前 EE goal 和当前 EE 状态，通过 IK 计算关节位置目标。
5. 四条腿由 PPO policy 直接控制，机械臂由 EE goal + IK 控制。

因此这里不是“18 个关节都由同一个 PPO action 直接驱动”的结构。arm policy head 仍存在，但当前环境执行路径将机械臂动作交给 IK；leg PPO 的学习则通过 locomotion advantage 和 arm advantage 的混合完成。

### EE reward 的归属和腿部关联

`tracking_ee_world` 当前只在 `rewards.arm_scales` 中保留：

```python
tracking_ee_world = 0.8
```

它不会再同时作为腿部 `rewards.scales` 和 arm `arm_scales` 各计算一次。环境会分别生成：

```text
rew_buf      -> locomotion/leg reward
arm_rew_buf  -> arm EE reward
```

PPO 在 `only_train_leg=True` 时使用：

```text
leg_advantage_for_policy
    = leg_advantage + mixing_ratio * arm_advantage
```

所以移除腿部 reward 字典中的重复项，并不切断 EE tracking 与腿策略的关联；EE reward 仍通过 `arm_advantage` 影响腿部 PPO surrogate loss。这样避免同一个 EE tracking 项重复累计，同时保留 whole-body 的训练耦合。

## 2. `reachable_workspace`

### 训练目标

该任务用于测试在一个相对保守、容易达到的目标空间内，机械臂 EE tracking 与基础 locomotion 的表现。它更接近 reachable workspace 的基准实验，而不是激进的全身协调训练。

### 目标空间

当前配置：

```python
sphere_center.x_offset = 0.0
sphere_center.z_invariant_offset = 0.8

pos_l = [0.45, 0.82]
pos_p = [-1.00, 0.80]
pos_y = [-0.90, 0.90]
```

含义：

- 目标球心在机身前方的额外 x 偏移为 `0.0`
- 目标高度参考偏移为 `0.8`
- 前后距离范围为 `0.45` 到 `0.82`
- pitch 范围为 `-1.00` 到 `0.80`
- lateral/yaw 方向范围为 `-0.90` 到 `0.90`

相比 B2/aggressive 配置，这个空间的纵向范围更短、pitch 范围也更保守。它适合做可达性和 tracking 稳定性基准，但不适合直接代表 B2 更大体型下的最大工作空间。

### Reward 配置

```python
walking_dof = 0.9
tracking_lin_vel_max = 2.75
tracking_ang_vel = 0.5
feet_height = 0.0
feet_height_target = 0.3  # 从 B1 基础配置继承
```

`feet_height_target=0.3` 仍然存在，但 `feet_height=0.0` 会使该 reward 项在 reward preparation 阶段被移除，因此训练中不会产生有效的脚高信号。这是 workspace 与其他模型之间一个可确认的差异：不是目标值变成了 0，而是整项 reward 被关闭了。

更准确地说，当前 `_reward_feet_height()` 并不是一个对四只脚逐只计算的标准 swing-foot clearance reward：

```python
# feet_height_allfeet=False，因此这里只取两只前脚
feet_height = rigid_body_state[:, feet_indices[:2], 2]
rew = clamp(norm(feet_height) - feet_height_target, max=0)
```

它只在机器人有 walking command 时生效，返回值不大于 0；配合正的 scale，实际作用是惩罚两只前脚的高度范数低于目标，而不是额外奖励超过目标的抬脚。该高度还是 rigid-body 的 world z，不是逐脚相对地形的 clearance。因而 `scale=0.0` 的确定含义是移除这个低高度惩罚，但仅凭它不能单独证明前腿不动的全部原因。

### PPO schedule 和 checkpoint

workspace 继承 B1/B2 的默认：

```python
mixing_schedule = [1.0, 0, 3000]
```

其语义为 `[最大混合比例, 开始 iteration, 混合持续时间]`：

- 从第 0 iteration 开始混入 arm advantage
- 在接下来的 3000 iteration 线性增加到 `1.0`
- 之后保持最大比例 `1.0`

当前 checkpoint 保存间隔覆盖为：

```python
save_interval = 500
```

### 适用场景和限制

workspace 更适合作为“较小 reachable 空间 + 基础 locomotion”的对照。由于 `feet_height=0.0`，它不能用来判断脚步抬高 reward 对 whole-body 协调的影响；如果需要研究腿部主动调整 EE 的能力，应重点和 balanced 配置对比，而不能只比较 EE reward 数值。

## 3. `reachable_balanced`

### 训练目标

该任务是在 B2/aggressive 目标空间上进行 whole-body balanced training：保留较大的 EE 目标范围，同时恢复脚步高度 reward，并限制 arm advantage 对腿部 PPO 的最大影响，避免 arm tracking 完全压过 body/leg locomotion。

### 目标空间

当前配置直接采用 B2/aggressive 的空间：

```python
sphere_center.x_offset = 0.2
sphere_center.z_invariant_offset = 0.8

pos_l = [0.45, 0.95]
pos_y = [-0.75, 0.75]
pos_p = [-pi / 2.5, pi / 3]
```

数值形式的 pitch 范围约为：

```text
[-1.2566370614, 1.0471975512]
```

与 workspace 相比，balanced：

- x 方向中心向前增加 `0.2`
- 最大纵向距离从 `0.82` 增加到 `0.95`
- pitch 使用 B2/aggressive 默认范围
- y 范围收敛为 `[-0.75, 0.75]`

这样不再人为把 B2 的目标工作空间压得过小，同时保留 grasp point 作为 EE 定义。

### Reward 配置

```python
walking_dof = 0.5
tracking_lin_vel_max = 3.5
tracking_ang_vel = 0.5
tracking_contacts_shaped_force = -0.2
tracking_contacts_shaped_vel = -0.2
feet_height = 1.0
feet_height_target = 0.3  # 继承基础配置
torques = -1.0e-5
```

这里 `feet_height=1.0` 的作用是重新启用前脚高度不足惩罚；`feet_height_target=0.3` 是该惩罚的阈值，不是 reward 开关。它能恢复前脚高度相关的训练信号，但当前实现不是逐脚、相对地形的标准 clearance tracking，因此仍应结合视频和 reward metric 判断效果。

`walking_dof` 的实际实现为：

```python
dof_error = sum(abs(leg_dof_pos - default_leg_dof_pos))
rew = exp(-0.05 * dof_error)
rew[not_walking] = 0
```

所以它不是“让腿摆动起来”的 reward，而是在 walking command 下奖励腿关节接近 default pose。权重越大，腿部偏离默认姿态的代价越明显；从 workspace 的 `0.9` 降到 balanced 的 `0.5`，实际是在放松这种默认姿态约束，给 body/leg 为 EE tracking 做调整留出更多空间。

### PPO mixed schedule

当前配置：

```python
mixing_schedule = [0.5, 2000, 2000]
```

按 PPO 实现的计算方式：

```python
ratio = min(max((counter - start) / duration, 0), 1) * max_ratio
```

因此：

- `0` 到 `2000` iteration：`mixing_ratio = 0`，先学习 locomotion
- `2000` 到 `4000` iteration：arm advantage 线性混入
- `4000` iteration 之后：最大混合比例为 `0.5`

腿部 policy 的有效 advantage 为：

```text
leg_advantage + mixing_ratio * arm_advantage
```

最大比例限制为 `0.5`，用于避免 EE/arm reward 完全压过 body/leg reward；但 arm tracking 仍然可以推动腿部通过 base 高度、机身姿态和动态稳定性来改善 EE 误差。

### Checkpoint 保存

balanced runner 当前覆盖为：

```python
save_interval = 500
```

训练会按 iteration 保存 `model_0.pt`、`model_500.pt`、`model_1000.pt` 等 checkpoint。

## 4. 两种方式对比

| 项目 | `reachable_workspace` | `reachable_balanced` |
|---|---:|---:|
| 目标空间 | 较保守 workspace | B2/aggressive workspace |
| `sphere_center.x_offset` | `0.0` | `0.2` |
| `pos_l` | `[0.45, 0.82]` | `[0.45, 0.95]` |
| `pos_y` | `[-0.90, 0.90]` | `[-0.75, 0.75]` |
| `pos_p` | `[-1.00, 0.80]` | `[-1.2566, 1.0472]` |
| `walking_dof` | `0.9` | `0.5` |
| `tracking_lin_vel_max` | `2.75` | `3.5` |
| `feet_height` | `0.0`，关闭该 reward | `1.0`，启用该 reward |
| `feet_height_target` | `0.3` | `0.3` |
| `tracking_ee_world` | arm scale `0.8` | arm scale `0.8` |
| arm advantage mixing | `[1.0, 0, 3000]` | `[0.5, 2000, 2000]` |
| checkpoint interval | `500` | `500` |

## 5. 如何解释之前的现象

### 前腿不动

前腿少动不能只归因于一个参数。当前代码中至少有两个方向一致的因素：

- `feet_height=0.0` 移除了前脚高度不足惩罚，降低了抬高前脚的直接驱动。
- `walking_dof=0.9` 奖励 walking 时腿关节靠近 default pose；它不是摆腿奖励，较大的 scale 反而会约束腿部偏离。

再加上机械臂由 IK 直接追踪 EE target，policy 可能没有足够收益去通过 body/leg 改变完成 tracking。因此此前把 `walking_dof` 理解成“增大后会促使四肢运动”是不准确的。balanced 使用 `feet_height=1.0` 和 `walking_dof=0.5`，分别恢复前脚低高度惩罚、放松默认腿姿态约束，但最终因果仍需通过消融回放验证。

### Arm tracking 好，但 body/leg tracking 变差

如果 arm advantage 从训练一开始就以较大比例进入腿部 PPO，或者同一个 EE reward 被重复计入，leg policy 可能优先优化 EE 相关目标，牺牲 locomotion。balanced 现在先进行 2000 iteration locomotion 学习，再在 2000 iteration 内逐步混入 arm advantage，并限制最大比例为 `0.5`，用来缓和这种失衡。

### target pose 和 current pose 看起来距离很远

EE target 的球坐标命令相对于机器人定义，但每一步都会重算为 world 坐标：目标中心的 x/y 跟随 `root_states`，方向跟随 `base_yaw_quat`，z 使用相对地形的固定 offset；observation 再把 world target 转回相对 arm base 的局部坐标。因此底盘平移和 yaw 会同时移动 EE 与 target，“走得更远”通常不会像追逐一个固定 world point 那样直接降低相对 EE 误差；base 高度、roll/pitch 和动态状态仍会改变两者关系。

workspace 与 balanced 的目标空间采样范围不同，会造成视频中 target/current 距离观感不同。测试脚本主要负责回放和渲染，目标距离的核心来源仍是训练环境中的 workspace、球心和轨迹采样配置，而不是单独的视频录制参数。

## 6. 代码修改清单

### `b2z1_config.py`

- 定义 B2-Z1 的机器人、初始姿态、PD 参数、action scale 和基础 reward。
- 增加 `B2Z1ReachableWorkspaceCfg` 与对应 PPO 配置。
- 增加 `B2Z1ReachableBalancedCfg` 与对应 PPO 配置。
- 为两种任务分别设置目标空间和 locomotion reward。
- balanced 显式设置 `[0.5, 2000, 2000]` 的 mixed schedule。
- 两种 runner 都把 checkpoint interval 覆盖为 `500`。

### `envs/__init__.py`

注册两个可以直接传给 `train.py --task` 的任务名：

```text
b2_z1_reachable_workspace
b2_z1_reachable_balanced
```

### `manip_loco.py`

- 根据 `grasp_offset` 计算 grasp point 的位置、速度和 point Jacobian。
- 在环境执行前 mask 后 6 维 policy arm action，避免 rollout action 与实际执行 action 不一致。
- 使用 EE position/orientation error 和 Jacobian IK 生成机械臂关节位置目标。
- 分开维护 `rew_buf` 与 `arm_rew_buf`。
- 将 world EE target 转换为相对 arm base 的 observation。
- 每一步根据 base x/y 和 yaw 更新 EE target 的 world 表达。

### `ppo.py`

- 当前训练路径固定 `only_train_leg=True`。
- leg surrogate loss 使用 `leg_advantage + mixing_ratio * arm_advantage`。
- `mixing_schedule` 按 `[max_ratio, start_iteration, duration]` 解释。

### `train.py`

- B2-Z1 reachable 任务会把 `b2z1_config.py` 保存到对应 W&B run，便于确认每次训练实际加载的配置，而不是误存 B1 配置。

### `on_policy_runner.py`

runner 本身仍使用通用的 `if it % save_interval == 0` 保存逻辑；本次没有为 reachable 单独复制保存代码，只在配置层将 interval 覆盖为 `500`。

## 7. 当前启动约定

两种训练均使用：

```text
conda environment: b1z1
num_envs: 6144
max_iterations: 45000
rows: 10
cols: 20
sim_device: cuda:0
rl_device: cuda:0
pipeline: gpu
```

两种任务都从零开始训练时，应使用新的、明确的 experiment ID，并确认没有携带 `--resume`、`--resumeid` 或旧 checkpoint 参数。视频回放应明确指定对应实验目录中的 `model_*.pt`，不要从 W&B 的其他 run 自动选择 checkpoint。

两套 `6144` environments 都使用 `cuda:0` 时，不应同时启动或同时进行 Isaac Gym terrain 初始化。正确流程是先启动并确认第一套训练的显存占用和 iteration 正常，再决定是否有足够显存启动第二套；默认应串行训练，避免 terrain 构建和 PhysX buffer 在初始化峰值阶段触发 OOM。

## 8. 关键代码位置

- B2 两种任务配置：[b2z1_config.py](low-level/legged_gym/envs/manip_loco/b2z1_config.py#L129)
- workspace PPO runner：[b2z1_config.py](low-level/legged_gym/envs/manip_loco/b2z1_config.py#L154)
- balanced reward 与 workspace：[b2z1_config.py](low-level/legged_gym/envs/manip_loco/b2z1_config.py#L159)
- balanced mixed schedule：[b2z1_config.py](low-level/legged_gym/envs/manip_loco/b2z1_config.py#L189)
- action mask、arm IK 和物理执行：[manip_loco.py](low-level/legged_gym/envs/manip_loco/manip_loco.py#L105)
- leg/arm reward 分开计算：[manip_loco.py](low-level/legged_gym/envs/manip_loco/manip_loco.py#L217)
- 两个任务注册：[__init__.py](low-level/legged_gym/envs/__init__.py#L44)
- PPO mixing ratio 计算：[ppo.py](third_party/rsl_rl/rsl_rl/algorithms/ppo.py#L310)
- leg advantage 混合：[ppo.py](third_party/rsl_rl/rsl_rl/algorithms/ppo.py#L205)
