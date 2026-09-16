2026-09-16：b2z1_Lu / model_18000.pt 不行走排查

后续配置更新：按用户要求，B2 各配置的命令死区已统一为高层 B1 的前进 `0.15 m/s`、转向 `0.35 rad/s`，包括原始 LU 及继承它的 gait_fix。本报告实测与训练日志分析发生在该调整之前，当时使用 `0.20 m/s`、`0.50 rad/s`。

后续资产更新：按用户截图中的高层 B1 URDF，将 B2 肘关节上限由 `0` 改为 `-0.4 rad`，下限保持 `-4.782202150464463 rad`；六个机械臂关节速度上限均为 `10 rad/s`，B2 安装位置保持不变。上述范围已通过 Isaac Gym 实际加载验证；本报告回放结果来自这次限位调整之前。

使用原始 `b2z1_config_LU.py` 加载 checkpoint，在 CPU 和 GPU 0 上都复现了收到前进命令却几乎不前进的行为。模型参数全部有限，加载后的 `global_steps=432025`，`stand_by=False`、`teleop_mode=False`，动作有输出。问题不能仅靠给 play 补上 `--config` 解决。

固定命令为 `[0.6, 0, 0]`，每组运行 9 秒，以下统计取最后 7 秒；所有固定命令测试均无重置。粗糙地形沿用 play 的 6 行、3 列，关闭推扰与质心随机化，保留质量随机化。CPU/GPU 的随机地形、初态不完全相同，因此这些结果用于验证现象，并非严格的后端误差测量。

| 回放条件 | 平均机体前向速度 m/s | 前向速度绝对值均值 m/s | 7 秒平面净位移 |
| --- | ---: | ---: | ---: |
| CPU，历史编码，原始 LU | 0.00957 | 0.03047 | 2.03 cm |
| CPU，特权编码，原始 LU | 0.00354 | 0.03756 | 0.65 cm |
| CPU，平地、关闭质量/摩擦/电机/夹爪质量随机化 | -0.00499 | 0.01820 | — |
| GPU，历史编码，确定性动作 | 0.000022 | 0.03144 | 3.66 cm |
| GPU，历史编码，随机动作 | 0.00968 | 0.04331 | 2.78 cm |

这说明该 checkpoint 在这些回放条件下没有有效跟踪前进指令。短时对照不能证明所有命令、初态下都不会走，也不能证明以下缺陷是唯一成因。

发现并修复的训练问题：

1. LU 的 `tracking_contacts_shaped_force`、`tracking_contacts_shaped_vel` 权重都是 `-0.2`，而对应函数已经返回非正惩罚值。相乘后，错误接触和滑动得到正分。例如摆动期两只脚仍触地时，原始力项约为 `-0.5`，乘权重后约为 `+0.1`（总奖励还会除以 100）。新增 `legged_gym/envs/manip_loco/b2z1_config_LU_gait_fix.py`，只将两个权重改为 `+0.2`，继承其他 LU/PPO 参数。原始 LU 文件未修改。
2. `manip_loco.py` 中 `foot_velocities` / `foot_positions` 的高级索引产生副本，旧代码只在初始化时赋值。现在每次刷新刚体状态后重新取当前脚状态，使滑动奖励使用实时速度。初始化快照可能非零；不能将此问题描述为“训练中的速度奖励始终为零”。
3. `maniploco_rewards.py` 的 `feet_jerk` 检查 `self.last_contact_forces`，却读写 `self.env.last_contact_forces`，导致每次都当作首次采样并返回零。已统一为检查环境上的缓存。

训练日志支持奖励方向错误，但不能单独证明训练过程始终静止：17800–18000 轮的 Rank0 平均回合长度为 482.59 步（上限约 500），两个步态奖励的记录均值分别为 `+2.06268`、`+4.51731`，`feet_jerk` 为零。归一到每步后的速度跟踪得分约为 0.468；该指标包括小速度命令下的静止奖励，不能当成实际速度。日志中没有直接保存实际 vx 和命令。

验证：配置选择 5 项测试、脚状态/接触变化奖励 2 项回归测试通过；比较配置确认只有两个奖励符号改变，PPO、观测、PD、动作缩放和资产保持 LU 值。修正版进行了 150 步 Isaac Gym CPU 检查，脚速每步与仿真状态一致，滑动惩罚随运动变化且权重为正。尚未进行修正后的完整训练，不能承诺仅这几处修改就一定收敛到行走。

建议新建实验重新训练。以下命令从 `low-level` 根目录运行，未在排查中执行：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=3 \
  legged_gym/scripts/train.py \
  --headless --task b2z1 \
  --config legged_gym/envs/manip_loco/b2z1_config_LU_gait_fix.py \
  --proj_name b2z1-low --exptid b2z1_Lu_gait_fix \
  --distributed --num_envs 2048 --disable_wandb --observe_gait_commands
```

新实验生成 checkpoint 后，播放也指定 `b2z1_config_LU_gait_fix.py` 和新的实验名。旧 `model_18000.pt` 的权重不会因修改播放时的奖励配置而自动学会行走。已有训练进程不会自动加载这些源码改动；需要由新启动的训练进程使用修正后的代码。
