# B2 + Z1 真机部署（Ubuntu 20.04 / ROS Noetic）

本目录是与下列训练任务匹配的 ROS1 真机控制包：

```bash
python train.py --headless --exptid b2_z1 --proj_name b2z1-low \
  --task b2z1 --sim_device cuda:0 --rl_device cuda:0 \
  --observe_gait_commands
```

实现依据是宇树官方的 [B2 开发文档](https://support.unitree.com/home/zh/B2_developer/About%20B2)、[Z1 开发文档](https://support.unitree.com/home/zh/Z1_developer/z1)、`unitree_sdk2`、`z1_controller` 和 `z1_sdk`。B2 由 SDK2 DDS 低层通道控制；Z1 由 `z1_controller` 连接下位机，本包的 SDK 桥接节点向它发送关节目标。

## 与训练环境的对应关系

- 腿关节及动作顺序：`FR, FL, RR, RL`，每条腿为 `hip, thigh, calf`。这正是训练代码 `_reindex_all()` 后的策略顺序，也是 B2 SDK 前 12 个电机的顺序。
- 当前本体观测为 71 维：姿态 2、角速度 3、18 个关节位置、18 个关节速度、上一次腿动作 12、足端接触 4、速度命令 3、末端目标位置 3、末端目标姿态占位 3、步态相位 1、步态时钟 4。
- 历史为 10 帧。因此训练环境张量是 `71 + 18(特权量) + 10×71 = 799` 维；真机历史编码策略不使用特权量，导出的输入是 `71 + 10×71 = 781` 维。
- 策略输出为 18 维，但当前 `ManipLoco.step()` 会把后 6 维清零。腿使用策略前 12 维；Z1 复现训练环境中的阻尼最小二乘 IK，不能把策略后 6 维直接发送给机械臂。
- 训练仿真周期为 50 Hz（`0.005×decimation 4`）；策略节点同样以 50 Hz 运行。B2 DDS 桥接以 500 Hz 重发最近命令，Z1 官方 SDK 通信线程为 500 Hz。
- 足端接触阈值 `1.5`、观测限幅 `±100`、速度/关节缩放及一步动作延迟均按当前 `b2z1` 训练代码实现。

当前配置的 `action_delay_steps: 1` 对应训练代码在 `global_steps >= 10000×24` 后的分支；若专门复现更早的 checkpoint，应改为 `0`。早期 checkpoint 通常也不应上真机。

`ee_goal_vector` 不是 Z1 安装座标系中的最终点，而是训练变量 `curr_ee_goal_sphere` 转成笛卡尔坐标后的向量，坐标系为重力对齐的 B2 yaw 坐标系。节点会加入训练配置的球心 `[0.2, 0, 0.8]`、B2 高度和 Z1 安装偏置，并补偿 B2 roll/pitch。默认向量对应训练初始球坐标 `[0.5, π/8, 0]`。

## 1. 依赖和编译

安装 ROS Noetic、PyTorch、NumPy、PyYAML，以及官方 SDK 依赖。ROS Noetic 在 Ubuntu 20.04 使用系统 Python 3.8；请确保同一个解释器能同时导入 `rospy` 和 `torch`，不要让不兼容的 Conda Python 覆盖它：

```bash
sudo apt update
sudo apt install -y ros-noetic-desktop-full python3-numpy python3-yaml \
  libyaml-cpp-dev libeigen3-dev libboost-all-dev libfmt-dev
python3 -c 'import rospy, torch, numpy, yaml; print("Python/ROS/PyTorch OK")'
```

获取官方代码（建议在实机联调前固定经过验证的 commit）：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
git clone https://github.com/unitreerobotics/z1_controller.git
git clone https://github.com/unitreerobotics/z1_sdk.git

cd z1_controller
mkdir build && cd build
cmake .. && make -j
```

将本目录作为 catkin 包编译。下面三个路径按实际位置修改：

```bash
mkdir -p ~/b2z1_ws/src
ln -s /home/ubuntu22/E/visual_wholebody/low-level/real ~/b2z1_ws/src/b2z1_real
cd ~/b2z1_ws
source /opt/ros/noetic/setup.bash
catkin_make \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DUNITREE_SDK2_ROOT=$HOME/unitree_sdk2 \
  -DZ1_SDK_ROOT=$HOME/z1_sdk
source devel/setup.bash
```

未提供某个 SDK 路径时，对应硬件桥不会编译，但消息、Python 策略节点和离线测试仍可编译。

## 2. 导出训练策略

不要部署 `model_0.pt` 或未收敛模型。选择已在仿真中验证的 checkpoint：

```bash
mkdir -p real/models
python3 real/scripts/export_b2z1_policy.py \
  --checkpoint logs/b2z1-low/b2_z1/model_40000.pt \
  --output real/models/policy_b2z1.pt
```

导出器会拒绝不是 71 维历史编码器的 checkpoint，并生成带 checkpoint SHA-256 的 `.json` 元数据。部署节点加载后还会执行一次 `781 → 18` 的形状和有限值检查。

## 3. 上真机前检查

第一次运行必须把 B2 用保护架悬空，Z1 工作区内无人、无负载，并有人握住物理急停。还需逐项确认：

1. B2 和 Z1 固件、SDK 与官方文档匹配；B2 网卡固定到机器人网段，Z1 的 `z1_controller/config/config.xml` 中 IP/端口正确。
2. 没有其他程序同时发送 `rt/lowcmd` 或 Z1 SDK 命令。
3. 用只读状态确认 B2 电机 0..11 确实为 `FR, FL, RR, RL`，关节正方向与 `config/b2z1_real.yaml` 一致。
4. B2 已接近配置中的站立角，最大误差小于 `initial_max_leg_error`。本节点不会从任意趴卧姿态强行站起。
5. 根据实机调参确认 `leg_kp/leg_kd`。默认 `250/5` 是训练参数，不代表每台 B2 的最终安全实机增益。
6. 根据 Z1 是否实际安装宇树夹爪设置 `z1_bridge/has_gripper`；它不改变策略的 6 个机械臂关节输入。
7. 先在仿真和悬空状态下验证策略输出、末端目标、命令限幅、状态超时和急停。

真机默认值有几项有意比训练环境保守：策略动作由训练的 `±100` 收紧为 `±1`，速度命令范围由 `±0.8 m/s`、`±1.0 rad/s` 收紧为 `±0.5 m/s`、`±0.8 rad/s`，并增加关节目标速度/位置限幅。这些值都集中在 `config/b2z1_real.yaml`，只有完成悬空测试后才应逐步放宽。

## 4. 启动

先从 `z1_controller/build` 目录启动 Z1 控制器（它用相对路径读取 `../config`）：

```bash
cd ~/z1_controller/build
./z1_ctrl
```

然后启动本包。`start_hardware_bridges` 默认是 `false`，避免误连接真机；`release_b2_motion_service` 只有在操作员明确准备进入低层控制后才设为 `true`：

```bash
source ~/b2z1_ws/devel/setup.bash
roslaunch b2z1_real b2z1_real.launch \
  policy:=/absolute/path/to/policy_b2z1.pt \
  start_hardware_bridges:=true \
  b2_network_interface:=enp3s0 \
  release_b2_motion_service:=true
```

两个桥在收到 `enabled=true` 前不会主动进入运动控制。策略节点初始为 `DISARMED`，且必须同时收到新鲜、有效的 B2 和 Z1 状态。

明确解锁：

```bash
rostopic pub -1 /b2z1/enable std_msgs/Bool 'data: true'
```

节点先用 3 秒平滑过渡到训练默认腿姿态，然后进入 `ACTIVE`。状态可查看：

```bash
rostopic echo /b2z1/control_state
```

速度命令示例（低于训练的 `0.2 m/s`、`0.5 rad/s` 步态阈值时会归零）：

```bash
rostopic pub -r 20 /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.3}, angular: {z: 0.0}}'
```

末端目标示例。此消息是“球心到目标”的原始向量，不是安装座到目标的最终位置：

```bash
rostopic pub -1 /b2z1/ee_goal_vector geometry_msgs/PointStamped \
  '{header: {frame_id: b2_base_yaw}, point: {x: 0.55, y: 0.0, z: 0.10}}'
```

如果有可靠的实时离地高度估计，可发布 `/b2z1/base_height`；否则使用配置中的 `0.55 m`。目标会经过训练范围、地下边界和机身碰撞排除区检查。

## 5. 停止、超时与故障

普通停止：

```bash
rostopic pub -1 /b2z1/enable std_msgs/Bool 'data: false'
```

紧急停止（锁存为 `FAULT`）：

```bash
rostopic pub -1 /b2z1/emergency_stop std_msgs/Bool 'data: true'
```

B2 桥在失去命令后发送纯阻尼命令；Z1 桥切换到 `PASSIVE`。纯阻尼不能保证 B2 继续站立，因此物理保护架和硬件急停不可省略。清故障前必须先发布 `enable=false`、`emergency_stop=false`，再执行：

```bash
rosservice call /b2z1/clear_fault
```

任何 CRC 错误、状态超过 100 ms、NaN、关节越界、roll/pitch 越界、温度越界或 SDK 电机错误都会阻止解锁或进入 `FAULT`。

## 6. 离线验证

```bash
python3 -m unittest discover -s real/test -p 'test_*.py' -v
python3 -m py_compile real/src/b2z1_real/*.py real/scripts/*.py
```

核心测试不需要 ROS master 或真机，包括观测维度/目标变换、Z1 Jacobian 有限差分和安全门检查。
