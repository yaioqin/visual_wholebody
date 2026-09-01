# Isaac Gym 无 Xorg 服务器录制回放视频手册

本文说明如何在没有桌面环境、没有 Xorg 的 GPU 服务器上，用 NVIDIA EGL、Isaac Gym camera sensor 和 GPU PhysX 回放 low-level checkpoint，并生成 H.264 MP4。文中的最终脚本还会把 VBC 调试标记叠加到视频中。

## 1. 已验证结论

当前 GPUHome 服务器不需要安装 Xorg。问题来自默认 NVIDIA Vulkan ICD：

```text
/etc/vulkan/icd.d/nvidia_icd.json
library_path = libGLX_nvidia.so.0
```

`libGLX_nvidia.so.0` 依赖 X Server 路径，在无桌面容器中容易导致 Isaac Gym 图形初始化失败。服务器已经提供 EGL 库：

```text
/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
```

给回放进程单独指定一个使用 `libEGL_nvidia.so.0` 的临时 Vulkan ICD 后，Isaac Gym camera sensor 可以正常离屏渲染。该方案：

- 不安装或启动 Xorg；
- 不修改 `/etc/vulkan` 或系统驱动；
- 不影响正在运行的训练进程；
- 只对当前回放进程设置环境变量；
- 仿真、策略推理和渲染都可以使用 GPU。

GPUHome 官方参考：

- [常见问题：Vulkan/EGL 渲染](https://docs.gpuhome.cc/docs/faq/common-issues.html)
- [远程桌面配置](https://docs.gpuhome.cc/docs/environment/remote-desktop.html)

## 2. 最终脚本

项目内脚本：

```text
low-level/legged_gym/scripts/play_aliengo_z1_video.py
```

脚本基于原来的 `play.py`，但只在本进程内替换录像函数。它不会修改 checkpoint、URDF、奖励、策略输入或训练代码，主要增加以下能力：

1. 自动将当前 checkout 的 Isaac Gym、RSL-RL 和 low-level 包放到 Python 路径最前面，防止误用另一份 checkout 的 editable install。
2. 自动创建 `/tmp/nvidia_icd_egl.json`，并设置 `VK_ICD_FILENAMES` 和 `XDG_RUNTIME_DIR`。
3. 不创建交互 viewer，但保留 graphics device，用 camera sensor 离屏渲染。
4. 在 `step_graphics()` 前调用 `gym.fetch_results()`，同步 GPU PhysX 和图形场景。
5. 使用世界坐标跟随机身，避免 terrain environment origin 导致相机拍向错误位置。
6. 把 VBC viewer debug lines 投影并叠加到 camera RGB 图像。

## 3. 标记含义

camera sensor 的 `IMAGE_COLOR` 不包含 `gymutil.draw_lines()` 所在的 viewer 调试层。因此脚本使用相同的环境状态，在输出帧上重绘：

| 标记 | 含义 | 数据来源 |
| --- | --- | --- |
| 蓝色圆环 | 实际末端位置，current pose | `env.ee_pos` |
| 黄色圆环 | 当前目标位置，target pose | `env.curr_ee_goal_cart_world` |
| 红绿蓝坐标轴 | 目标姿态 | `env.ee_goal_orn_quat` |
| 青色圆环 | EE 球坐标目标空间中心 | `_get_ee_goal_spherical_center()` |
| 红色点 | EE 起点到最终目标的插值轨迹 | `ee_start_sphere`, `ee_goal_sphere` |
| 红色边框 | 机械臂目标碰撞限制区域 | `collision_lower_limits`, `collision_upper_limits` |

这些标记仅用于可视化，不反馈到仿真或策略。

## 4. 环境检查

在仓库根目录执行：

```bash
cd /root/rivermind-data/visual_wholebody_origin

grep -R 'library_path' /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null
ls -l /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
nvidia-smi
```

预期能够看到默认 ICD 使用 `libGLX_nvidia.so.0`，同时 `libEGL_nvidia.so.0` 存在。

检查 checkpoint：

```bash
ls -lh low-level/logs/aliengo-z1-low/aliengo_z1_vbc_fixed_20260805/model_12000.pt
```

## 5. Conda 环境

推荐使用项目已经验证的 `b1z1` 环境：

```bash
conda activate b1z1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
```

脚本需要 `numpy`、`Pillow`、`imageio`、`imageio-ffmpeg`、PyTorch 和项目原有依赖。检查 Pillow：

```bash
python -c 'from PIL import Image, ImageDraw; print("Pillow OK")'
```

如果当前 shell 不方便激活 Conda，也可以直接使用：

```bash
/opt/conda/envs/b1z1/bin/python
```

## 6. Terrain 场景录制

这是推荐命令。没有 `--flat_terrain` 时，`play.py` 使用项目的 `trimesh` rough terrain，并在回放时设置 `6 x 3` terrain grid：

```bash
cd /root/rivermind-data/visual_wholebody_origin

export LD_LIBRARY_PATH="/opt/conda/envs/b1z1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

/opt/conda/envs/b1z1/bin/python -u \
  low-level/legged_gym/scripts/play_aliengo_z1_video.py \
  --task aliengo_z1_bounded_actions \
  --exptid aliengo_z1_vbc_fixed_20260805 \
  --proj_name aliengo-z1-low \
  --checkpoint 12000 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --pipeline gpu \
  --record_video \
  --play_seconds 9 \
  --video_tag model12000-vbc-markers-terrain-follow
```

本机验证时，terrain 创建了 370,248 个顶点和 740,490 个三角形，高度范围为 `0–0.1 m`。回放结果为 451 步、9.02 秒、0 次 reset。

## 7. Flat 场景录制

只需添加 `--flat_terrain`：

```bash
cd /root/rivermind-data/visual_wholebody_origin

export LD_LIBRARY_PATH="/opt/conda/envs/b1z1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

/opt/conda/envs/b1z1/bin/python -u \
  low-level/legged_gym/scripts/play_aliengo_z1_video.py \
  --task aliengo_z1_bounded_actions \
  --exptid aliengo_z1_vbc_fixed_20260805 \
  --proj_name aliengo-z1-low \
  --checkpoint 12000 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --pipeline gpu \
  --record_video \
  --flat_terrain \
  --play_seconds 9 \
  --video_tag model12000-vbc-markers-flat-follow
```

## 8. 其他回放模式

底盘命令置零，但机械臂目标继续更新：

```bash
... --zero_commands
```

静止/待机检查：

```bash
... --stand_by
```

使用随机策略动作而不是确定性推理：

```bash
... --stochastic
```

指定另一张 GPU：

```bash
... --sim_device cuda:1 --rl_device cuda:1
```

如果同一张 GPU 上正在训练，保持回放为单环境即可。原 `play.py` 已固定 `env_cfg.env.num_envs = 1`，短时回放结束后显存会释放。

## 9. 路径规则

checkpoint 输入路径由参数组成：

```text
low-level/logs/<proj_name>/<exptid>/model_<checkpoint>.pt
```

以上示例对应：

```text
low-level/logs/aliengo-z1-low/aliengo_z1_vbc_fixed_20260805/model_12000.pt
```

视频输出目录：

```text
low-level/logs/videos/<exptid>/
```

文件名格式：

```text
<exptid>-<video_tag>-<environment_index>-<checkpoint>.mp4
```

## 10. 验证视频

以 terrain 视频为例：

```bash
VIDEO=/root/rivermind-data/visual_wholebody_origin/low-level/logs/videos/aliengo_z1_vbc_fixed_20260805/aliengo_z1_vbc_fixed_20260805-model12000-vbc-markers-terrain-follow-0-12000.mp4

ffprobe -v error \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of default=noprint_wrappers=1 \
  "$VIDEO"
```

已验证输出：

```text
codec_name=h264
width=720
height=480
r_frame_rate=25/1
nb_frames=226
duration=9.040000
```

抽取中间帧：

```bash
ffmpeg -y -ss 4.5 -i "$VIDEO" -frames:v 1 /tmp/aliengo_z1_video_mid.png
```

回放结束时还会输出：

```text
Play summary: steps=451, duration_s=9.02, resets=0, max_leg_action_abs=1.000
```

`resets=0` 表示这次短回放没有发生环境 reset，但不能代替大规模或长时间策略评估。

## 11. 为什么不能直接使用普通 `--headless`

项目的 `BaseTask` 在 `headless == True` 时把 `graphics_device_id` 设置为 `-1`。这样适合训练，但 camera sensor 没有可用 graphics device，无法生成 RGB 视频。

最终脚本使用一个特殊 sentinel：

- 它不等于 `True`，所以 graphics device 保留；
- 它也不等于 `False`，所以不会创建需要显示服务器的 viewer；
- camera sensor 仍然可以通过 EGL 离屏渲染。

因此不要自行在该脚本后追加普通 `--headless` 并覆盖脚本内部行为。

## 12. 两个已解决的录像问题

### 12.1 手臂和腿像折叠在机身上

这不是 checkpoint、URDF 或 IK 错误。GPU pipeline 下，物理张量已经更新，但录像前没有执行 `gym.fetch_results()`，图形场景仍停留在 actor 创建时的零关节姿态。

最终脚本在每次录像前执行：

```python
env.gym.fetch_results(env.sim, True)
env.gym.step_graphics(env.sim)
```

同步后机械狗正常站立，Z1 手臂也与刚体位置一致。

### 12.2 同步后画面为空

terrain 环境的机器人世界坐标包含 `env_origin`。旧临时脚本对机器人位置减去了 `env_origin`，却把结果传给要求世界坐标的 `set_camera_location()`。同步图形后机器人位于真实世界位置，而相机仍在原点，因此拍不到机器人。

最终脚本直接使用 `root_states[..., :3]` 的世界坐标作为相机跟随基准。

## 13. 常见故障

### 图形初始化失败或出现 GLFW/X11 错误

确认脚本是在进程启动、导入 Isaac Gym 之前设置 EGL ICD。查看临时文件：

```bash
cat /tmp/nvidia_icd_egl.json
```

应为：

```json
{
  "file_format_version": "1.0.1",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0",
    "api_version": "1.4.312"
  }
}
```

### 导入了另一份 `visual_wholebody`

最终脚本会优先使用自身所在 checkout。仍可检查启动日志中的：

```text
Importing module 'gym_38' (.../visual_wholebody_origin/third_party/isaacgym/...)
```

路径必须包含 `visual_wholebody_origin`。

### 找不到 checkpoint

检查 `--proj_name`、`--exptid` 和 `--checkpoint`，并按第 9 节拼出完整路径验证。

### 视频存在但没有红蓝黄标记

必须使用 `play_aliengo_z1_video.py`。原始 `play.py` 读取 camera sensor 的 `IMAGE_COLOR`，不会捕获 viewer 的 `gymutil.draw_lines()`。

### `cv2` 提示缺少 `libGL.so.1`

最终脚本不依赖 OpenCV，而使用 Pillow 绘制标记，因此不需要为这个功能安装 `libGL`。

### terrain 初始化耗时或显存增加

rough terrain 要构建大量三角形，初始化比 flat 慢且会增加临时显存。先用 `--flat_terrain --play_seconds 1` 做渲染 smoke test，再运行完整 terrain 版本。

## 14. 不影响训练的确认方法

回放前后分别检查训练 PID：

```bash
ps -p 126645 -o pid,stat,etime,cmd
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

本次验证中，训练 PID `126645` 始终存活；回放结束后训练进程显存恢复为约 `7680 MiB`。PID 和 iteration 属于当时的运行快照，后续使用时应以当前服务器状态为准。

## 15. 当前已生成的视频

带 VBC 标记的 terrain 视频：

```text
/root/rivermind-data/visual_wholebody_origin/low-level/logs/videos/aliengo_z1_vbc_fixed_20260805/aliengo_z1_vbc_fixed_20260805-model12000-vbc-markers-terrain-follow-0-12000.mp4
```

带 VBC 标记的 flat 视频：

```text
/root/rivermind-data/visual_wholebody_origin/low-level/logs/videos/aliengo_z1_vbc_fixed_20260805/aliengo_z1_vbc_fixed_20260805-model12000-vbc-markers-follow-0-12000.mp4
```
