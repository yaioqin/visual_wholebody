#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vbc

# 设置IsaacGym路径
export ISAACGYM_PATH=/home/dxy/projects/visual_wholebody_new/third_party/isaacgym
export ISAACGYM_BINDINGS=$ISAACGYM_PATH/python/isaacgym/_bindings/linux-x86_64

# 设置库路径
export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS/usd/lib:$LD_LIBRARY_PATH

# 设置Vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/$(id -u)

# 设置Python路径
export PYTHONPATH=$ISAACGYM_PATH/python:$PYTHONPATH

# 使用GPU 1
export CUDA_VISIBLE_DEVICES=1

# 切换到high-level目录
cd ~/projects/visual_wholebody_new/high-level

# 运行训练（去掉CUDA_LAUNCH_BLOCKING以提高性能）
python train_multistate.py \
  --rl_device "cuda:0" \
  --sim_device "cuda:0" \
  --timesteps 120000 \
  --headless \
  --task B1Z1PickMulti \
  --experiment_dir b1-pick-multi-teacher \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --stop_pick