# export RLINF_NODE_RANK=0 # set to the number in the cfg for 4090
# export RLINF_NODE_RANK=1
# export RLINF_NODE_RANK=2
# unset PYTHONPATH
# export PYTHONPATH=/home/ubuntu/wangyinghan/wangyitao
# export RLINF_COMM_NET_DEVICES=enp4s0
# export RLINF_LOG_PATH=/home/ubuntu/wangyinghan/wangyitao/results
# export RLINF_COMM_NET_DEVICES=rlinf
# source /home/ubuntu/wangyinghan/RLinf/.venv/bin/activate
# export HF_LEROBOT_HOME=/home/ubuntu/wangyinghan/wangyitao/pnp_lerobot_dataset
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# source /home/ubuntu/wangyinghan/wangyitao/openpi-pnp-venv/bin/activate
#!/usr/bin/env bash
# 只使用第0张GPU
export CUDA_VISIBLE_DEVICES=0

# PyTorch CUDA显存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 当前节点在 YAML cluster.component_placement 中的编号
export RLINF_NODE_RANK=0

# 项目路径
unset PYTHONPATH
export PYTHONPATH=/home/ubuntu/wangyinghan/wangyitao

# RLinf 通信和日志
export RLINF_COMM_NET_DEVICES=enp4s0
export RLINF_LOG_PATH=/home/ubuntu/wangyinghan/wangyitao/results

# 本地 LeRobot 数据集
export HF_LEROBOT_HOME=/home/ubuntu/wangyinghan/wangyitao/pnp_lerobot_dataset
export HF_DATASETS_CACHE=/tmp/pnp_v2_hf_cache

# PyTorch CUDA 显存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SFT 使用 PyTorch GPU；阻止 JAX/XLA 占用或初始化 CUDA
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# 激活包含 OpenPI、RLinf 和 Ray 的训练环境
source /home/ubuntu/wangyinghan/wangyitao/openpi-pnp-venv/bin/activate

echo "Environment ready"
echo "Python: $(which python)"
echo "Ray:    $(which ray)"
echo "Node:   ${RLINF_NODE_RANK}"
echo "Data:   ${HF_LEROBOT_HOME}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
