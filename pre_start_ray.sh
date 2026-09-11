#!/usr/bin/env bash

export RLINF_NODE_RANK=0
export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_PATH}"
source /home/shiliangzhi/work-space/wangyinghan/RLinf/.venv/bin/activate
export PYTHONPATH="${REPO_PATH}:/home/shiliangzhi/work-space/wangyinghan/RLinf/.venv/libero"
export RLINF_COMM_NET_DEVICES=rlinf
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
