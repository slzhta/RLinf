#!/usr/bin/env bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/wangyinghan/RLinf/.venv/bin/python}"

export EMBODIED_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RAY_ADDRESS=local
export RLINF_NODE_RANK=0
export RLINF_REAL_PNP_DATA="${RLINF_REAL_PNP_DATA:-${REPO_PATH}/data_collection/shuangqing_real_bc_single_view}"
export RLINF_LOG_PATH="${RLINF_LOG_PATH:-${REPO_PATH}/results/realworld_pick_and_place_bc_single_view}"

exec "${PYTHON_BIN}" "${EMBODIED_PATH}/train_embodied_agent.py" \
  --config-name realworld_pick_and_place_bc_cnn "$@"
