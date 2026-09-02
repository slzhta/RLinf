#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RLINF_VENV="${RLINF_VENV:-/home/ubuntu/wangyinghan/RLinf/.venv}"
RUN_DIR="${1:-${REPO_ROOT}/results/maniskill_pick_and_place_bc_v3}"
MAX_STEPS="${MAX_STEPS:-3000}"
EVAL_INTERVAL=1000
SAVE_INTERVAL=1000

if [[ ! -x "${RLINF_VENV}/bin/python" ]]; then
    echo "RLinf Python not found: ${RLINF_VENV}/bin/python" >&2
    exit 1
fi

if (( SAVE_INTERVAL % EVAL_INTERVAL != 0 )); then
    echo "SAVE_INTERVAL must be divisible by EVAL_INTERVAL" >&2
    exit 1
fi

export VIRTUAL_ENV="${RLINF_VENV}"
export PATH="${RLINF_VENV}/bin:${PATH}"
export RAY_ADDRESS=local
export RLINF_NODE_RANK=0
export EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"
export RLINF_LOG_PATH="${REPO_ROOT}/results"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export ROBOT_PLATFORM=LIBERO
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}"
echo "Python: ${RLINF_VENV}/bin/python"
echo "BC run directory: ${RUN_DIR}"
echo "BC checkpoints: ${RUN_DIR}/checkpoints"
echo "BC training steps: ${MAX_STEPS}"

exec "${RLINF_VENV}/bin/python" -u "${EMBODIED_PATH}/train_embodied_agent.py" \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name maniskill_pick_and_place_bc_cnn_state \
    runner.max_epochs="${MAX_STEPS}" \
    runner.max_steps="${MAX_STEPS}" \
    runner.val_check_interval="${EVAL_INTERVAL}" \
    runner.save_interval="${SAVE_INTERVAL}" \
    runner.resume_dir=null \
    runner.ckpt_path=null \
    runner.logger.log_path="${RUN_DIR}" \
    runner.logger.experiment_name=
