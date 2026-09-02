#! /bin/bash

# Dedicated real-world evaluation launcher for the OpenPI pi0.5 PnP policy.
# Place this file at:
#   examples/embodiment/run_pi05_realworld_eval.sh

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH
REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# OpenPI inference is PyTorch-based here. Keep JAX away from the GPU to avoid
# competing CUDA initialization and memory allocation.
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="${1:-realworld_pnp_pi05_eval}"
if [[ $# -gt 0 ]]; then
    shift
fi
EXTRA_ARGS=("$@")

echo "Using Python at $(which python)"

LOG_DIR="/tmp/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_realworld_eval.log"
mkdir -p "${LOG_DIR}"

CMD=(
    python
    "${SRC_FILE}"
    --config-path
    "${EMBODIED_PATH}/config/"
    --config-name
    "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "env.train.keyboard_reward_wrapper=null"
    "env.eval.keyboard_reward_wrapper=null"
    "${EXTRA_ARGS[@]}"
)

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"

echo "Starting pi0.5 real-world evaluation."
echo "Human feedback: S=success, F=failure, X=abort, R=next episode ready"

"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"