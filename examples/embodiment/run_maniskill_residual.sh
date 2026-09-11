#!/usr/bin/env bash
# Simulation-only launcher using the standard embodied PPO entry point.
set -euo pipefail

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="${1:-maniskill_pick_and_place_ppo_residual}"
if [[ $# -gt 0 ]]; then
    shift
fi
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
python "${EMBODIED_PATH}/train_embodied_agent.py" \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name "${CONFIG_NAME}" \
    "runner.logger.log_path=${LOG_DIR}" "$@" 2>&1 | tee "${LOG_DIR}/run.log"
