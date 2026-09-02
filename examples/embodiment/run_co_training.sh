#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_NAME="${1:-co_rl_pick_and_place_async_ppo_cnn_paired}"
CONFIG_NAME="${CONFIG_NAME%.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

CONFIG_FILE="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Missing required file: ${CONFIG_FILE}" >&2
    exit 2
fi

export EMBODIED_PATH="${SCRIPT_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

log_root="${RLINF_LOG_PATH:-${REPO_DIR}/results}"
log_dir="${log_root}/${CONFIG_NAME}/$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${log_dir}"

cmd=(
    python "${SCRIPT_DIR}/train_async.py"
    --config-path "${SCRIPT_DIR}/config"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${log_dir}"
    "$@"
)
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "${log_dir}/train.log"
