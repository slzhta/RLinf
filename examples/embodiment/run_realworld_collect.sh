#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_NAME="${1:-real_eval_peg_insertion_openpi_residual_wrist_state}"
CONFIG_NAME="${CONFIG_NAME%.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ ! "${CONFIG_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Use a config filename, not a path.' >&2
    exit 2
fi
cd "${REPO_DIR}"
if [[ ! -f pre_start_ray.sh || ! -x .venv/bin/python ]]; then
    echo 'Run this on 4090-2 in the deployed RLinf repository.' >&2
    exit 2
fi
source pre_start_ray.sh
set -u
export EMBODIED_PATH="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export RAY_ADDRESS="${RAY_ADDRESS:-172.16.88.2:6379}"
export RLINF_EVAL_RUN_ID="$(date +'%Y%m%d-%H%M%S')-collect-$$"
export HYDRA_FULL_ERROR=1
log_dir="${RLINF_LOG_PATH}/collect_${CONFIG_NAME}/${RLINF_EVAL_RUN_ID}"
mkdir -p "${log_dir}"

echo 'Collection only: no training and no automatic Ray/ROS restart.'
echo 'Stop other training/evaluation and release its robot controller before launching.'
echo 'After each episode: k=keep, d=discard, q=discard and quit. No next reset until reviewed.'
"${REPO_DIR}/.venv/bin/python" "${SCRIPT_DIR}/collect_realworld.py" \
    --config-name "${CONFIG_NAME}" \
    "runner.logger.experiment_name=collect_${CONFIG_NAME}" \
    "runner.logger.log_path=${log_dir}" "$@" 2>&1 | tee "${log_dir}/collect.log"
