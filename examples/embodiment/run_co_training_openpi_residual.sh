#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_NAME="${1:-co_rl_peg_insertion_openpi_residual_wrist_state_n32_ue4}"
if [[ $# -gt 0 ]]; then
    shift
fi

cd "${REPO_DIR}"
source "${REPO_DIR}/.venv/bin/activate"
unset ROS_PACKAGE_PATH
echo 'Use the existing three-node Ray cluster. This script does not restart Ray or ROS.'
echo 'Stop other robot training/evaluation/collection before launching.'
exec bash "${SCRIPT_DIR}/run_co_training.sh" "${CONFIG_NAME}" "$@"
