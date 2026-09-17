#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
source pre_start_ray.sh
set -u
export RAY_ADDRESS="${RAY_ADDRESS:-172.16.88.2:6379}"

echo 'PnP real-only training: NUC + 4090-2; 240 transitions/update, update_epoch=4.'
echo 'Initial SFT OpenPI + BC shared features/gripper; no co-training checkpoint.'
echo 'Stop other robot training/evaluation and release its controller before launch.'
echo 'Requires the existing Ray cluster. No automatic Ray/ROS restart.'
exec bash "${SCRIPT_DIR}/run_co_training_real_world.sh" \
    real_rl_pick_and_place_openpi_residual_shared_t1_t240_ue4 "$@"
