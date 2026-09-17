#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
source pre_start_ray.sh
set -u
export RAY_ADDRESS="${RAY_ADDRESS:-172.16.88.2:6379}"

echo 'PnP co-training: 16 sim envs, shared features, gripper temperature 1, frozen BC backbone.'
echo 'Requires the existing three-node Ray cluster. No automatic Ray/ROS restart.'
echo 'Confirm the PnP workspace, target/reset pose and two cameras before launch.'
exec bash "${SCRIPT_DIR}/run_co_training.sh" \
    co_rl_pick_and_place_openpi_residual_shared_t1_n16_ue4 "$@"
