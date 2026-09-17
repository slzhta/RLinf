#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_NAME="real_rl_peg_insertion_bc10000_rlpd_cnn_wrist_state"
DEFAULT_RESUME="${REPO_DIR}/results/${CONFIG_NAME}/20260915-214911/${CONFIG_NAME}/checkpoints/global_step_2770"
CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
    CHECK_ONLY=true
    shift
fi
RESUME_DIR="${1:-${DEFAULT_RESUME}}"
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash $0 [--check-only] [checkpoint_directory]" >&2
    exit 2
fi
if [[ ! -d "${RESUME_DIR}" ]]; then
    echo "Missing checkpoint directory: ${RESUME_DIR}" >&2
    exit 2
fi
RESUME_DIR="$(cd "${RESUME_DIR}" && pwd)"
if [[ ! "${RESUME_DIR##*/}" =~ ^global_step_[0-9]+$ ]]; then
    echo "Pass the global_step_N directory, not full_weights.pt." >&2
    exit 2
fi
required_files=(
    "actor/dcp_checkpoint/.metadata"
    "actor/dcp_checkpoint/__0_0.distcp"
    "actor/sac_components/alpha/dcp_checkpoint/.metadata"
    "actor/sac_components/alpha/dcp_checkpoint/__0_0.distcp"
    "actor/sac_components/target_model/checkpoint_rank_0.pt"
)
for relative_path in "${required_files[@]}"; do
    if [[ ! -s "${RESUME_DIR}/${relative_path}" ]]; then
        echo "Missing or empty checkpoint component: ${relative_path}" >&2
        exit 2
    fi
done
printf 'Resume checkpoint: %s\n' "${RESUME_DIR}"
echo "Restore actor/Q, target Q, optimizers and learned alpha/logstd; collect a fresh online buffer."
if [[ "${RESUME_DIR}" == "${DEFAULT_RESUME}" ]]; then
    echo "Source: 20260915-214911, 2400 additional online steps after resuming from 13200, global_step_2770."
    echo "New online sample counter starts at zero; add 15600 for the retained training-branch sample-step axis."
fi
echo "No automatic Ray/ROS restart. Stop old training and release its controller before launching."
if [[ "${CHECK_ONLY}" == true ]]; then
    echo "Checkpoint files checked; no environment, Ray or robot was started."
    exit 0
fi
cd "${REPO_DIR}"
source "${REPO_DIR}/pre_start_ray.sh"
exec bash "${SCRIPT_DIR}/run_co_training_real_world.sh" "${CONFIG_NAME}" \
    "runner.resume_dir=${RESUME_DIR}" \
    runner.checkpoint_replay_buffer=false
