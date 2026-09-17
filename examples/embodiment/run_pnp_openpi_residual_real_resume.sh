#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_NAME="real_rl_pick_and_place_openpi_residual_shared_t1_t240_ue4"
DEFAULT_RESUME="${REPO_DIR}/results/${CONFIG_NAME}/20260917-135928/${CONFIG_NAME}/checkpoints/global_step_50"
CHECK_ONLY=false

usage() {
    echo "Usage: bash $0 [--check-only] [global_step_N_directory]"
    echo "Default: resume the 20260917-135928 real-only run at step 50."
    echo "No automatic training termination, Ray/ROS restart or robot recovery."
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == "--check-only" ]]; then
    CHECK_ONLY=true
    shift
fi
RESUME_DIR="${1:-${DEFAULT_RESUME}}"
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi
if [[ ! -d "${RESUME_DIR}" ]]; then
    echo "Missing checkpoint directory: ${RESUME_DIR}" >&2
    exit 2
fi
RESUME_DIR="$(cd "${RESUME_DIR}" && pwd)"
if [[ ! "${RESUME_DIR##*/}" =~ ^global_step_[0-9]+$ ]]; then
    echo "Pass the global_step_N directory, not actor/ or full_weights.pt." >&2
    exit 2
fi
for relative_path in \
    actor/dcp_checkpoint/.metadata \
    actor/dcp_checkpoint/__0_0.distcp \
    actor/training_meta.pt \
    actor/model_state_dict/full_weights.pt; do
    if [[ ! -s "${RESUME_DIR}/${relative_path}" ]]; then
        echo "Missing or empty checkpoint component: ${relative_path}" >&2
        exit 2
    fi
done

printf 'Real-only PPO resume: %s\nCheckpoint: %s\n' "${CONFIG_NAME}" "${RESUME_DIR}"
echo 'Restore policy/value/logstd, optimizer, LR scheduler and training step; collect fresh data.'
echo 'First synchronize restored actor weights to rollout. Keep the original 500-step total limit.'
echo 'New timestamped logs/checkpoints; the original run is not overwritten.'
echo 'Stop old training/evaluation, release the controller and finish robot/gripper recovery first.'
if [[ "${CHECK_ONLY}" == true ]]; then
    echo 'File checks passed. No Ray, ROS, model or robot was started.'
    exit 0
fi

if ! command -v pgrep >/dev/null; then
    echo 'pgrep is required to check for an existing local driver.' >&2
    exit 2
fi
if pgrep -af '[p]ython[^ ]* .*examples/embodiment/(train_async|eval_realworld|collect_realworld)\.py'; then
    echo 'An embodiment driver is still running on this node. Stop it before resuming.' >&2
    exit 2
fi

cd "${REPO_DIR}"
set +u
source "${REPO_DIR}/pre_start_ray.sh"
set -u
export RAY_ADDRESS="${RAY_ADDRESS:-172.16.88.2:6379}"
exec bash "${SCRIPT_DIR}/run_co_training_real_world.sh" "${CONFIG_NAME}" \
    "runner.resume_dir=${RESUME_DIR}" runner.ckpt_path=null
