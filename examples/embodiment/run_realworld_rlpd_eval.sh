#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="real_eval_peg_insertion_rlpd_cnn_wrist_state"
EPISODES=40
EPISODE_LENGTH=240
POLICY_MODE=deterministic
CHECK_ONLY=false
CHECKPOINT=""
usage() {
    echo "Usage: bash $0 [--checkpoint FILE_OR_STEP_DIR] [--episodes N] [--episode-length N] [--policy-mode deterministic|sample] [--check-only]"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --checkpoint|--episodes|--episode-length|--policy-mode)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                usage >&2
                exit 2
            fi
            case "$1" in
                --checkpoint) CHECKPOINT="$2" ;;
                --episodes) EPISODES="$2" ;;
                --episode-length) EPISODE_LENGTH="$2" ;;
                --policy-mode) POLICY_MODE="$2" ;;
            esac
            shift 2
            ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ || ! "${EPISODE_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Episode count and length must be positive integers." >&2
    exit 2
fi
if [[ "${POLICY_MODE}" != deterministic && "${POLICY_MODE}" != sample ]]; then
    echo "Policy mode must be deterministic or sample." >&2
    exit 2
fi
overrides=(
    "evaluation.num_episodes=${EPISODES}"
    "evaluation.episode_length=${EPISODE_LENGTH}"
    "evaluation.policy_mode=${POLICY_MODE}"
    "evaluation.check_only=${CHECK_ONLY}"
)
if [[ -n "${CHECKPOINT}" ]]; then
    if [[ -d "${CHECKPOINT}" ]]; then
        CHECKPOINT="${CHECKPOINT%/}/actor/model_state_dict/full_weights.pt"
    fi
    if [[ ! -s "${CHECKPOINT}" ]]; then
        echo "Missing or empty checkpoint: ${CHECKPOINT}" >&2
        exit 2
    fi
    CHECKPOINT="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"
    overrides+=("runner.ckpt_path=${CHECKPOINT}")
fi
printf 'RLPD evaluation: %s episodes, at most %s steps each, policy=%s\n' \
    "${EPISODES}" "${EPISODE_LENGTH}" "${POLICY_MODE}"
exec bash "${SCRIPT_DIR}/run_realworld_eval.sh" "${CONFIG_NAME}" "${overrides[@]}"
