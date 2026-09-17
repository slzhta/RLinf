#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config=real_eval_peg_insertion_openpi_residual_wrist_state
checkpoint=""
step=""
ckpt_root=""
selection=""
episodes=40
mode=deterministic
check_only=false

usage() {
    echo "Usage: bash $0 [--initial | --checkpoint FILE | --step N --ckpt-root DIR]"
    echo '       [--episodes N] [--sample] [--check-only] [--config YAML_NAME]'
    echo 'Default: initial residual, 40 episodes, residual mean actions; OpenPI remains loaded.'
    echo '--step 0 selects the initial residual. Positive steps require --ckpt-root.'
}
select_once() {
    if [[ -n "$selection" ]]; then echo 'Select only one policy source.' >&2; exit 2; fi
    selection="$1"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --initial) select_once initial; shift ;;
        --checkpoint) select_once checkpoint; checkpoint="${2:?Missing checkpoint path}"; shift 2 ;;
        --step) select_once step; step="${2:?Missing step}"; shift 2 ;;
        --ckpt-root) ckpt_root="${2:?Missing checkpoint root}"; shift 2 ;;
        --episodes) episodes="${2:?Missing episode count}"; shift 2 ;;
        --config) config="${2:?Missing config name}"; shift 2 ;;
        --sample) mode=sample; shift ;;
        --check-only) check_only=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || { echo 'Episodes must be positive.' >&2; exit 2; }
if [[ "$selection" == step ]]; then
    [[ "$step" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Step must be a nonnegative integer.' >&2; exit 2; }
    if [[ "$step" != 0 ]]; then
        [[ -n "$ckpt_root" ]] || { echo 'Positive --step requires --ckpt-root.' >&2; exit 2; }
        checkpoint="${ckpt_root%/}/global_step_${step}/actor/model_state_dict/full_weights.pt"
    fi
elif [[ -n "$ckpt_root" ]]; then
    echo '--ckpt-root requires --step.' >&2; exit 2
fi
if [[ -n "$checkpoint" ]]; then
    checkpoint="${checkpoint/#\~/$HOME}"
    [[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
    checkpoint="$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")"
    echo "Residual checkpoint: $checkpoint"
else
    checkpoint=null
    echo 'Initial residual (no RL checkpoint); the configured SFT OpenPI base is still loaded.'
fi
exec bash "${SCRIPT_DIR}/run_realworld_eval.sh" "$config" \
    "runner.ckpt_path=$checkpoint" \
    "evaluation.num_episodes=$episodes" \
    "evaluation.policy_mode=$mode" \
    "evaluation.check_only=$check_only"
