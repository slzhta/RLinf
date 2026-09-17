#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy=residual
selection=""
checkpoint=""
ckpt_root=""
step=""
episodes=40
episode_length=120
mode=deterministic
check_only=false

usage() {
    echo "Usage: bash $0 [--policy residual|cnn]"
    echo '       [--initial | --checkpoint FILE | --step N --ckpt-root DIR]'
    echo '       [--episodes N] [--episode-length N] [--sample] [--check-only]'
    echo 'Defaults: residual initial policy, 40 episodes, 120 steps, deterministic.'
    echo 'Residual initial = SFT OpenPI + BC gripper + zero arm residual mean.'
    echo 'CNN initial = original PnP BC step3000. --step 0 also selects initial.'
    echo 'Stop training and release its robot controller before actual evaluation.'
}
select_once() {
    if [[ -n "$selection" ]]; then echo 'Select only one policy source.' >&2; exit 2; fi
    selection="$1"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy) policy="${2:?Missing policy}"; shift 2 ;;
        --initial) select_once initial; shift ;;
        --checkpoint) select_once checkpoint; checkpoint="${2:?Missing checkpoint}"; shift 2 ;;
        --step) select_once step; step="${2:?Missing step}"; shift 2 ;;
        --ckpt-root) ckpt_root="${2:?Missing checkpoint root}"; shift 2 ;;
        --episodes) episodes="${2:?Missing episode count}"; shift 2 ;;
        --episode-length) episode_length="${2:?Missing episode length}"; shift 2 ;;
        --sample) mode=sample; shift ;;
        --check-only) check_only=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done
[[ "$policy" == residual || "$policy" == cnn ]] || { echo 'Policy must be residual or cnn.' >&2; exit 2; }
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || { echo 'Episodes must be positive.' >&2; exit 2; }
[[ "$episode_length" =~ ^[1-9][0-9]*$ ]] || { echo 'Episode length must be positive.' >&2; exit 2; }
if [[ "$selection" == step ]]; then
    [[ "$step" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Step must be a nonnegative integer.' >&2; exit 2; }
    if [[ "$step" != 0 ]]; then
        [[ -n "$ckpt_root" ]] || { echo 'Positive --step requires --ckpt-root.' >&2; exit 2; }
        checkpoint="${ckpt_root%/}/global_step_${step}/actor/model_state_dict/full_weights.pt"
    fi
elif [[ -n "$ckpt_root" ]]; then
    echo '--ckpt-root requires --step.' >&2; exit 2
fi
if [[ "$policy" == residual ]]; then
    config=real_eval_pick_and_place_openpi_residual_shared
else
    config=real_eval_pick_and_place_cnn_dual_state
    if [[ -z "$checkpoint" ]]; then
        checkpoint="$HOME/shiliangzhi/co-training/pretrained-model/pnp_clean_sim500_real250_bc_step3000.pt"
    fi
fi
if [[ -n "$checkpoint" ]]; then
    checkpoint="${checkpoint/#\~/$HOME}"
    [[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
    checkpoint="$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")"
else
    checkpoint=null
fi
echo "PnP eval: policy=$policy checkpoint=$checkpoint episodes=$episodes episode_length=$episode_length mode=$mode"
exec bash "${SCRIPT_DIR}/run_realworld_eval.sh" "$config" \
    "runner.ckpt_path=$checkpoint" \
    "evaluation.num_episodes=$episodes" \
    "evaluation.episode_length=$episode_length" \
    "evaluation.policy_mode=$mode" \
    "evaluation.check_only=$check_only"
