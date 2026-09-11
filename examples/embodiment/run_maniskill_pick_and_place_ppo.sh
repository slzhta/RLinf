#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RLINF_VENV="${RLINF_VENV:-/home/ubuntu/wangyinghan/RLinf/.venv}"
MAX_STEPS="${MAX_STEPS:-3000}"

if [[ ! -x "${RLINF_VENV}/bin/python" ]]; then
    echo "RLinf Python not found: ${RLINF_VENV}/bin/python" >&2
    exit 1
fi

export VIRTUAL_ENV="${RLINF_VENV}"
export PATH="${RLINF_VENV}/bin:${PATH}"
export RAY_ADDRESS=local
export RLINF_NODE_RANK=0
export EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"
export RLINF_LOG_PATH="${REPO_ROOT}/results"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export ROBOT_PLATFORM=LIBERO
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BC_CHECKPOINT="${PNP_BC_CHECKPOINT:-}"
RESUME_DIR="${PNP_PPO_RESUME_DIR:-}"
RESUME_DIR="${RESUME_DIR%/}"

if [[ -n "${RESUME_DIR}" ]]; then
    if [[ ! -d "${RESUME_DIR}" ]]; then
        echo "PPO resume checkpoint not found: ${RESUME_DIR}" >&2
        exit 1
    fi
    if [[ ! -f "${RESUME_DIR}/actor/trainer_state.pt" ]]; then
        echo "This checkpoint has no current PPO training-state marker." >&2
        echo "Restart from a current BCE BC checkpoint instead of resuming it." >&2
        exit 1
    fi
    if ! "${RLINF_VENV}/bin/python" -c 'import sys, torch; state = torch.load(sys.argv[1], map_location="cpu", weights_only=True); assert state.get("format_version") == 2 and state.get("training_stage") == "ppo" and state.get("action_distribution_version") == 1 and state.get("continuous_action_distribution") == "tanh_normal_v1" and tuple(state.get("binary_action_indices", ())) == (6,) and float(state.get("binary_action_temperature")) == 1.0' "${RESUME_DIR}/actor/trainer_state.pt"; then
        echo "Checkpoint stage or action-distribution metadata is incompatible." >&2
        exit 1
    fi
    DEFAULT_RUN_DIR="$(dirname "$(dirname "${RESUME_DIR}")")"
    RUN_DIR="${1:-${DEFAULT_RUN_DIR}}"
    CHECKPOINT_OVERRIDES=(runner.resume_dir="${RESUME_DIR}" runner.ckpt_path=null)
else
    if [[ -z "${BC_CHECKPOINT}" ]]; then
        echo "Set PNP_BC_CHECKPOINT to a checkpoint trained with the current BCE BC config." >&2
        exit 1
    fi
    if [[ ! -f "${BC_CHECKPOINT}" ]]; then
        echo "BC checkpoint not found: ${BC_CHECKPOINT}" >&2
        exit 1
    fi
    BC_POLICY_STATE="$(dirname "$(dirname "${BC_CHECKPOINT}")")/bc_policy_state.pt"
    if [[ ! -f "${BC_POLICY_STATE}" ]]; then
        echo "BC checkpoint predates the BCE binary-action format: ${BC_CHECKPOINT}" >&2
        exit 1
    fi
    if ! "${RLINF_VENV}/bin/python" -c 'import sys, torch; state = torch.load(sys.argv[1], map_location="cpu", weights_only=True); assert state.get("format_version") == 2 and state.get("training_stage") == "bc" and state.get("data_format_version") == 2 and state.get("model_type") == "cnn_policy" and state.get("binary_loss") == "bce_with_logits" and state.get("continuous_action_distribution") == "tanh_normal_v1" and tuple(state.get("binary_action_indices", ())) == (6,) and float(state.get("binary_action_temperature")) == 1.0' "${BC_POLICY_STATE}"; then
        echo "BC checkpoint binary-action metadata does not match the PPO config." >&2
        exit 1
    fi
    RUN_DIR="${1:-${REPO_ROOT}/results/maniskill_pick_and_place_ppo_bc_sparse_v6/$(date +'%Y%m%d-%H%M%S')}"
    CHECKPOINT_OVERRIDES=(runner.resume_dir=null runner.ckpt_path="${BC_CHECKPOINT}")
fi

mkdir -p "${RUN_DIR}/video/train" "${RUN_DIR}/video/eval"
echo "Python: ${RLINF_VENV}/bin/python"
echo "PnP PPO run directory: ${RUN_DIR}"
if [[ -n "${RESUME_DIR}" ]]; then
    echo "PPO resume checkpoint: ${RESUME_DIR}"
else
    echo "BC initialization: ${BC_CHECKPOINT}"
fi

exec "${RLINF_VENV}/bin/python" -u "${EMBODIED_PATH}/train_embodied_agent.py" \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name maniskill_pick_and_place_ppo_cnn_state \
    "${CHECKPOINT_OVERRIDES[@]}" \
    runner.max_epochs="${MAX_STEPS}" \
    runner.max_steps="${MAX_STEPS}" \
    runner.logger.log_path="${RUN_DIR}" \
    runner.logger.experiment_name= \
    env.train.video_cfg.video_base_dir="${RUN_DIR}/video/train" \
    env.eval.video_cfg.video_base_dir="${RUN_DIR}/video/eval"
