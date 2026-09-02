#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_NAME="${1:-co_rl_pick_and_place_async_ppo_cnn_paired}"
CONFIG_NAME="${CONFIG_NAME%.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

required_env=(
    RAY_ADDRESS
    RLINF_CO_TRAINING_ASSETS
    RLINF_PNP_INITIAL_CKPT
    RLINF_FRANKA_IP
    RLINF_REAL_WRIST_CAMERA
    RLINF_REAL_POLICY_CAMERA
)
for env_name in "${required_env[@]}"; do
    if [[ -z "${!env_name:-}" ]]; then
        echo "Missing required environment variable: ${env_name}" >&2
        exit 2
    fi
done

CONFIG_FILE="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"
ASSET_ROOT="${RLINF_CO_TRAINING_ASSETS}"
required_files=(
    "${CONFIG_FILE}"
    "${RLINF_PNP_INITIAL_CKPT}"
    "${ASSET_ROOT}/VERSION"
    "${ASSET_ROOT}/manifest.sha256"
    "${ASSET_ROOT}/models/resnet10_pretrained.pt"
    "${ASSET_ROOT}/digital_twin/backgrounds/thirdview_background.png"
    "${ASSET_ROOT}/digital_twin/tables/table.glb"
    "${ASSET_ROOT}/digital_twin/robots/panda_umi.urdf"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 2
    fi
done

if ! (cd "${ASSET_ROOT}" && sha256sum --check manifest.sha256); then
    echo "Co-training asset manifest verification failed." >&2
    exit 2
fi

if ! timeout 10 ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    echo "Cannot connect to Ray at ${RAY_ADDRESS}." >&2
    exit 2
fi

if ! cluster_counts="$(python - <<'PY'
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR", log_to_driver=False)
nodes = [node for node in ray.nodes() if node["Alive"]]
print(len(nodes))
ray.shutdown()
PY
2>/dev/null)"; then
    echo "Failed to inspect Ray cluster nodes." >&2
    exit 2
fi
if [[ "${cluster_counts}" != "3" ]]; then
    echo "Expected 3 alive Ray nodes, found ${cluster_counts}." >&2
    exit 2
fi

export EMBODIED_PATH="${SCRIPT_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

log_root="${RLINF_LOG_PATH:-${REPO_DIR}/results}"
log_dir="${log_root}/${CONFIG_NAME}/$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${log_dir}"

cmd=(
    python "${SCRIPT_DIR}/train_async.py"
    --config-path "${SCRIPT_DIR}/config"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${log_dir}"
    "$@"
)
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "${log_dir}/train.log"
