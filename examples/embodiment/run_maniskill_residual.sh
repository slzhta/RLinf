#!/usr/bin/env bash
# Simulation-only launcher using the standard embodied PPO entry point.
set -euo pipefail


export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="${1:-maniskill_pick_and_place_ppo_residual_a4}"
if [[ $# -gt 0 ]]; then
    shift
fi
# Use this repository's runtime without requiring manual venv activation.
PYTHON_BIN="${REPO_PATH}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python)"
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

# Read launch settings before importing CUDA/Ray. shlex.quote prevents shell
# expansion of YAML values; Hydra overrides are applied before extracting them.
LAUNCH_ENV="$("${PYTHON_BIN}" - "${EMBODIED_PATH}/config" "${CONFIG_NAME}" "$@" <<'PYENV'
import re
import shlex
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(config_dir=sys.argv[1], version_base="1.1"):
    cfg = compose(config_name=sys.argv[2], overrides=sys.argv[3:])
settings = OmegaConf.to_container(cfg.get("launch", OmegaConf.create({})), resolve=True)
for key in settings.get("unset_env_vars", []):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise ValueError(f"Invalid environment variable: {key}")
    print(f"unset {key}")
for key, value in settings.get("env_vars", {}).items():
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise ValueError(f"Invalid environment variable: {key}")
    print(f"export {key}={shlex.quote(str(value))}")
PYENV
)"
eval "${LAUNCH_ENV}"
cd "${REPO_PATH}"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
"${PYTHON_BIN}" "${EMBODIED_PATH}/train_embodied_agent.py" \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name "${CONFIG_NAME}" \
    "runner.logger.log_path=${LOG_DIR}" "$@" 2>&1 | tee "${LOG_DIR}/run.log"
