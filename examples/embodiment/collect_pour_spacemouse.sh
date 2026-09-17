#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EPISODES="${1:-2}"
RUN_NAME="${2:-$(date +'%Y%m%d-%H%M%S')}"
if (( $# > 2 )); then shift 2; else set --; fi
cd "${REPO_DIR}"
source "${REPO_DIR}/pre_start_ray.sh"
set -u
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
exec python "${SCRIPT_DIR}/collect_pour_spacemouse.py" \
    --episodes "${EPISODES}" --run-name "${RUN_NAME}" "$@"
