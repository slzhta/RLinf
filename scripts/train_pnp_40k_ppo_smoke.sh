#!/usr/bin/env bash
set -euo pipefail

cd /home/shiliangzhi/work-space/wangyinghan/RLinf
source .venv/bin/activate

export EMBODIED_PATH=/home/shiliangzhi/work-space/wangyinghan/RLinf/examples/embodiment
export PYTHONPATH=/home/shiliangzhi/work-space/wangyinghan/RLinf:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HYDRA_FULL_ERROR=1
unset RLINF_TRACE_PNP

RESULT_DIR=/data/wangyinghan/results/pi05_pnp_center_crop_40k_ppo_smoke_v2
mkdir -p "${RESULT_DIR}"

python examples/embodiment/train_embodied_agent.py \
    --config-name maniskill_pnp_ppo_pi05_center_crop_40k \
    runner.max_epochs=3 \
    runner.val_check_interval=1 \
    runner.save_interval=1 \
    runner.logger.log_path="${RESULT_DIR}" \
    runner.logger.experiment_name=pi05_pnp_center_crop_40k_ppo_smoke_v2 \
    env.train.total_num_envs=8 \
    env.eval.total_num_envs=8 \
    env.eval.video_cfg.save_video=false \
    actor.micro_batch_size=8 \
    actor.global_batch_size=32 \
    2>&1 | tee "${RESULT_DIR}/train_terminal.log"
