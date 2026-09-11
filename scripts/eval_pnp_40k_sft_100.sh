#!/usr/bin/env bash
set -euo pipefail

cd /home/shiliangzhi/work-space/wangyinghan/RLinf
source .venv/bin/activate

export EMBODIED_PATH=/home/shiliangzhi/work-space/wangyinghan/RLinf/examples/embodiment
export PYTHONPATH=/home/shiliangzhi/work-space/wangyinghan/RLinf:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HYDRA_FULL_ERROR=1

# 正式评测关闭逐步调试输出
unset RLINF_TRACE_PNP

CONFIG=maniskill_pnp_pi05_center_crop_40k_pure_eval
RESULT_DIR=/data/wangyinghan/results/pi05_pnp_40k_sft_eval_100
RUN_LOG="${RESULT_DIR}/eval_terminal.log"

mkdir -p "${RESULT_DIR}"

echo "========================================"
echo "PnP 40K SFT simulation evaluation"
echo "config:       ${CONFIG}"
echo "trajectories: 100"
echo "parallel envs: 10"
echo "eval epochs:   10"
echo "result dir:    ${RESULT_DIR}"
echo "========================================"

python examples/embodiment/eval_embodied_agent.py \
    --config-name "${CONFIG}" \
    env.train.total_num_envs=1 \
    env.eval.total_num_envs=10 \
    env.eval.group_size=1 \
    env.eval.video_cfg.save_video=false \
    algorithm.eval_rollout_epoch=10 \
    runner.logger.experiment_name=pi05_pnp_40k_sft_eval_100 \
    runner.logger.log_path="${RESULT_DIR}" \
    2>&1 | tee "${RUN_LOG}"

echo
echo "========================================"
echo "Final evaluation records"
echo "========================================"

grep -E \
"'eval/success_once'|'eval/num_trajectories'|'eval/return'|'eval/episode_len'" \
"${RUN_LOG}" \
| tail -n 20
