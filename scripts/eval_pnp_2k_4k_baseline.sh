#!/usr/bin/env bash
set -euo pipefail

cd /home/shiliangzhi/work-space/wangyinghan/RLinf
source .venv/bin/activate

export EMBODIED_PATH=/home/shiliangzhi/work-space/wangyinghan/RLinf/examples/embodiment
export PYTHONPATH=/home/shiliangzhi/work-space/wangyinghan/RLinf:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HYDRA_FULL_ERROR=1
unset RLINF_TRACE_PNP

run_eval() {
    local step="$1"
    local config="$2"
    local result_dir="/data/wangyinghan/results/pi05_pnp_${step}_fixed_env_baseline"
    local log="${result_dir}/eval_terminal.log"

    mkdir -p "${result_dir}"

    echo
    echo "========================================"
    echo "Evaluating ${step} SFT checkpoint"
    echo "Config: ${config}"
    echo "========================================"

    python examples/embodiment/eval_embodied_agent.py \
        --config-name "${config}" \
        env.train.total_num_envs=1 \
        env.eval.total_num_envs=8 \
        env.eval.video_cfg.save_video=false \
        algorithm.eval_rollout_epoch=7 \
        actor.model.policy_setup=panda_pnp \
        runner.logger.experiment_name="pi05_pnp_${step}_fixed_env_baseline" \
        runner.logger.log_path="${result_dir}" \
        2>&1 | tee "${log}"

    echo
    echo "===== ${step} FINAL METRICS ====="

    grep -E \
    "'eval/success_once'|'eval/fail_once'|'eval/return'|'eval/episode_len'|'eval/num_trajectories'" \
    "${log}" \
    | tail -n 10
}

run_eval \
"2k" \
"maniskill_pnp_pi05_center_crop_2k_pure_eval"

ray stop >/dev/null 2>&1 || true

run_eval \
"4k" \
"maniskill_pnp_pi05_center_crop_4k_pure_eval"

echo
echo "========================================"
echo "2K AND 4K SUMMARY"
echo "========================================"

for step in 2k 4k; do
    log="/data/wangyinghan/results/pi05_pnp_${step}_fixed_env_baseline/eval_terminal.log"

    echo
    echo "===== ${step} ====="

    grep -E \
    "'eval/success_once'|'eval/fail_once'|'eval/return'|'eval/episode_len'|'eval/num_trajectories'" \
    "${log}" \
    | tail -n 5
done
