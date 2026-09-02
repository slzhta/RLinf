#!/usr/bin/env bash
set -e

cd /home/ubuntu/wangyitao/co-training/RLinf
source pre_start_ray.sh

export EMBODIED_PATH=/home/ubuntu/wangyitao/co-training/RLinf/examples/embodiment
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1

CONFIG_NAME=realworld_push_button_async_ppo_cnn_state
RUN_ID=$(date +'%Y%m%d-%H%M%S')
LOG_DIR=${RLINF_LOG_PATH}/${CONFIG_NAME}/${RUN_ID}

mkdir -p "${LOG_DIR}"
echo "Saving this real-world-only run to: ${LOG_DIR}"

python examples/embodiment/train_async.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name ${CONFIG_NAME} \
  runner.logger.log_path=${LOG_DIR} \
  actor.model.model_path=/mnt/RLinf \
  rollout.model.model_path=/mnt/RLinf \
  2>&1 | tee "${LOG_DIR}/train.log"
