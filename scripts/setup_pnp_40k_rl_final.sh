#!/usr/bin/env bash
set -euo pipefail

cd /home/shiliangzhi/work-space/wangyinghan/RLinf

SOURCE_CONFIG="examples/embodiment/config/maniskill_pnp_ppo_pi05_center_crop_2k.yaml"
TARGET_CONFIG="examples/embodiment/config/maniskill_pnp_ppo_pi05_center_crop_40k_rl_lr2e6_u2.yaml"
ENV_CONFIG="examples/embodiment/config/env/maniskill_pick_and_place_co_rl.yaml"

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
    echo "缺少模板配置：${SOURCE_CONFIG}"
    exit 1
fi

if [[ ! -f "${ENV_CONFIG}" ]]; then
    echo "缺少环境配置：${ENV_CONFIG}"
    exit 1
fi

# 备份已有的最终配置
if [[ -f "${TARGET_CONFIG}" ]]; then
    BACKUP="${TARGET_CONFIG}.bak_$(date +%Y%m%d_%H%M%S)"
    cp -a "${TARGET_CONFIG}" "${BACKUP}"
    echo "已备份原配置：${BACKUP}"
fi

# 每次都从干净模板重新生成
cp -a "${SOURCE_CONFIG}" "${TARGET_CONFIG}"

python - "${TARGET_CONFIG}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

def replace_once(old: str, new: str):
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"要求恰好匹配一次，实际匹配 {count} 次：{old!r}"
        )
    text = text.replace(old, new, 1)

# 日志和实验名称
replace_once(
    "    log_path: /data/wangyinghan/results/pi05_pnp_center_crop_2k_ppo\n",
    "    log_path: /data/wangyinghan/results/"
    "pi05_pnp_center_crop_40k_rl_lr2e6_u2\n",
)

replace_once(
    "    experiment_name: pi05_pnp_center_crop_2k_ppo\n",
    "    experiment_name: pi05_pnp_center_crop_40k_rl_lr2e6_u2\n",
)

# 正式训练长度、验证和保存间隔
replace_once(
    "  max_epochs: 1000\n",
    "  max_epochs: 100\n",
)

replace_once(
    "  val_check_interval: 1\n",
    "  val_check_interval: 25\n",
)

replace_once(
    "  save_interval: 10\n",
    "  save_interval: 50\n",
)

# 每次验证约 16 × 7 = 112 条轨迹
replace_once(
    "  eval_rollout_epoch: 1\n",
    "  eval_rollout_epoch: 7\n",
)

# 每批 rollout 做两轮 PPO 更新
replace_once(
    "  update_epoch: 1\n",
    "  update_epoch: 2\n",
)

# 40K SFT checkpoint，rollout 与 actor 两处一起替换
old_model = (
    "/data/wangyinghan/pytorch_checkpoints/"
    "pi05_pnp_center_crop_500_2k"
)
new_model = (
    "/data/wangyinghan/pytorch_checkpoints/"
    "pi05_pnp_center_crop_500_40k"
)

if text.count(old_model) != 2:
    raise SystemExit(
        f"预期找到两处旧模型路径，实际找到 {text.count(old_model)} 处"
    )

text = text.replace(old_model, new_model)

# 动作不经过 WidowX 的 0/1 夹爪转换
replace_once(
    "    policy_setup: widowx_bridge\n",
    "    policy_setup: panda_pnp\n",
)

# 提高策略学习速度
replace_once(
    "    lr: 1.0e-06\n",
    "    lr: 2.0e-06\n",
)

# flow_noise 的标准差范围
noise_anchor = "      noise_method: flow_noise\n"
replace_once(
    noise_anchor,
    noise_anchor + "      noise_logvar_range: [0.08, 0.16]\n",
)

# 只修改 eval block，保留 train block 的配置
eval_marker = "  eval:\n"

if text.count(eval_marker) != 1:
    raise SystemExit("eval block 数量异常")

before_eval, eval_block = text.split(eval_marker, 1)

if eval_block.count("    total_num_envs: 8\n") < 1:
    raise SystemExit("eval total_num_envs 不是预期的 8")

eval_block = eval_block.replace(
    "    total_num_envs: 8\n",
    "    total_num_envs: 16\n",
    1,
)

if "    use_fixed_reset_state_ids: False\n" not in eval_block:
    raise SystemExit("eval use_fixed_reset_state_ids 配置缺失")

eval_block = eval_block.replace(
    "    use_fixed_reset_state_ids: False\n",
    "    use_fixed_reset_state_ids: True\n",
    1,
)

if "      save_video: True\n" not in eval_block:
    raise SystemExit("eval save_video 配置缺失")

eval_block = eval_block.replace(
    "      save_video: True\n",
    "      save_video: False\n",
    1,
)

text = before_eval + eval_marker + eval_block

# 明确禁止从旧 RL 状态恢复
if "  resume_dir: null\n" not in text:
    raise SystemExit("resume_dir 不是 null")

path.write_text(text)

print(f"已生成最终配置：{path}")
PY

# 检查环境 Z 下界修复，若不存在则插入
python - "${ENV_CONFIG}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

target = (
    "    target_ee_pose: "
    "[0.54515648, 0.03448609, -0.01847301, "
    "3.07488508, -0.01941186, -0.0446862]\n"
)

limit_min = (
    "    ee_pose_limit_min: "
    "[0.39515648, -0.11551391, -0.04347301, "
    "3.06488508, -0.02941186, -0.5446862]\n"
)

limit_max = (
    "    ee_pose_limit_max: "
    "[0.69515648, 0.18448609, 0.13152699, "
    "3.08488508, -0.00941186, 0.4553138]\n"
)

if limit_min in text and limit_max in text:
    print("环境 EE 限位已经正确，无需重复修改")
elif "ee_pose_limit_min:" in text or "ee_pose_limit_max:" in text:
    raise SystemExit("环境中存在其他 EE 限位，请人工检查")
else:
    if text.count(target) != 1:
        raise SystemExit("无法找到 target_ee_pose，未修改环境")
    text = text.replace(
        target,
        target + limit_min + limit_max,
        1,
    )
    path.write_text(text)
    print("已补充环境 EE 限位")
PY

echo
echo "===== FINAL RL CONFIG ====="

grep -nE \
'component_placement|log_path|experiment_name|max_epochs|max_steps|only_eval|val_check_interval|save_interval|resume_dir|rollout_epoch|eval_rollout_epoch|update_epoch|loss_type|total_num_envs|use_fixed_reset_state_ids|max_episode_steps|save_video|unnorm_key|model_path|micro_batch_size|global_batch_size|add_value_head|config_name|num_steps|noise_method|noise_logvar_range|noise_params|joint_logprob|policy_setup|entropy_bonus|lr:|value_lr' \
"${TARGET_CONFIG}"

echo
echo "===== ENV SAFETY AND GRIPPER CONFIG ====="

grep -nE \
'target_ee_pose|ee_pose_limit_min|ee_pose_limit_max|action_scale|binary_gripper_action|binary_gripper_threshold|open_command|close_command' \
"${ENV_CONFIG}"

echo
echo "===== BATCH CHECK ====="

python - <<'PY'
train_envs = 16
episode_steps = 120
action_chunk = 10
global_batch = 64
eval_envs = 16
eval_epochs = 7

rollout_size = train_envs * episode_steps // action_chunk

print("train rollout size:", rollout_size)
print("global batch size:", global_batch)
print("optimizer batches:", rollout_size // global_batch)
print("batch divisible:", rollout_size % global_batch == 0)
print("approx eval trajectories:", eval_envs * eval_epochs)

assert rollout_size % global_batch == 0
PY

echo
echo "配置生成及检查完成。"
