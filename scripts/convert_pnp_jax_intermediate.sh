#!/usr/bin/env bash
set -euo pipefail

STEP="${1:?用法：bash scripts/convert_pnp_jax_intermediate.sh 10000}"

case "$STEP" in
    10000)
        LABEL=10k
        ;;
    20000)
        LABEL=20k
        ;;
    30000)
        LABEL=30k
        ;;
    *)
        echo "暂时仅允许转换 10000、20000 或 30000"
        exit 1
        ;;
esac

OPENPI=/home/shiliangzhi/work-space/openpi
JAX_EXP=/data/wangyinghan/openpi_checkpoints/pi05_franka/pi05_pnp_center_crop_500_40k_offline_v1
CKPT="$JAX_EXP/$STEP"
OUTPUT="/data/wangyinghan/pytorch_checkpoints/pi05_pnp_center_crop_500_${LABEL}"
BASE_40K=/data/wangyinghan/pytorch_checkpoints/pi05_pnp_center_crop_500_40k
REPO_ID=pnp_center_crop_train_500_lerobot_v2

echo "===== CONVERSION SETTINGS ====="
echo "STEP=$STEP"
echo "LABEL=$LABEL"
echo "OPENPI=$OPENPI"
echo "CKPT=$CKPT"
echo "OUTPUT=$OUTPUT"

test -x "$OPENPI/.venv/bin/python" || {
    echo "ERROR: OpenPI Python 不存在：$OPENPI/.venv/bin/python"
    exit 1
}

test -f "$OPENPI/examples/convert_jax_model_to_pytorch.py" || {
    echo "ERROR: 转换脚本不存在"
    exit 1
}

test -d "$CKPT/params" || {
    echo "ERROR: params 目录不存在：$CKPT/params"
    exit 1
}

test -s "$CKPT/_CHECKPOINT_METADATA" || {
    echo "ERROR: Orbax checkpoint metadata 不存在或为空"
    exit 1
}

TMP_COUNT=$(
    find "$JAX_EXP" \
        -maxdepth 1 \
        -type d \
        -name "${STEP}.orbax-checkpoint-tmp-*" \
    | wc -l
)

if [[ "$TMP_COUNT" -ne 0 ]]; then
    echo "ERROR: 该 checkpoint 仍存在 Orbax 临时目录"
    exit 1
fi

echo
echo "===== CHECK EXISTING OUTPUT ====="

if test -s "$OUTPUT/model.safetensors" &&
   test -s "$OUTPUT/config.json"; then
    echo "转换结果已经存在，复用：$OUTPUT"
elif test -e "$OUTPUT"; then
    BACKUP="${OUTPUT}.incomplete_$(date +%Y%m%d_%H%M%S)"
    echo "发现不完整输出，将其移动为：$BACKUP"
    mv "$OUTPUT" "$BACKUP"
else
    echo "输出目录不存在，可以开始转换"
fi

if ! test -s "$OUTPUT/model.safetensors"; then
    echo
    echo "===== CONVERT JAX TO PYTORCH ====="

    cd "$OPENPI"

    CUDA_VISIBLE_DEVICES="" \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$OPENPI/.venv/bin/python" -u \
        examples/convert_jax_model_to_pytorch.py \
        --checkpoint-dir "$CKPT" \
        --config-name pi05_franka \
        --output-path "$OUTPUT" \
        --precision bfloat16
fi

echo
echo "===== INSTALL NORM STATS ====="

NORM_SOURCE=""

for candidate in \
    "$OUTPUT/assets/$REPO_ID/norm_stats.json" \
    "$JAX_EXP/assets/$REPO_ID/norm_stats.json" \
    "$BASE_40K/assets/$REPO_ID/norm_stats.json" \
    "$BASE_40K/$REPO_ID/norm_stats.json"
do
    if test -s "$candidate"; then
        NORM_SOURCE="$candidate"
        break
    fi
done

if [[ -z "$NORM_SOURCE" ]]; then
    echo "ERROR: 找不到 norm_stats.json"
    exit 1
fi

echo "NORM_SOURCE=$NORM_SOURCE"

mkdir -p \
    "$OUTPUT/assets/$REPO_ID" \
    "$OUTPUT/$REPO_ID"

cp -a \
    "$NORM_SOURCE" \
    "$OUTPUT/assets/$REPO_ID/norm_stats.json"

cp -a \
    "$NORM_SOURCE" \
    "$OUTPUT/$REPO_ID/norm_stats.json"

echo
echo "===== VALIDATE OUTPUT ====="

for file in \
    "$OUTPUT/model.safetensors" \
    "$OUTPUT/config.json" \
    "$OUTPUT/assets/$REPO_ID/norm_stats.json" \
    "$OUTPUT/$REPO_ID/norm_stats.json"
do
    if test -s "$file"; then
        ls -lh "$file"
    else
        echo "ERROR: 文件缺失或为空：$file"
        exit 1
    fi
done

echo
echo "===== CONFIG ====="
cat "$OUTPUT/config.json"

echo
echo "===== SAFETENSORS ====="

"$OPENPI/.venv/bin/python" - "$OUTPUT/model.safetensors" <<'PY'
from safetensors import safe_open
import sys

path = sys.argv[1]

with safe_open(path, framework="pt", device="cpu") as file:
    keys = list(file.keys())
    print("model:", path)
    print("tensor count:", len(keys))
    print("first key:", keys[0])
    print("last key:", keys[-1])
PY

echo
echo "转换成功：$OUTPUT"
