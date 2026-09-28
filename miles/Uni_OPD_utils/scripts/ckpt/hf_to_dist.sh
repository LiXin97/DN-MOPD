#!/bin/bash
# shellcheck disable=SC1090,SC1091

# 模型设置
INPUT_DIR="<YOUR PATH TO HUGGINGFACE CKPT>"
OUTPUT_DIR="<YOUR PATH TO DIST CKPT>"

# 环境设置(不必修改)
CONDA_PATH="<YOUR PATH TO CONDA>"
CONDA_ENV_NAME="Uni-OPD"
MEGATRON_PATH="<YOUR PATH TO UNI OPD>/Megatron-LM"
MILES_PATH="<YOUR PATH TO UNI OPD>/miles"

# 自动推导 MODEL_ARGS_FILE
MODEL_NAME=$(basename "${INPUT_DIR}")

if echo "${MODEL_NAME}" | grep -q "VL"; then
    # VL 模型：Qwen3-VL-2B-Instruct → qwen3-1.7B.sh，rotary-base 5000000
    MODEL_ARGS_FILE=$(echo "${MODEL_NAME}" |
        sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
    # VL models require rotary-base 5000000
    MODEL_ARGS_ROTARY_BASE=5000000
else
    # 纯文本模型：Qwen3-4B → qwen3-4B.sh
    MODEL_ARGS_FILE=$(echo "${MODEL_NAME}" |
        sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-/qwen3-/g' |
        tr '[:upper:]' '[:lower:]')
    MODEL_ARGS_ROTARY_BASE=""
fi

MODEL_ARGS_PATH="scripts/models/${MODEL_ARGS_FILE}.sh"
MODEL_ARGS_FULL_PATH="${MILES_PATH}/${MODEL_ARGS_PATH}"

echo "[INFO] MODEL_NAME      : ${MODEL_NAME}"
echo "[INFO] MODEL_ARGS_FILE : ${MODEL_ARGS_FILE}.sh"
echo "[INFO] MODEL_ARGS_PATH : ${MODEL_ARGS_FULL_PATH}"
[ -n "${MODEL_ARGS_ROTARY_BASE}" ] &&
    echo "[INFO] ROTARY_BASE     : ${MODEL_ARGS_ROTARY_BASE}"

if [ -f "${MODEL_ARGS_FULL_PATH}" ]; then
    if [ -n "${MODEL_ARGS_ROTARY_BASE}" ]; then
        MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE}" source "${MODEL_ARGS_FULL_PATH}"
    else
        source "${MODEL_ARGS_FULL_PATH}"
    fi
else
    echo "Error: File '${MODEL_ARGS_FULL_PATH}' not found"
    exit 1
fi

if [ -d "${CONDA_PATH}" ]; then
    source "${CONDA_PATH}/bin/activate"
else
    echo "Error: Failed to source conda at ${CONDA_PATH}/bin/activate"
    exit 1
fi

conda activate "${CONDA_ENV_NAME}" || {
    echo "Error: Conda environment '$CONDA_ENV_NAME' not found or failed to activate"
    exit 1
}

export PYTHONPATH="${PYTHONPATH}:${MEGATRON_PATH}"
cd "${MILES_PATH}" || exit 1

python "tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${INPUT_DIR}" \
    --save "${OUTPUT_DIR}"
