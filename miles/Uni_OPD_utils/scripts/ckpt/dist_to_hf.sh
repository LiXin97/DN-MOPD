#!/bin/bash
# shellcheck disable=SC1090,SC1091

# 需要转换的 dist ckpt 路径
DEFAULT_INPUT_DIST_DIR="<YOUR PATH TO DIST CKPT>"
# 对应的原始 huggingface 模型路径
DEFAULT_ORIGIN_MODEL_DIR="<YOUR PATH TO HUGGINGFACE CKPT>"
# 转换后的 huggingface 模型路径
DEFAULT_OUTPUT_DIR="<YOUR PATH TO OUTPUT CKPT>"

# 支持外部传参覆盖默认路径
INPUT_DIST_DIR="${1:-${DEFAULT_INPUT_DIST_DIR}}"
ORIGIN_MODEL_DIR="${2:-${DEFAULT_ORIGIN_MODEL_DIR}}"
OUTPUT_DIR="${3:-${DEFAULT_OUTPUT_DIR}}"

# 环境设置
CONDA_PATH="<YOUR PATH TO CONDA>"
CONDA_ENV_NAME="Uni-OPD"
MEGATRON_PATH="<YOUR PATH TO UNI OPD>/Megatron-LM"
CONVERT_SCRIPT="<YOUR PATH TO UNI OPD>/miles/tools/convert_torch_dist_to_hf.py"

if [ -z "${INPUT_DIST_DIR}" ]; then
    echo "Error: Input dist dir is empty"
    exit 1
fi
if [ -z "${ORIGIN_MODEL_DIR}" ]; then
    echo "Error: Origin model dir is empty"
    exit 1
fi
if [ -z "${OUTPUT_DIR}" ]; then
    echo "Error: Output dir is empty"
    exit 1
fi

shift 3

if [ -d "${CONDA_PATH}" ]; then
    source "${CONDA_PATH}/bin/activate"
else
    echo "Error: Failed to source conda at ${CONDA_PATH}/bin/activate"
    exit 1
fi

conda activate "${CONDA_ENV_NAME}" || {
    echo "Error: Conda environment '${CONDA_ENV_NAME}' not found or failed to activate"
    exit 1
}

export PYTHONPATH="${PYTHONPATH}:${MEGATRON_PATH}"
python "${CONVERT_SCRIPT}" \
    --input-dir "${INPUT_DIST_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --origin-hf-dir "${ORIGIN_MODEL_DIR}"
