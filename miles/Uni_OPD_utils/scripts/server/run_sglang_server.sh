#!/bin/bash
# shellcheck disable=SC1090,SC1091,SC2034,SC2012,SC2015

CONDA_PATH="<YOUR PATH TO CONDA>"
CONDA_ENV_NAME="Uni-OPD"

# Teacher 模型，只会部署第一个找到的有效路径，其他的路径会被忽略
TEACHER_DIRS=(
    # 纯文本 Teacher 模型
    # Math
    # "<Your path to Qwen3-4B-Math-Teacher>"
    # Code
    "<Your path to Qwen3-4B-Code-Teacher>"
)

# 每个 GPU 对应一个 server，端口从 BASE_PORT 递增
GPU_LIST=(0 1 2 3 4 5 6 7)

BASE_PORT=13141
SERVER_GPU_UTILS="0.85"
CHUNKED_PREFILL_SIZE=4096
TP=1 # tensor parallel size（每个 server 占用的 GPU 数）
WAIT_SLEEP_TIME=20
WAIT_TIMEOUT=600

# Log 目录
MILES_PATH="<YOUR PATH TO UNI OPD>/miles"
LOG_BASE="${MILES_PATH}/outputs/logs/teacher_servers"
CUR_TIME=$(date +%Y%m%d%H%M%S)

resolve_dir() {
    for d in "$@"; do
        [ -d "$d" ] && {
            echo "$d"
            return 0
        }
    done
    echo "Error: No valid teacher model directory found" >&2
    exit 1
}

is_server_alive() {
    local url="$1"
    curl -sf "${url}/health_generate" >/dev/null 2>&1
}

start_server() {
    local gpu_id="$1"
    local port="$2"
    local teacher_dir="$3"
    local log_file="${LOG_DIR}/GPU_${gpu_id}_Port_${port}.log"

    echo "[GPU ${gpu_id}] Starting teacher server on port ${port}..."
    echo "[GPU ${gpu_id}] Model: ${teacher_dir}"
    echo "[GPU ${gpu_id}] Log: ${log_file}"

    nohup env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        python -m sglang.launch_server \
        --model-path "${teacher_dir}" \
        --host 0.0.0.0 \
        --port "${port}" \
        --tp "${TP}" \
        --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
        --mem-fraction-static "${SERVER_GPU_UTILS}" \
        >"${log_file}" 2>&1 &

    echo "[GPU ${gpu_id}] PID: $!"
    # 记录 pid 方便后续停止
    echo "$!" >>"${LOG_DIR}/pids.txt"
    sleep 1
}

wait_server() {
    local gpu_id="$1"
    local port="$2"
    local url="http://${LOCAL_IP}:${port}"
    local log_file="${LOG_DIR}/GPU_${gpu_id}_Port_${port}.log"

    echo "[GPU ${gpu_id}] Waiting for server on port ${port}..."
    for ((i = 1; i <= WAIT_TIMEOUT; i++)); do
        if is_server_alive "${url}"; then
            echo "[GPU ${gpu_id}] Server ready on port ${port}."
            return 0
        fi
        echo "[GPU ${gpu_id}] Not ready yet (${i}/${WAIT_TIMEOUT})..."
        [ -f "${log_file}" ] && tail -n 3 "${log_file}"
        sleep "${WAIT_SLEEP_TIME}"
    done
    echo "[GPU ${gpu_id}] ERROR: Server on port ${port} failed to start. Check log: ${log_file}"
    exit 1
}

set -e

if [ -d "${CONDA_PATH}" ]; then
    source "${CONDA_PATH}/bin/activate"
    echo "Success sourced conda from ${CONDA_PATH}"
else
    echo "Error: Failed to source conda at ${CONDA_PATH}/bin/activate"
    exit 1
fi

conda activate "${CONDA_ENV_NAME}" && echo "Activated conda env: ${CONDA_ENV_NAME}" || {
    echo "Error: Failed to activate conda env '${CONDA_ENV_NAME}'"
    exit 1
}

TEACHER_DIR=$(resolve_dir "${TEACHER_DIRS[@]}")
_basename=$(basename "${TEACHER_DIR}")
if [ "${_basename}" = "huggingface" ]; then
    # 去掉 /actor/huggingface 后缀，取最后两级目录用下划线连接
    # 例如 .../OPD_math_teacher/global_step_500/actor/huggingface -> OPD_math_teacher_global_step_500
    _stripped="${TEACHER_DIR%/actor/huggingface}"
    _parent=$(basename "$(dirname "${_stripped}")")
    _leaf=$(basename "${_stripped}")
    TEACHER_NAME="${_parent}_${_leaf}"
else
    TEACHER_NAME="${_basename}"
fi
LOG_DIR="${LOG_BASE}/${TEACHER_NAME}/${CUR_TIME}"
mkdir -p "${LOG_DIR}"
echo "Using teacher model: ${TEACHER_DIR}"
echo "Teacher name: ${TEACHER_NAME}"
echo "LOCAL_IP: ${LOCAL_IP}"
echo "GPU list: ${GPU_LIST[*]}"
echo "Base port: ${BASE_PORT}"
echo "Log dir: ${LOG_DIR}"
echo "================================================================"

for i in "${!GPU_LIST[@]}"; do
    gpu_id="${GPU_LIST[$i]}"
    port=$((BASE_PORT + i))
    url="http://${LOCAL_IP}:${port}"

    if is_server_alive "${url}"; then
        echo "[GPU ${gpu_id}] Server already running on port ${port}, skipping."
    else
        start_server "${gpu_id}" "${port}" "${TEACHER_DIR}"
    fi
done

echo ""
echo "All servers started, waiting for them to be ready..."
echo "================================================================"

for i in "${!GPU_LIST[@]}"; do
    gpu_id="${GPU_LIST[$i]}"
    port=$((BASE_PORT + i))
    wait_server "${gpu_id}" "${port}"
done

echo ""
echo "================================================================"
echo "All teacher servers are up!"

# 构造逗号分隔的完整 server 列表（供 _parse_urls 使用）
SERVER_URL_LIST=""
for i in "${!GPU_LIST[@]}"; do
    gpu_id="${GPU_LIST[$i]}"
    port=$((BASE_PORT + i))
    url="http://${LOCAL_IP}:${port}"
    echo "  GPU ${gpu_id} -> ${url}"
    curl -s "${url}/model_info" | python3 -m json.tool 2>/dev/null || true
    if [ -z "${SERVER_URL_LIST}" ]; then
        SERVER_URL_LIST="${url}"
    else
        SERVER_URL_LIST="${SERVER_URL_LIST},${url}"
    fi
done

echo "================================================================"
echo "SERVER_URL_LIST: ${SERVER_URL_LIST}"
echo "================================================================"
