#!/bin/bash
# shellcheck disable=SC1091

# Usage:
#   bash start_ray.sh                      # 使用 NODE_IP_LIST 动态生成 slave hosts
#   bash start_ray.sh <HOST_NUM> <INDEX>   # 使用 $HOME/hosts/pssh.hosts_${HOST_NUM}node_${INDEX}，排除本机
#   bash start_ray.sh local                # 仅本机启动（单机模式）
#   bash start_ray.sh [local|NUM INDEX] [EXTRA_ENV...]

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MAX_RAY_WAIT_TIME=60
HOSTS_DIR="$HOME/hosts"
mkdir -p "$HOSTS_DIR"

CONDA_PATH="<YOUR PATH TO CONDA>"
CONDA_ENV_NAME="Uni-OPD"

ARG1="$1"
ARG2="$2"

# Conda 激活（本机）
CONDA_INIT_CMD=""
if [ -f "${CONDA_PATH}/bin/activate" ]; then
    source "${CONDA_PATH}/bin/activate"
    if conda activate "${CONDA_ENV_NAME}"; then
        echo "Conda env '${CONDA_ENV_NAME}' activated on head node."
    else
        echo "Error: failed to activate conda env '${CONDA_ENV_NAME}'"
        exit 1
    fi
    CONDA_INIT_CMD="source ${CONDA_PATH}/bin/activate && conda activate ${CONDA_ENV_NAME};"
else
    echo "Error: conda not found at ${CONDA_PATH}"
    exit 1
fi

# PSSH_SLAVE_HOSTS_FILE：仅包含 slave 节点，不修改完整的 pssh.hosts
if [ "$ARG1" = "local" ]; then
    # 单机模式：不需要 slave
    EXTRA_ENV="${*:2}"
    USE_PSSH=false
    echo "单机模式，仅启动本机"

elif [ -n "$ARG1" ] && [ -n "$ARG2" ] && [[ "$ARG1" =~ ^[0-9]+$ ]] && [[ "$ARG2" =~ ^[0-9]+$ ]]; then
    # 指定节点数和索引：从完整 hosts 文件中排除本机，生成临时 slave hosts
    FULL_HOSTS_FILE="$HOSTS_DIR/pssh.hosts_${ARG1}node_${ARG2}"
    PSSH_SLAVE_HOSTS_FILE="$HOSTS_DIR/pssh.hosts_${ARG1}node_${ARG2}_slave"
    echo "使用完整 hosts 文件: ${FULL_HOSTS_FILE}"

    if [ ! -f "${FULL_HOSTS_FILE}" ]; then
        echo "Error: hosts file '${FULL_HOSTS_FILE}' not found"
        exit 1
    fi

    # 排除本机 IP，生成 slave hosts 文件
    grep -v "^${LOCAL_IP}$" "${FULL_HOSTS_FILE}" >"${PSSH_SLAVE_HOSTS_FILE}"
    echo "生成 slave hosts 文件: ${PSSH_SLAVE_HOSTS_FILE}"
    cat "${PSSH_SLAVE_HOSTS_FILE}"

    EXTRA_ENV="${*:3}"
    USE_PSSH=true

else
    # 默认：从 NODE_IP_LIST 动态生成 slave hosts
    PSSH_SLAVE_HOSTS_FILE=$(mktemp "$HOSTS_DIR/pssh.hosts_slave_XXXXXX")
    echo "$NODE_IP_LIST" |
        sed 's/:[^,]*//g' |
        sed 's/,/\n/g' |
        sort -u |
        grep -v "^${LOCAL_IP}$" \
            >"${PSSH_SLAVE_HOSTS_FILE}"
    echo "动态生成 slave hosts 文件: ${PSSH_SLAVE_HOSTS_FILE}"
    cat "${PSSH_SLAVE_HOSTS_FILE}"

    EXTRA_ENV="${*:1}"
    USE_PSSH=true
fi

#! master node
pip install -U ray --root-user-action ignore
if [ -n "$EXTRA_ENV" ]; then
    echo "set ${EXTRA_ENV} for ray head node process"
    eval "${EXTRA_ENV}" ray start --head --node-ip-address "${LOCAL_IP}" --num-gpus 8 --disable-usage-stats
else
    ray start --head --node-ip-address "${LOCAL_IP}" --num-gpus 8 --disable-usage-stats
fi

if [[ "${CI_TEST,,}" == "true" ]]; then
    CI_TEST=true
    DETERMINISTIC_MODE=true
else
    CI_TEST=false
fi

#! slave node
if [ "$USE_PSSH" = true ]; then
    if [ -n "$EXTRA_ENV" ]; then
        echo "set ${EXTRA_ENV} for ray slave node process"
        pssh -i -t 0 -h "${PSSH_SLAVE_HOSTS_FILE}" \
            "export CI_TEST=${CI_TEST} DETERMINISTIC_MODE=${DETERMINISTIC_MODE}; \
            ${CONDA_INIT_CMD} \
            source ${SCRIPT_DIR}/network_envs.sh; \
            eval \"$EXTRA_ENV\" ray start --address ${LOCAL_IP}:6379 --num-gpus 8"
    else
        pssh -i -t 0 -h "${PSSH_SLAVE_HOSTS_FILE}" \
            "export CI_TEST=${CI_TEST} DETERMINISTIC_MODE=${DETERMINISTIC_MODE}; \
            ${CONDA_INIT_CMD} \
            source ${SCRIPT_DIR}/network_envs.sh; \
            ray start --address ${LOCAL_IP}:6379 --num-gpus 8"
    fi
else
    echo "单机模式，跳过 slave 节点启动"
fi

# 等待 Ray 就绪
start_time=$(date +%s)
while true; do
    # Use ray status command to check if Ray is running
    if ray status >/dev/null 2>&1; then
        echo "Ray has been successfully started"
        break
    fi

    # Check if the maximum waiting time has been exceeded
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -ge $MAX_RAY_WAIT_TIME ]; then
        echo "Ray startup timed out, please check if Ray is correctly installed and configured"
        exit 1
    fi

    echo "Waiting for Ray to start..."
    sleep 3
done
echo "Ray has been successfully started, continue to execute the subsequent commands"
