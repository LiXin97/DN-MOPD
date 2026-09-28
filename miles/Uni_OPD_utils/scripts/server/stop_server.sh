#!/bin/bash
# shellcheck disable=SC1091

# Usage:
#   bash stop_server.sh                      # 使用 $HOME/hosts/pssh.hosts
#   bash stop_server.sh <HOST_NUM> <INDEX>   # 使用 hosts/pssh.hosts_${HOST_NUM}node_${INDEX}
#   bash stop_server.sh local                # 使用本机 LOCAL_IP

set -e

HOSTS_DIR="$HOME/hosts"
mkdir -p "$HOSTS_DIR"

ARG1="$1"
ARG2="$2"

if [ "$ARG1" = "local" ]; then
    PSSH_HOSTS_FILE="$HOSTS_DIR/pssh.hosts_local"
    echo "$LOCAL_IP" >"${PSSH_HOSTS_FILE}"
    echo "使用本机 hosts 文件: ${PSSH_HOSTS_FILE} (${LOCAL_IP})"
elif [ -n "$ARG1" ] && [ -n "$ARG2" ]; then
    PSSH_HOSTS_FILE="$HOSTS_DIR/pssh.hosts_${ARG1}node_${ARG2}"
    echo "使用 hosts 文件: ${PSSH_HOSTS_FILE}"
else
    PSSH_HOSTS_FILE="$HOSTS_DIR/pssh.hosts"
    echo "使用默认 hosts 文件: ${PSSH_HOSTS_FILE}"
fi

echo "停止 SGLang 服务..."
pssh -P -t 0 -h "${PSSH_HOSTS_FILE}" "
    ps aux | grep '[s]glang' | grep -v 'pssh' | awk '{print \$2}' | xargs -r kill -9 || true
" || true

echo "SGLang 停止命令已发送 (目标 hosts: ${PSSH_HOSTS_FILE})"
