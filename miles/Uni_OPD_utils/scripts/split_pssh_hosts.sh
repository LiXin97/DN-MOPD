#!/bin/bash
# shellcheck disable=SC1091,SC2155

if [ -z "$NODE_IP_LIST" ]; then
    echo "错误: 环境变量 NODE_IP_LIST 未设置"
    exit 1
fi

# 设置 host 相关环境变量
readonly PSSH_HOSTS_DIR="${HOME}/hosts"
readonly PSSH_HOSTS_FILE="${PSSH_HOSTS_DIR}/pssh.hosts"
echo "正在生成 PSSH hosts 文件到 $PSSH_HOSTS_FILE ..."
mkdir -p "$(dirname "$PSSH_HOSTS_FILE")"
echo "$NODE_IP_LIST" | sed 's/:.[0-9]*//g' | sed 's/,/\n/g' >"$PSSH_HOSTS_FILE"

# 计算总 IP 数量
readonly TOTAL_IP_NUM="$(wc -l <"$PSSH_HOSTS_FILE")"
echo "总 IP 数量: $TOTAL_IP_NUM"

if [ "${TOTAL_IP_NUM}" -le 0 ]; then
    echo "错误：IP 列表为空，无法分组"
    exit 1
fi

# 校验函数：必须是正整数，且是 1 或 2 的倍数
validate_group_num() {
    local n="$1"

    if ! [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误：参数 [$n] 不是正整数"
        return 1
    fi

    if [ "$n" -ne 1 ] && [ "$n" -ne 2 ] && [ $((n % 2)) -ne 0 ]; then
        echo "错误：参数 [$n] 非法。每组 IP 数量必须是 1 或 2 的倍数"
        echo "有效值示例: 1, 2, 4, 6, 8, 10, 12..."
        return 1
    fi

    if [ "$n" -gt "$TOTAL_IP_NUM" ]; then
        echo "错误：参数 [$n] 必须小于或等于总 IP 数量 ($TOTAL_IP_NUM)"
        return 1
    fi

    return 0
}

# 构建待处理分组大小列表
declare -a GROUP_SIZES=()

if [ "$#" -eq 0 ]; then
    echo "未传入参数，将自动按 1 2 4 8 16 32 中 <= $TOTAL_IP_NUM 的值进行分组"
    for n in 1 2 4 8 16 32; do
        if [ "$n" -le "$TOTAL_IP_NUM" ]; then
            GROUP_SIZES+=("$n")
        fi
    done

    if [ "${#GROUP_SIZES[@]}" -eq 0 ]; then
        echo "错误：没有可用分组大小（总IP数量过小）"
        exit 1
    fi
else
    echo "检测到多参数输入: $*"
    for n in "$@"; do
        validate_group_num "$n" || exit 1
        GROUP_SIZES+=("$n")
    done
fi

# 去重（避免重复生成同一分组）
declare -A seen=()
declare -a FINAL_GROUP_SIZES=()
for n in "${GROUP_SIZES[@]}"; do
    if [ -z "${seen[$n]}" ]; then
        FINAL_GROUP_SIZES+=("$n")
        seen[$n]=1
    fi
done

echo "最终分组大小集合: ${FINAL_GROUP_SIZES[*]}"

# 准备待传输文件列表（包含原始 hosts 文件）
declare -a FILES_TO_TRANSFER=("$PSSH_HOSTS_FILE")

# 为每个分组大小生成对应文件
for IP_GROUP_NUM in "${FINAL_GROUP_SIZES[@]}"; do
    total_groups=$(((TOTAL_IP_NUM + IP_GROUP_NUM - 1) / IP_GROUP_NUM))
    echo "---"
    echo "开始处理每组 $IP_GROUP_NUM 个 IP，共 $total_groups 组"

    for i in $(seq 1 "$total_groups"); do
        start_line=$(((i - 1) * IP_GROUP_NUM + 1))

        # 结束行要么是理论上的结束，要么是总 IP 数量
        end_line=$((i * IP_GROUP_NUM > TOTAL_IP_NUM ? TOTAL_IP_NUM : i * IP_GROUP_NUM))
        # 构建分组文件名
        group_file_name="pssh.hosts_${IP_GROUP_NUM}node_${i}"
        group_file_path="$PSSH_HOSTS_DIR/$group_file_name"

        # 提取对应行的IP地址并写入分组文件
        sed -n "${start_line},${end_line}p" "$PSSH_HOSTS_FILE" >"$group_file_path"

        # 记录该分组文件以便后续传输
        FILES_TO_TRANSFER+=("$group_file_path")

        first_ip=$(head -n 1 "$group_file_path")
        actual_count=$(wc -l <"$group_file_path")

        echo "文件 $group_file_name 已创建 (行 $start_line-$end_line)"
        echo "    首个 IP 地址: $first_ip"
        echo "    包含 $actual_count 个 IP 地址"
    done
done

echo "IP地址分组完成！"

# 使用 pssh 在所有机器上执行相同的操作
echo "正在在所有机器上同步文件..."
echo "1) 创建远端目录: ${PSSH_HOSTS_DIR}"
pssh -h "${PSSH_HOSTS_FILE}" -i "mkdir -p '${PSSH_HOSTS_DIR}'"

# 传输所有分组文件 (它们位于 $HOME 目录)
echo "2) 一次性传输原始列表 + 全部分组文件"
pscp -h "${PSSH_HOSTS_FILE}" "${FILES_TO_TRANSFER[@]}" "${PSSH_HOSTS_DIR}"

echo "---"
echo "所有机器上的 IP 地址列表文件和分组文件均已更新！"
