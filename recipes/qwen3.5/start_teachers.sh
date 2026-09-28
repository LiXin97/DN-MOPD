#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the three teacher servers of one size (SGLang, TP 1, one GPU each; GPUs and ports in configs/student_opd.yaml:
# math GPU 4 port 13141, code GPU 5 port 13142, IF GPU 6 port 13140) and wait until all answer.
#
#   bash recipes/qwen3.5/start_teachers.sh SIZE          # start (servers already serving the same model are reused)
#   bash recipes/qwen3.5/start_teachers.sh SIZE stop     # stop the servers this script started
#
# Teachers: TEACHER_MATH / TEACHER_CODE / TEACHER_IF (HF directories); default: the exports of train_teacher_grpo.sh,
# $CKPT_ROOT/qwen3.5-<SIZE>_teacher_<domain>_hf. Prints STARTED_BY_THIS_CALL=0|1 as its last line.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SIZE="${1:?usage: start_teachers.sh SIZE [stop]}"
check_size "${SIZE}"
SERVE_DIR="${OUT_ROOT}/teacher_servers/qwen3.5-${SIZE}"
PID_FILE="${SERVE_DIR}/pids"
mkdir -p "${SERVE_DIR}"

if [ "${2:-}" = stop ]; then
    if [ -f "${PID_FILE}" ]; then
        while read -r pid; do
            [ -n "${pid}" ] && { kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true; }
        done < "${PID_FILE}"
        rm -f "${PID_FILE}"
        log "teacher servers of ${SIZE} stopped"
    fi
    exit 0
fi

serving() {   # port model -> 0 when the server on port answers and serves exactly model
    "${PYTHON}" - "$1" "$2" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    info = json.loads(opener.open(f"http://127.0.0.1:{port}/get_model_info", timeout=10).read())
except Exception:
    sys.exit(3)
sys.exit(0 if info.get("model_path") == model else 4)
PY
}

sanitize_env
started=0
mapfile -t SERVERS < <(recipe teacher-servers)
[ "${#SERVERS[@]}" -eq 3 ] || die "configs/student_opd.yaml must define three teacher servers"
mapfile -t COMMON < <(recipe get teacher_servers.common_args --file student_opd \
                        | "${PYTHON}" -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')
PORTS=()
for row in "${SERVERS[@]}"; do
    read -r domain gpu port memfrac name <<< "${row}"
    model="$(teacher_export "${SIZE}" "${domain}")"
    [ -f "${model}/config.json" ] || die "no ${domain} teacher at ${model} (set $(teacher_var "${domain}") or run train_teacher_grpo.sh)"
    PORTS+=( "${port}" )
    if curl -sf -m 5 "http://127.0.0.1:${port}/health_generate" >/dev/null 2>&1; then
        serving "${port}" "${model}" || die "port ${port} serves another model than ${model}; stop it first"
        log "${name}: reusing the server on port ${port} (${model})"
        continue
    fi
    gpus_free "${gpu}" || die "GPU ${gpu} (the ${domain} teacher) is busy"
    log "${name}: starting on GPU ${gpu}, port ${port}, mem fraction ${memfrac} (${model})"
    setsid env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m sglang.launch_server \
        --model-path "${model}" --host 127.0.0.1 --port "${port}" "${COMMON[@]}" \
        --mem-fraction-static "${memfrac}" > "${SERVE_DIR}/${domain}.log" 2>&1 < /dev/null &
    echo $! >> "${PID_FILE}"
    started=1
done

for port in "${PORTS[@]}"; do
    wait_http "http://127.0.0.1:${port}/health_generate" 1800 || die "teacher on port ${port} did not come up (logs: ${SERVE_DIR})"
done
log "teacher servers of ${SIZE} ready on ports ${PORTS[*]}"
echo "STARTED_BY_THIS_CALL=${started}"
