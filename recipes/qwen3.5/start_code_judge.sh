#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the PRIME code judge on 127.0.0.1:$CODE_JUDGE_PORT (default 17580) and wait until it grades correctly under
# concurrency. Reuses a judge that already passes the check. CPU only.
#
#   bash recipes/qwen3.5/start_code_judge.sh          # start (or reuse)
#   bash recipes/qwen3.5/start_code_judge.sh stop     # stop the judge this script started
#
# The judge executes model-written programs with PRIME's reliability_guard only (not a sandbox): it runs with ulimits,
# bound to localhost. Run the recipes on a machine you control. Settings: CODE_JUDGE_WORKERS (uvicorn workers, 6),
# CODE_JUDGE_POOL (judging processes per worker, 16), CODE_JUDGE_TIMEOUT (seconds per judgment, 30).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

JUDGE_DIR="${OUT_ROOT}/code_judge"
PID_FILE="${JUDGE_DIR}/judge.pid"
EP="127.0.0.1:${CODE_JUDGE_PORT}"
mkdir -p "${JUDGE_DIR}"

if [ "${1:-}" = stop ]; then
    if [ -f "${PID_FILE}" ]; then
        pid=$(cat "${PID_FILE}")
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        rm -f "${PID_FILE}"
        log "code judge stopped (pid ${pid})"
    fi
    exit 0
fi

judge_ok() { "${PYTHON}" "${RECIPE_DIR}/lib/judge_check.py" --ep "${EP}" > "${JUDGE_DIR}/check.log" 2>&1; }

if curl -sf -m 5 "http://${EP}/health" >/dev/null 2>&1; then
    if judge_ok; then log "code judge at ${EP} is up and passes the check (reused)"; exit 0; fi
    die "a service on ${EP} fails the judge check (see ${JUDGE_DIR}/check.log); stop it or set CODE_JUDGE_PORT"
fi

log "starting the code judge at ${EP} (workers=${CODE_JUDGE_WORKERS:-6})"
(
    ulimit -c 0
    ulimit -u "${CODE_JUDGE_NPROC_CAP:-30000}" 2>/dev/null || true
    ulimit -v 8388608 2>/dev/null || true         # 8 GiB of address space per process
    ulimit -f 1048576 2>/dev/null || true         # 1 GiB per written file
    cd "${JUDGE_DIR}"
    exec setsid nice -n 5 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        http_proxy="" https_proxy="" no_proxy="*" PYTHONPATH="${MILES_DIR}" \
        "${PYTHON}" -m uvicorn Uni_OPD_utils.outcome_reward.PRIME_code_server.judge_app:app \
        --host 127.0.0.1 --port "${CODE_JUDGE_PORT}" --workers "${CODE_JUDGE_WORKERS:-6}" \
        --log-level info --no-access-log
) > "${JUDGE_DIR}/judge.log" 2>&1 < /dev/null &
echo $! > "${PID_FILE}"

for i in $(seq 1 24); do
    sleep 10
    if judge_ok; then log "code judge ready after $((i * 10)) s: $(grep JUDGE_CHECK "${JUDGE_DIR}/check.log" | head -1)"; exit 0; fi
done
tail -n 20 "${JUDGE_DIR}/judge.log" >&2
die "the code judge did not pass the concurrency check within 240 s (logs: ${JUDGE_DIR})"
