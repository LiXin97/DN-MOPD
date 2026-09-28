#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Shared settings and helpers of the Qwen3.5 recipes. Sourced by every script in this directory; not run directly.
#
# Paths (environment variables; defaults relative to the repository):
#   DN_MOPD_ROOT  repository root                          (default: two levels above this file)
#   MODEL_ROOT    base models: $MODEL_ROOT/Qwen3.5-<S>      (default: $DN_MOPD_ROOT/models; BASE_MODEL overrides)
#   DATA_ROOT     released prompt sets                      (default: $DN_MOPD_ROOT/data)
#   CKPT_ROOT     training checkpoints and HF exports       (default: $DN_MOPD_ROOT/outputs/checkpoints)
#   OUT_ROOT      logs, run configs, SeqKD corpora, banks   (default: $DN_MOPD_ROOT/outputs)
# Interpreters:
#   PYTHON        the training environment's python (env/requirements-train.txt)          (default: python)
#   VLLM_PYTHON   a python with vLLM (env/requirements-eval.txt), for SeqKD / bank generation (default: $PYTHON)
#   RAY           the ray CLI of the training environment                                   (default: ray)
# Environment hygiene (single node):
#   DN_MOPD_SANITIZE_NCCL=1 (default) unsets NCCL_* / UCX_* inherited from a cluster profile and sets
#   NCCL_NET_PLUGIN=none (all collectives stay on the node); set it to 0 to keep your NCCL settings.
#   Every training script runs `ray stop --force` before starting its own head node: use a dedicated node.

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DN_MOPD_ROOT="${DN_MOPD_ROOT:-$(cd "${RECIPE_DIR}/../.." && pwd)}"
MILES_DIR="${MILES_DIR:-${DN_MOPD_ROOT}/miles}"
MODEL_ROOT="${MODEL_ROOT:-${DN_MOPD_ROOT}/models}"
DATA_ROOT="${DATA_ROOT:-${DN_MOPD_ROOT}/data}"
CKPT_ROOT="${CKPT_ROOT:-${DN_MOPD_ROOT}/outputs/checkpoints}"
OUT_ROOT="${OUT_ROOT:-${DN_MOPD_ROOT}/outputs}"
PYTHON="${PYTHON:-python}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON}}"
RAY="${RAY:-ray}"
CODE_JUDGE_PORT="${CODE_JUDGE_PORT:-17580}"
export DN_MOPD_ROOT MILES_DIR MODEL_ROOT DATA_ROOT CKPT_ROOT OUT_ROOT PYTHON VLLM_PYTHON RAY CODE_JUDGE_PORT

log() { echo "[dn-mopd $(date -u +%H:%M:%S)] $*"; }
die() { echo "[dn-mopd] ERROR: $*" >&2; exit 2; }

recipe() { "${PYTHON}" "${RECIPE_DIR}/lib/recipe.py" "$@"; }

check_size() { case "${1:-}" in 2b|4b|9b) ;; *) die "SIZE must be one of 2b, 4b, 9b (got '${1:-}')" ;; esac; }
check_domain() { case "${1:-}" in math|code|ifeval) ;; *) die "DOMAIN must be one of math, code, ifeval (got '${1:-}')" ;; esac; }

# base_model SIZE -> local directory of the Qwen3.5 base model (BASE_MODEL overrides). If the directory is missing and
# HF_HUB_OFFLINE is not 1, the model is downloaded into the Hugging Face cache.
base_model() {
    local size=$1 dir hf_id path
    if [ -n "${BASE_MODEL:-}" ]; then echo "${BASE_MODEL}"; return 0; fi
    dir=$(recipe get "sizes.${size}.dir" --file models) || return 2
    hf_id=$(recipe get "sizes.${size}.hf_id" --file models) || return 2
    path="${MODEL_ROOT}/${dir}"
    if [ -f "${path}/config.json" ]; then echo "${path}"; return 0; fi
    [ "${HF_HUB_OFFLINE:-0}" = 1 ] && { echo "missing ${path} (HF_HUB_OFFLINE=1): download ${hf_id} there" >&2; return 2; }
    "${PYTHON}" -c "from huggingface_hub import snapshot_download; print(snapshot_download('${hf_id}'))"
}

# teacher_export SIZE DOMAIN -> the HF export of the size's GRPO teacher (TEACHER_MATH / TEACHER_CODE / TEACHER_IF
# override; default: the export written by train_teacher_grpo.sh).
teacher_var() { case "$1" in math) echo TEACHER_MATH ;; code) echo TEACHER_CODE ;; ifeval) echo TEACHER_IF ;; esac; }
teacher_export() {
    local size=$1 domain=$2 var
    var=$(teacher_var "${domain}")
    if [ -n "${!var:-}" ]; then echo "${!var}"; else echo "${CKPT_ROOT}/qwen3.5-${size}_teacher_${domain}_hf"; fi
}

# Environment hygiene for a single-node run.
sanitize_env() {
    if [ "${DN_MOPD_SANITIZE_NCCL:-1}" = 1 ]; then
        local v
        for v in $(env | grep -oE '^(NCCL|UCX)_[A-Z_0-9]+' || true); do unset "${v}"; done
        export NCCL_NET_PLUGIN=none
    fi
    # local services (teachers, code judge, ray) must not go through a proxy
    export no_proxy="127.0.0.1,localhost,${no_proxy:-}" NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
    export http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY=""
    export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
    # PyTorch CUDA 13 wheels ship their CUDA libraries under site-packages/nvidia/*/lib, which some dynamic loaders do
    # not search (SGLang's deep_gemm then fails to load libnvrtc); put them on LD_LIBRARY_PATH.
    local sp d extra=""
    sp="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)" || sp=""
    if [ -n "${sp}" ]; then
        for d in "${sp}"/nvidia/*/lib "${sp}"/nvidia/*/lib64 "${sp}"/nvidia/*/*/lib; do
            [ -d "${d}" ] && extra="${extra}${d}:"
        done
    fi
    [ -n "${extra}" ] && export LD_LIBRARY_PATH="${extra}${LD_LIBRARY_PATH:-}"
    return 0
}

# gpus_free "0,1,2" -> fails unless every listed GPU exists and holds at most 1 GiB (no other job); never kills
# anything.
gpus_free() {
    local list=$1 table verdict
    command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found: this job needs NVIDIA GPUs" >&2; return 1; }
    table=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits) \
        || { echo "nvidia-smi failed" >&2; return 1; }
    verdict=$(echo "${table}" | awk -F", " -v L="${list}" '
        BEGIN { n = split(L, a, ","); for (i = 1; i <= n; i++) want[a[i]] = 1 }
        ($1 in want) { seen[$1] = 1; if ($2 > 1024) busy++ }
        END { for (g in want) if (!(g in seen)) missing++; print (missing + 0) " " (busy + 0) }')
    [ "${verdict}" = "0 0" ] || { echo "GPUs ${list}: missing/busy = ${verdict}; free them first" >&2; return 1; }
}

# local_ip -> an address of this host that ray can bind (LOCAL_IP overrides); IPv6 is bracketed by ray_address.
local_ip() {
    if [ -n "${LOCAL_IP:-}" ]; then echo "${LOCAL_IP}"; return; fi
    local ip
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo "${ip:-127.0.0.1}"
}
ray_address() { case "$1" in *:*:*) echo "[$1]:$2" ;; *) echo "$1:$2" ;; esac; }

# ray_start GPU_LIST -> a fresh single-node ray head on the listed GPUs (exports RAY_ADDRESS). The environment of
# `ray start` is what the rollout workers inherit, so every variable they read must be exported before this call.
# RAY_TMPDIR (default /tmp/ray_dn_mopd) must be a short path: ray's unix-socket paths are length-limited.
ray_start() {
    local gpus=$1 tmp="${RAY_TMPDIR:-/tmp/ray_dn_mopd}" n ip
    n=$(echo "${gpus}" | tr ',' '\n' | grep -c .)
    ip=$(local_ip)
    "${RAY}" stop --force >/dev/null 2>&1 || true
    mkdir -p "${tmp}"
    RAY_ADDRESS="$(ray_address "${ip}" "${RAY_PORT:-6379}")"
    export RAY_ADDRESS
    CUDA_VISIBLE_DEVICES="${gpus}" "${RAY}" start --head --node-ip-address "${ip}" --port "${RAY_PORT:-6379}" \
        --dashboard-port "${RAY_DASHBOARD_PORT:-8265}" --temp-dir "${tmp}" --num-gpus "${n}" --disable-usage-stats \
        >/dev/null || return 4
    local i
    for i in $(seq 1 20); do
        "${RAY}" status --address "${RAY_ADDRESS}" >/dev/null 2>&1 && return 0
        sleep 3
    done
    return 4
}
ray_stop() { "${RAY}" stop --force >/dev/null 2>&1 || true; }

# wait_http URL TIMEOUT_S -> 0 once URL answers 2xx.
wait_http() {
    local url=$1 t=${2:-600} i=0
    while [ "${i}" -lt "${t}" ]; do
        curl -sf -m 10 "${url}" >/dev/null 2>&1 && return 0
        sleep 5; i=$((i + 5))
    done
    return 1
}

# prune_partial_saves SAVE_DIR -> remove iter_* directories newer than the checkpoint pointer (an interrupted save);
# complete checkpoints are never touched. The trainer then resumes from the pointer.
prune_partial_saves() {
    local save=$1 latest d it
    [ -d "${save}" ] || return 0
    latest=$(cat "${save}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    for d in "${save}"/iter_*; do
        [ -d "${d}" ] || continue
        it=$((10#$(basename "${d}" | sed 's/iter_//')))
        if [ "${it}" -gt "${latest}" ]; then log "removing partial save $(basename "${d}") (pointer=${latest})"; rm -rf "${d}"; fi
    done
}

# iter_done SAVE_DIR ITER -> 0 when the checkpoint of iteration ITER is complete.
iter_done() { [ -f "$1/iter_$(printf %07d "$2")/meta.json" ]; }

# code_judge_env -> exports CODE_JUDGE_ENDPOINTS for the rule-based verifier (the judge on 127.0.0.1:CODE_JUDGE_PORT).
code_judge_env() {
    mkdir -p "${OUT_ROOT}/code_judge"
    printf '["127.0.0.1:%s"]\n' "${CODE_JUDGE_PORT}" > "${OUT_ROOT}/code_judge/endpoints.json"
    export CODE_JUDGE_ENDPOINTS="${OUT_ROOT}/code_judge/endpoints.json"
}
