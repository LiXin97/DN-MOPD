#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SeqKD corpus of one size (paper App. A.2): each domain's teacher answers the 900 student prompts of its domain once
# (vLLM, temperature 1.0, top-p 1.0, seed 42, max_tokens 16,384, no correctness filter). One GPU per domain.
#
#   bash recipes/qwen3.5/gen_seqkd.sh SIZE
#
# Needs VLLM_PYTHON (a python with vLLM, env/requirements-eval.txt). Teachers: TEACHER_MATH / TEACHER_CODE / TEACHER_IF
# (default: the exports of train_teacher_grpo.sh). Prompts: $DATA_ROOT/student/train_3domain.jsonl (STUDENT_DATA).
# Output: $OUT_ROOT/seqkd/qwen3.5-<SIZE>/targets.jsonl and targets.stats.json. Resumable per domain (parts/*.done).
# GPUS (default 0,1,2: math, code, ifeval) and PROFILE=smoke (short answers, a path test) are optional.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SIZE="${1:?usage: gen_seqkd.sh SIZE}"
check_size "${SIZE}"
PROFILE="${PROFILE:-}"
PROFILE_ARGS=(); [ -n "${PROFILE}" ] && PROFILE_ARGS=( --profile "${PROFILE}" )
g() { recipe get "generation.$1" --file seqkd "${PROFILE_ARGS[@]}"; }
OUTD="${SEQKD_DIR:-${OUT_ROOT}/seqkd/qwen3.5-${SIZE}}"
PARTS="${OUTD}/parts"
FINAL="${OUTD}/targets.jsonl"
STATS="${OUTD}/targets.stats.json"
PROMPTS="${STUDENT_DATA:-${DATA_ROOT}/$(recipe get student_data --file models)}"
STUDENT_TOK="$(base_model "${SIZE}")" || die "no base model for ${SIZE}"
GPUS="${GPUS:-$(g gpus | "${PYTHON}" -c 'import json,sys; print(",".join(map(str, json.load(sys.stdin))))')}"
IFS=',' read -r -a CARDS <<< "${GPUS}"
DOMS=(math code ifeval)
mkdir -p "${PARTS}"
[ -f "${PROMPTS}" ] || die "missing student data ${PROMPTS} (fetch it with: python data/download.py)"
N=$(grep -c . "${PROMPTS}")

if [ -s "${FINAL}" ] && [ -s "${STATS}" ] && [ "$(grep -c . "${FINAL}")" -eq "${N}" ]; then
    log "SeqKD corpus already complete: ${FINAL}"; exit 0
fi
sanitize_env
export VLLM_NO_USAGE_STATS=1

run_domain() {   # domain gpu
    local dom=$1 gpu=$2 part="${PARTS}/part_$1.jsonl" model
    if [ -f "${part}.done" ] && [ -s "${part}" ]; then log "${dom}: done"; return 0; fi
    model="$(teacher_export "${SIZE}" "${dom}")"
    [ -f "${model}/config.json" ] || { echo "no ${dom} teacher at ${model}" >&2; return 1; }
    log "${dom}: generating on GPU ${gpu} with ${model}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${VLLM_PYTHON}" "${RECIPE_DIR}/lib/seqkd_generate.py" worker \
        --prompts "${PROMPTS}" --domain "${dom}" --model "${model}" --template-tokenizer "${STUDENT_TOK}" \
        --out "${part}" --max-tokens "$(g max_tokens)" --max-model-len "$(g max_model_len)" \
        --max-num-seqs "$(g max_num_seqs)" --temperature "$(g temperature)" --top-p "$(g top_p)" --seed "$(g seed)" \
        --gpu-memory-utilization "$(g gpu_memory_utilization)" > "${PARTS}/gen_${dom}.log" 2>&1 \
        || { tail -n 20 "${PARTS}/gen_${dom}.log" >&2; return 1; }
    touch "${part}.done"
    grep -h SEQKD_WORKER_DONE "${PARTS}/gen_${dom}.log" || true
}

# one queue per GPU: domains assigned round-robin, a GPU runs its domains one after another
declare -a QUEUE
for i in "${!DOMS[@]}"; do c=$(( i % ${#CARDS[@]} )); QUEUE[c]="${QUEUE[$c]:-} ${DOMS[$i]}"; done
gpus_free "${GPUS}" || die "SeqKD generation needs GPUs ${GPUS}"
PIDS=()
for c in "${!QUEUE[@]}"; do
    ( for dom in ${QUEUE[$c]}; do run_domain "${dom}" "${CARDS[$c]}" || exit 1; done ) &
    PIDS+=( $! )
done
fail=0
for p in "${PIDS[@]}"; do wait "${p}" || fail=1; done
[ "${fail}" -eq 0 ] || die "generation incomplete (rerun the same command; finished domains are kept)"
"${PYTHON}" "${RECIPE_DIR}/lib/seqkd_generate.py" merge --out "${FINAL}" --stats "${STATS}" --n "${N}" \
    "${PARTS}/part_math.jsonl" "${PARTS}/part_code.jsonl" "${PARTS}/part_ifeval.jsonl"
log "SEQKD_DONE ${FINAL}"
