#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the teacher-trajectory bank that annealed injection reads (one per size; configs/injection_bank.yaml).
#
#   bash recipes/qwen3.5/build_injection_bank.sh SIZE
#
# For every student prompt, the prompt's domain teacher samples 4 answers (vLLM, T 1.0, top-p 1.0, max 8,192 tokens,
# seed 42 + shard); sample 0 is kept when it finished, the trainer's verifier grades it correct (code answers go to the
# code judge, started here) and its prompt ids equal the rollout's; the teacher then scores the kept trajectory.
# Output: $OUT_ROOT/injection_bank/qwen3.5-<SIZE>/inject_bank.pt (+ .sha256, BANK_META.json, parts/ for resuming).
# Needs VLLM_PYTHON (vLLM) for generation/scoring and PYTHON (training env) for verification. GPUS default 0-7 (one
# shard per GPU). Teachers: TEACHER_MATH / TEACHER_CODE / TEACHER_IF. PROFILE=smoke is a short path test.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SIZE="${1:?usage: build_injection_bank.sh SIZE}"
check_size "${SIZE}"
PROFILE="${PROFILE:-}"
PROFILE_ARGS=(); [ -n "${PROFILE}" ] && PROFILE_ARGS=( --profile "${PROFILE}" )
BANK_DIR="${INJECTION_BANK_DIR:-${OUT_ROOT}/injection_bank/qwen3.5-${SIZE}}"
PARTS="${BANK_DIR}/parts"
BANK="${BANK_DIR}/inject_bank.pt"
PROMPTS="${STUDENT_DATA:-${DATA_ROOT}/$(recipe get student_data --file models)}"
TOK="$(base_model "${SIZE}")" || die "no base model for ${SIZE}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a CARDS <<< "${GPUS}"
NSHARDS="$(recipe get generation.shards --file injection_bank "${PROFILE_ARGS[@]}")"
[ "${NSHARDS}" -le "${#CARDS[@]}" ] || die "${NSHARDS} shards need ${NSHARDS} GPUs (GPUS=${GPUS})"
mkdir -p "${PARTS}"
[ -f "${PROMPTS}" ] || die "missing student data ${PROMPTS} (fetch it with: python data/download.py)"
if [ -f "${BANK}" ]; then log "bank exists: ${BANK} (delete it to rebuild)"; exit 0; fi
declare -A TEACHER
for d in math code ifeval; do
    TEACHER[$d]="$(teacher_export "${SIZE}" "${d}")"
    [ -f "${TEACHER[$d]}/config.json" ] || die "no ${d} teacher at ${TEACHER[$d]} (set $(teacher_var "${d}"))"
done
code_judge_env
bash "${RECIPE_DIR}/start_code_judge.sh"
sanitize_env
export VLLM_NO_USAGE_STATS=1
bank_py() { "$1" "${RECIPE_DIR}/lib/injection_bank.py" "${@:2}" --prompts "${PROMPTS}" --parts "${PARTS}" \
                --tokenizer "${TOK}" --nshards "${NSHARDS}" "${PROFILE_ARGS[@]}"; }

gpu_phase() {   # phase domain
    local phase=$1 dom=$2 g pids=() fail=0
    gpus_free "$(IFS=,; echo "${CARDS[*]:0:${NSHARDS}}")" || die "bank ${phase} needs GPUs ${CARDS[*]:0:${NSHARDS}}"
    for g in $(seq 0 $((NSHARDS - 1))); do
        CUDA_VISIBLE_DEVICES="${CARDS[$g]}" bank_py "${VLLM_PYTHON}" "${phase}" --domain "${dom}" --shard "${g}" \
            --teacher "${TEACHER[$dom]}" > "${PARTS}/${phase}_${dom}_s${g}.log" 2>&1 &
        pids+=( $! )
    done
    for g in "${pids[@]}"; do wait "${g}" || fail=1; done
    [ "${fail}" -eq 0 ] || die "phase ${phase} ${dom} failed (logs: ${PARTS}/${phase}_${dom}_s*.log; rerun to resume)"
}

for dom in math code ifeval; do
    log "${dom}: G (generation, ${NSHARDS} shards)"
    gpu_phase G "${dom}"
    log "${dom}: A (verification)"
    PYTHONPATH="${MILES_DIR}${PYTHONPATH:+:${PYTHONPATH}}" bank_py "${PYTHON}" A --domain "${dom}" \
        > "${PARTS}/A_${dom}.log" 2>&1 || die "phase A ${dom} failed (log: ${PARTS}/A_${dom}.log)"
    grep -h A_DONE "${PARTS}/A_${dom}.log" || true
    log "${dom}: B (teacher scoring)"
    gpu_phase B "${dom}"
done
teacher_args=(); for d in math code ifeval; do teacher_args+=( --teacher "${d}=${TEACHER[$d]}" ); done
bank_py "${PYTHON}" C --out "${BANK}" "${teacher_args[@]}"
log "BANK_DONE ${BANK}"
