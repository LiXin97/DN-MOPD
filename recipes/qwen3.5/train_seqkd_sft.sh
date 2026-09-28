#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SeqKD-SFT row (paper App. A.2): supervised fine-tuning of the size's base model on the SeqKD corpus (gen_seqkd.sh),
# 4 epochs = 84 updates of 128 answers, lr 1e-5 cosine to 1e-6, response-only per-token loss, on all 8 GPUs.
#
#   bash recipes/qwen3.5/train_seqkd_sft.sh SIZE
#
# Input:  $OUT_ROOT/seqkd/qwen3.5-<SIZE>/targets.jsonl (SEQKD_CORPUS overrides)
# Output: $CKPT_ROOT/qwen3.5-<SIZE>_seqkd_sft/ (resumable) and $CKPT_ROOT/qwen3.5-<SIZE>_seqkd_sft_hf/ (EXPORT_HF=0 skips)
# RUN_NAME, GPUS (default 0-7), PROFILE=smoke and EXTRA_TRAINER_ARGS (extra miles options) are optional.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SIZE="${1:?usage: train_seqkd_sft.sh SIZE}"
check_size "${SIZE}"
PROFILE="${PROFILE:-}"
PROFILE_ARGS=(); [ -n "${PROFILE}" ] && PROFILE_ARGS=( --profile "${PROFILE}" )
RUN="${RUN_NAME:-qwen3.5-${SIZE}_seqkd_sft}"
SAVE="${CKPT_ROOT}/${RUN}"
LOGD="${OUT_ROOT}/logs/${RUN}"
CORPUS="${SEQKD_CORPUS:-${OUT_ROOT}/seqkd/qwen3.5-${SIZE}/targets.jsonl}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STUDENT="$(base_model "${SIZE}")" || die "no base model for ${SIZE}"
[ -s "${CORPUS}" ] || die "missing SeqKD corpus ${CORPUS} (bash recipes/qwen3.5/gen_seqkd.sh ${SIZE})"
N_ROWS=$(grep -c . "${CORPUS}")
mkdir -p "${SAVE}" "${LOGD}"
mapfile -t ARGV < <(recipe sft-argv --size "${SIZE}" --student "${STUDENT}" --data "${CORPUS}" --save "${SAVE}" \
                      --n-rows "${N_ROWS}" "${PROFILE_ARGS[@]}")
[ "${#ARGV[@]}" -gt 10 ] || die "could not build the trainer arguments"
UPDATES=""
for i in "${!ARGV[@]}"; do [ "${ARGV[$i]}" = --num-rollout ] && UPDATES="${ARGV[$((i + 1))]}"; done
n_gpu=$(echo "${GPUS}" | tr ',' '\n' | grep -c .)
for i in "${!ARGV[@]}"; do
    case "${ARGV[$i]}" in --actor-num-gpus-per-node|--num-gpus-per-node) ARGV[i + 1]="${n_gpu}" ;; esac
done
log "SeqKD-SFT ${RUN}: size=${SIZE} rows=${N_ROWS} updates=${UPDATES} corpus=${CORPUS}"

if iter_done "${SAVE}" "${UPDATES}"; then
    log "iteration ${UPDATES} already complete"
else
    sanitize_env
    gpus_free "${GPUS}" || die "SFT needs GPUs ${GPUS}"
    export PYTHONPATH="${MILES_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    prune_partial_saves "${SAVE}"
    # optional memory-only trainer options (e.g. --gradient-checkpointing) for GPUs smaller than the paper's
    read -r -a EXTRA <<< "${EXTRA_TRAINER_ARGS:-}"
    ARGV+=( "${EXTRA[@]}" )
    printf '%s\n' "${ARGV[@]}" > "${LOGD}/argv.txt"
    trap ray_stop EXIT
    ray_start "${GPUS}" || die "ray did not start"
    log "training (log: ${LOGD}/train.log)"
    set +e
    ( cd "${MILES_DIR}" && "${PYTHON}" Uni_OPD_utils/ray_launcher.py train.py "${ARGV[@]}" ) 2>&1 | tee -a "${LOGD}/train.log"
    rc=${PIPESTATUS[0]}
    set -e
    ray_stop
    [ "${rc}" -eq 0 ] || die "training exited with ${rc} (rerun the same command to resume)"
    iter_done "${SAVE}" "${UPDATES}" || die "training finished without a complete iteration ${UPDATES}"
fi

if [ "${EXPORT_HF:-1}" = 1 ]; then
    bash "${RECIPE_DIR}/export_hf.sh" "${SAVE}" "${UPDATES}" "${SIZE}" "${HF_OUT:-${SAVE}_hf}"
fi
log "SFT_DONE ${RUN} iteration ${UPDATES}"
