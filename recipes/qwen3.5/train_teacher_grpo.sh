#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Train one GRPO teacher (paper App. A.1) on all 8 GPUs and export it to Hugging Face format.
#
#   bash recipes/qwen3.5/train_teacher_grpo.sh DOMAIN SIZE [UPDATES]
#     DOMAIN   math | code | ifeval        SIZE  2b | 4b | 9b
#     UPDATES  rollout steps (default: the checkpoint the paper used, configs/teacher_grpo.yaml updates_used)
#
# Data:     $DATA_ROOT/teacher/<DOMAIN>_train.jsonl (TEACHER_DATA overrides; `python data/download.py` fetches it)
# Output:   $CKPT_ROOT/qwen3.5-<SIZE>_teacher_<DOMAIN>/         FSDP checkpoints (resumable: rerun the same command)
#           $CKPT_ROOT/qwen3.5-<SIZE>_teacher_<DOMAIN>_hf/      HF export of iteration UPDATES (EXPORT_HF=0 skips it)
#           $OUT_ROOT/logs/qwen3.5-<SIZE>_teacher_<DOMAIN>/train.log
# Rewards:  math = grade_answer_verl (time-limited); code = the PRIME judge (started by start_code_judge.sh);
#           ifeval = the vendored IF constraint checkers. Dynamic sampling drops zero-variance groups (<= 8 batches).
# RUN_NAME, GPUS (default 0-7), PROFILE=smoke (a short path test) and EXTRA_TRAINER_ARGS (extra miles options) are
# optional.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DOMAIN="${1:?usage: train_teacher_grpo.sh DOMAIN SIZE [UPDATES]}"
SIZE="${2:?usage: train_teacher_grpo.sh DOMAIN SIZE [UPDATES]}"
check_domain "${DOMAIN}"
check_size "${SIZE}"
UPDATES="${3:-$(recipe teacher-updates --domain "${DOMAIN}" --size "${SIZE}")}"
PROFILE="${PROFILE:-}"
PROFILE_ARGS=(); [ -n "${PROFILE}" ] && PROFILE_ARGS=( --profile "${PROFILE}" )
RUN="${RUN_NAME:-qwen3.5-${SIZE}_teacher_${DOMAIN}}"
SAVE="${CKPT_ROOT}/${RUN}"
LOGD="${OUT_ROOT}/logs/${RUN}"
DATA="${TEACHER_DATA:-${DATA_ROOT}/$(recipe get "domains.${DOMAIN}.teacher_data" --file models)}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BASE="$(base_model "${SIZE}")" || die "no base model for ${SIZE}"
mkdir -p "${SAVE}" "${LOGD}"
[ -f "${DATA}" ] || die "missing teacher data ${DATA} (fetch it with: python data/download.py)"
log "GRPO teacher ${RUN}: domain=${DOMAIN} size=${SIZE} updates=${UPDATES} base=${BASE} data=${DATA}"

if iter_done "${SAVE}" "${UPDATES}"; then
    log "iteration ${UPDATES} already complete"
else
    if [ "${DOMAIN}" = ifeval ]; then
        "${PYTHON}" "${MILES_DIR}/Uni_OPD_utils/OPD_reward/ifeval_checkers.py" || die "IF checkers self-test failed"
    fi
    code_judge_env
    [ "${DOMAIN}" = code ] && bash "${RECIPE_DIR}/start_code_judge.sh"
    sanitize_env
    gpus_free "${GPUS}" || die "the teacher needs GPUs ${GPUS}"
    [ "$(echo "${GPUS}" | tr ',' '\n' | grep -c .)" = "$(recipe get recipe.gpus --file teacher_grpo)" ] \
        || die "GPUS must list $(recipe get recipe.gpus --file teacher_grpo) GPUs"
    export PYTHONPATH="${MILES_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    DYNAMIC_SAMPLING_MAX_GEN_BATCHES="$(recipe get recipe.dynamic_sampling_max_gen_batches --file teacher_grpo)"
    export DYNAMIC_SAMPLING_MAX_GEN_BATCHES
    prune_partial_saves "${SAVE}"
    mapfile -t ARGV < <(recipe teacher-argv --domain "${DOMAIN}" --size "${SIZE}" --base "${BASE}" --data "${DATA}" \
                          --save "${SAVE}" --updates "${UPDATES}" "${PROFILE_ARGS[@]}")
    [ "${#ARGV[@]}" -gt 10 ] || die "could not build the trainer arguments"
    trap ray_stop EXIT
    ray_start "${GPUS}" || die "ray did not start"
    # optional memory-only trainer options (e.g. --gradient-checkpointing) for GPUs smaller than the paper's
    read -r -a EXTRA <<< "${EXTRA_TRAINER_ARGS:-}"
    ARGV+=( "${EXTRA[@]}" )
    printf '%s\n' "${ARGV[@]}" > "${LOGD}/argv.txt"
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
log "TEACHER_DONE ${RUN} iteration ${UPDATES}"
