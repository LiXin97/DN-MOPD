#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Train one on-policy-distillation student (paper App. A.2) on one 8-GPU node: FSDP actor + colocated SGLang rollout on
# GPUs 0-3, the size's three teacher servers on GPUs 4, 5, 6 (started here unless already running), the code judge on
# the CPU (correctness labels; read by annealed injection, logged for every method).
#
#   bash recipes/qwen3.5/train_student.sh METHOD SIZE [SEED] [UPDATES]
#     METHOD   label | label_hook | dn_mopd | single_math | single_code | single_if | uniform_pool | dynamic_router |
#              annealed_injection | fixed_w | fixed_math | fixed_if | frozen_update0 | observe   (configs/methods.yaml)
#     SIZE     2b | 4b | 9b          SEED  default 42 (sets --rollout-seed)          UPDATES  default 80
#
# UPDATES=160 continues the 80-update run of the same METHOD/SIZE/SEED in place (the paper's 160-update rows).
# Inputs:   the size's base model (MODEL_ROOT or BASE_MODEL), the three teacher exports (TEACHER_MATH / TEACHER_CODE /
#           TEACHER_IF, default $CKPT_ROOT/qwen3.5-<SIZE>_teacher_<domain>_hf), $DATA_ROOT/student/train_3domain.jsonl
#           (STUDENT_DATA overrides); annealed_injection also needs the size's bank (build_injection_bank.sh;
#           INJECTION_BANK overrides).
# Output:   $CKPT_ROOT/qwen3.5-<SIZE>_<METHOD>_s<SEED>/                  FSDP checkpoints (rerun the command to resume)
#           $CKPT_ROOT/qwen3.5-<SIZE>_<METHOD>_s<SEED>_hf_iter<UPDATES>/   HF export (EXPORT_HF=0 skips it)
#           $OUT_ROOT/runs/<run>/  train.log, argv.txt, teacher server list/map, hook_config.json, audit/ (per-batch
#           hook records: routing, DN multipliers, injections)
# Optional: RUN_NAME, STUDENT_GPUS (default 0,1,2,3), PROFILE=smoke (short path test), KEEP_TEACHERS=1 (leave teacher
#           servers this script started running), EXTRA_TRAINER_ARGS (memory-only miles options such as
#           "--gradient-checkpointing" for smaller GPUs; the paper's student runs used none).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHOD="${1:?usage: train_student.sh METHOD SIZE [SEED] [UPDATES]}"
SIZE="${2:?usage: train_student.sh METHOD SIZE [SEED] [UPDATES]}"
SEED="${3:-42}"
check_size "${SIZE}"
recipe methods | grep -qx "${METHOD}" || die "unknown METHOD '${METHOD}'; one of: $(recipe methods | tr '\n' ' ')"
[[ "${SEED}" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer"
PROFILE="${PROFILE:-}"
PROFILE_ARGS=(); [ -n "${PROFILE}" ] && PROFILE_ARGS=( --profile "${PROFILE}" )
UPDATES="${4:-$(recipe get recipe.updates --file student_opd "${PROFILE_ARGS[@]}")}"
[[ "${UPDATES}" =~ ^[1-9][0-9]*$ ]] || die "UPDATES must be a positive integer"
read -r ROUTE _ _ <<< "$(recipe student-route --method "${METHOD}")"   # legacy | hook

RUN="${RUN_NAME:-qwen3.5-${SIZE}_${METHOD}_s${SEED}}"
SAVE="${CKPT_ROOT}/${RUN}"
RUND="${OUT_ROOT}/runs/${RUN}"
DATA="${STUDENT_DATA:-${DATA_ROOT}/$(recipe get student_data --file models)}"
STUDENT_GPUS="${STUDENT_GPUS:-$(recipe get recipe.student_gpus --file student_opd | "${PYTHON}" -c 'import json,sys; print(",".join(map(str, json.load(sys.stdin))))')}"
STUDENT="$(base_model "${SIZE}")" || die "no base model for ${SIZE}"
mkdir -p "${SAVE}" "${RUND}"
[ -f "${DATA}" ] || die "missing student data ${DATA} (fetch it with: python data/download.py)"
declare -A TEACHER
for d in math code ifeval; do
    TEACHER[$d]="$(teacher_export "${SIZE}" "${d}")"
    [ -f "${TEACHER[$d]}/config.json" ] || die "no ${d} teacher at ${TEACHER[$d]} (set $(teacher_var "${d}"))"
done
log "student ${RUN}: method=${METHOD} route=${ROUTE} size=${SIZE} seed=${SEED} updates=${UPDATES} student=${STUDENT}"

if iter_done "${SAVE}" "${UPDATES}"; then
    log "iteration ${UPDATES} already complete"
else
    # ---- teacher routing files (legacy route: the map is the router) and, for hook methods, the hook config
    mapfile -t SERVER_FILES < <(recipe server-files --method "${METHOD}" --math "${TEACHER[math]}" \
                                  --code "${TEACHER[code]}" --ifeval "${TEACHER[ifeval]}" --out-dir "${RUND}")
    [ "${#SERVER_FILES[@]}" -eq 2 ] || die "could not write the teacher server files"
    export OPD_TEACHER_SERVER_LIST="${SERVER_FILES[0]}" OPD_TEACHER_SERVER_MAP="${SERVER_FILES[1]}"
    if [ "${ROUTE}" = hook ]; then
        BANK_ARGS=()
        if [ "${METHOD}" = annealed_injection ]; then
            BANK="${INJECTION_BANK:-${OUT_ROOT}/injection_bank/qwen3.5-${SIZE}/inject_bank.pt}"
            [ -f "${BANK}" ] || die "annealed_injection needs the size's bank at ${BANK} (bash recipes/qwen3.5/build_injection_bank.sh ${SIZE})"
            BANK_ARGS=( --bank "${BANK}" )
        fi
        export MOPD_HOOK_CONFIG="${RUND}/hook_config.json"
        recipe hook-config --method "${METHOD}" --size "${SIZE}" --seed "${SEED}" --updates "${UPDATES}" \
            --audit-dir "${RUND}/audit" --out "${MOPD_HOOK_CONFIG}" "${BANK_ARGS[@]}" "${PROFILE_ARGS[@]}" >/dev/null \
            || die "could not write the hook config"
    else
        unset MOPD_HOOK_CONFIG
    fi

    # ---- services: code judge (correctness labels) and the three teacher servers
    code_judge_env
    bash "${RECIPE_DIR}/start_code_judge.sh"
    TEACHER_OUT="$(bash "${RECIPE_DIR}/start_teachers.sh" "${SIZE}")" || die "teacher servers failed"
    echo "${TEACHER_OUT}" | grep -v '^STARTED_BY_THIS_CALL=' || true
    STARTED_TEACHERS="$(echo "${TEACHER_OUT}" | sed -n 's/^STARTED_BY_THIS_CALL=//p' | tail -1)"
    cleanup() {
        ray_stop
        if [ "${STARTED_TEACHERS:-0}" = 1 ] && [ "${KEEP_TEACHERS:-0}" != 1 ]; then
            bash "${RECIPE_DIR}/start_teachers.sh" "${SIZE}" stop || true
        fi
    }
    trap cleanup EXIT

    sanitize_env
    gpus_free "${STUDENT_GPUS}" || die "the student needs GPUs ${STUDENT_GPUS}"
    export PYTHONPATH="${MILES_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    OPD_RM_MAX_INFLIGHT="$(recipe get recipe.opd_rm_max_inflight --file student_opd)"
    export OPD_RM_MAX_INFLIGHT
    prune_partial_saves "${SAVE}"
    mapfile -t ARGV < <(recipe student-argv --method "${METHOD}" --size "${SIZE}" --seed "${SEED}" \
                          --updates "${UPDATES}" --student "${STUDENT}" --data "${DATA}" --save "${SAVE}" "${PROFILE_ARGS[@]}")
    [ "${#ARGV[@]}" -gt 10 ] || die "could not build the trainer arguments"
    n_student=$(echo "${STUDENT_GPUS}" | tr ',' '\n' | grep -c .)
    for i in "${!ARGV[@]}"; do    # the GPU count follows STUDENT_GPUS
        case "${ARGV[$i]}" in --actor-num-gpus-per-node|--num-gpus-per-node|--rollout-num-gpus) ARGV[i + 1]="${n_student}" ;; esac
    done
    # optional memory-only trainer options (e.g. --gradient-checkpointing) for GPUs smaller than the paper's
    read -r -a EXTRA <<< "${EXTRA_TRAINER_ARGS:-}"
    ARGV+=( "${EXTRA[@]}" )
    printf '%s\n' "${ARGV[@]}" > "${RUND}/argv.txt"
    # the rollout workers inherit these from `ray start`
    ray_start "${STUDENT_GPUS}" || die "ray did not start"
    log "training (log: ${RUND}/train.log)"
    set +e
    ( cd "${MILES_DIR}" && "${PYTHON}" Uni_OPD_utils/ray_launcher.py train.py "${ARGV[@]}" ) 2>&1 | tee -a "${RUND}/train.log"
    rc=${PIPESTATUS[0]}
    set -e
    ray_stop
    [ "${rc}" -eq 0 ] || die "training exited with ${rc} (rerun the same command to resume)"
    iter_done "${SAVE}" "${UPDATES}" || die "training finished without a complete iteration ${UPDATES}"
fi

if [ "${EXPORT_HF:-1}" = 1 ]; then
    bash "${RECIPE_DIR}/export_hf.sh" "${SAVE}" "${UPDATES}" "${SIZE}"
fi
log "STUDENT_DONE ${RUN} iteration ${UPDATES}"
