#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# End-to-end GPU smoke test of the trainer on ONE 8-GPU node (Qwen3.5-2B, short settings; a path test, not a paper
# setting). Self-checking: prints a PASS/FAIL line per step and a summary, and exits non-zero when any step fails.
#
#   MODEL_ROOT=/path/to/models bash recipes/qwen3.5/smoke_test.sh
#
# Prerequisites: the training environment (env/install.md) active or PYTHON / RAY pointing at it; MODEL_ROOT holding
# Qwen3.5-2B (or BASE_MODEL=<dir>); the released prompt sets under DATA_ROOT (data/student/train_3domain.jsonl and
# data/teacher/math_train.jsonl; `python data/download.py` fetches them); eight free GPUs. Optional:
# SMOKE_TEACHER_MATH / SMOKE_TEACHER_CODE / SMOKE_TEACHER_IF (three 2B teacher exports; default: the base model serves
# as all three teachers).
# Steps:
#   1. start the code judge and check it under concurrency
#   2. 2 GRPO updates of a math teacher on a 128-prompt slice (8 GPUs, dynamic sampling on)
#   3. start the three teacher servers (GPUs 4, 5, 6)
#   4. 2 updates each of a Label student and a DN-MOPD student on a 48-prompt slice (GPUs 0-3); the DN-MOPD run must
#      log its per-domain multipliers (MOPD_HOOK_PANEL "dnorm") and the trainer must apply them (MOPD_DN_APPLIED)
#   5. HF export of the DN-MOPD student and a load check of the export
# Everything is written under $OUT_ROOT/smoke (checkpoints under $OUT_ROOT/smoke/checkpoints).
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SMOKE="${OUT_ROOT}/smoke"
export OUT_ROOT="${SMOKE}" CKPT_ROOT="${SMOKE}/checkpoints"
export PROFILE=smoke EXPORT_HF=0
SIZE=2b
mkdir -p "${SMOKE}/data" "${CKPT_ROOT}"
RESULTS=()
FAILED=0
record() {  # name status detail
    RESULTS+=( "$(printf '%-34s %s  %s' "$1" "$2" "$3")" )
    [ "$2" = PASS ] || FAILED=1
    log "SMOKE $2 $1 $3"
}
finish() {
    bash "${RECIPE_DIR}/start_teachers.sh" "${SIZE}" stop >/dev/null 2>&1 || true
    bash "${RECIPE_DIR}/start_code_judge.sh" stop >/dev/null 2>&1 || true
    ray_stop
    echo
    echo "================ DN-MOPD trainer smoke test ================"
    printf '%s\n' "${RESULTS[@]}"
    if [ "${FAILED}" -eq 0 ] && [ "${#RESULTS[@]}" -ge 7 ]; then echo "SMOKE_TEST: PASS"; else echo "SMOKE_TEST: FAIL"; fi
    echo "outputs: ${SMOKE}"
}
trap finish EXIT
stop_if_failed() { [ "${FAILED}" -eq 0 ] || { log "stopping after a failed step"; exit 1; }; }

# ---- 0. inputs
n_gpu=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
if [ "${n_gpu:-0}" -lt 8 ]; then record "0 inputs" FAIL "needs 8 GPUs, nvidia-smi lists ${n_gpu:-0}"; exit 1; fi
BASE="$(base_model "${SIZE}")" || { record "0 inputs" FAIL "no Qwen3.5-2B under MODEL_ROOT=${MODEL_ROOT}"; exit 1; }
export BASE_MODEL="${BASE}"
STUDENT_SRC="${DATA_ROOT}/$(recipe get student_data --file models)"
MATH_SRC="${DATA_ROOT}/$(recipe get domains.math.teacher_data --file models)"
if [ -f "${STUDENT_SRC}" ] && [ -f "${MATH_SRC}" ]; then
    "${PYTHON}" - "${STUDENT_SRC}" "${SMOKE}/data/student_slice.jsonl" 16 <<'PY'
import json, sys
src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen, out = {}, []
for line in open(src):
    r = json.loads(line)
    d = (r.get("metadata") or {}).get("domain")
    if seen.get(d, 0) < k:
        seen[d] = seen.get(d, 0) + 1
        out.append(line)
assert sorted(seen) == ["code", "ifeval", "math"], seen
open(dst, "w").writelines(out)
print(f"student slice: {len(out)} prompts {seen}")
PY
    head -n 128 "${MATH_SRC}" > "${SMOKE}/data/math_slice.jsonl"
    record "0 inputs" PASS "base=${BASE}"
else
    record "0 inputs" FAIL "missing ${STUDENT_SRC} or ${MATH_SRC} (fetch them with: python data/download.py)"
fi
stop_if_failed
export TEACHER_MATH="${SMOKE_TEACHER_MATH:-${BASE}}" TEACHER_CODE="${SMOKE_TEACHER_CODE:-${BASE}}" \
       TEACHER_IF="${SMOKE_TEACHER_IF:-${BASE}}"
export STUDENT_DATA="${SMOKE}/data/student_slice.jsonl"

# ---- 1. code judge
if bash "${RECIPE_DIR}/start_code_judge.sh" > "${SMOKE}/1_code_judge.log" 2>&1; then
    record "1 code judge" PASS "$(grep -h 'JUDGE_CHECK ep' "${SMOKE}/code_judge/check.log" | head -1)"
else
    record "1 code judge" FAIL "see ${SMOKE}/1_code_judge.log"
fi
stop_if_failed

# ---- 2. GRPO teacher: 2 updates
T_RUN=smoke_teacher_math
if RUN_NAME="${T_RUN}" TEACHER_DATA="${SMOKE}/data/math_slice.jsonl" \
        bash "${RECIPE_DIR}/train_teacher_grpo.sh" math "${SIZE}" 2 > "${SMOKE}/2_teacher.log" 2>&1 \
        && iter_done "${CKPT_ROOT}/${T_RUN}" 2; then
    record "2 GRPO teacher (2 updates)" PASS "${CKPT_ROOT}/${T_RUN}/iter_0000002"
else
    record "2 GRPO teacher (2 updates)" FAIL "see ${SMOKE}/2_teacher.log and ${SMOKE}/logs/${T_RUN}/train.log"
fi
stop_if_failed

# ---- 3. teacher servers
if bash "${RECIPE_DIR}/start_teachers.sh" "${SIZE}" > "${SMOKE}/3_teachers.log" 2>&1; then
    record "3 teacher servers" PASS "ports $(recipe teacher-servers | awk '{print $3}' | tr '\n' ' ')"
else
    record "3 teacher servers" FAIL "see ${SMOKE}/3_teachers.log and ${SMOKE}/teacher_servers/"
fi
stop_if_failed

# ---- 4. students: Label (legacy route) and DN-MOPD (hook), 2 updates each
for METHOD in label dn_mopd; do
    RUN="smoke_${METHOD}"
    LOG="${SMOKE}/runs/${RUN}/train.log"
    if RUN_NAME="${RUN}" bash "${RECIPE_DIR}/train_student.sh" "${METHOD}" "${SIZE}" 42 2 \
            > "${SMOKE}/4_${METHOD}.log" 2>&1 && iter_done "${CKPT_ROOT}/${RUN}" 2; then
        record "4 student ${METHOD} (2 updates)" PASS "${CKPT_ROOT}/${RUN}/iter_0000002"
    else
        record "4 student ${METHOD} (2 updates)" FAIL "see ${SMOKE}/4_${METHOD}.log and ${LOG}"
        continue
    fi
    if [ "${METHOD}" = dn_mopd ]; then
        n_panel=$(grep -c 'MOPD_HOOK_PANEL .*"dnorm"' "${LOG}" || true)
        n_applied=$(grep -c 'MOPD_DN_APPLIED' "${LOG}" || true)
        audit_ok=$("${PYTHON}" - "${SMOKE}/runs/${RUN}/audit" <<'PY'
import glob, json, sys
rows = [r for f in glob.glob(sys.argv[1] + "/samples_*.json") for r in json.load(open(f))["rows"]]
ok = rows and all(0.25 <= r["dnorm_factor"] <= 4.0 and r["dnorm_factor"] == r["dnorm_measured_factor"] for r in rows)
doms = sorted({r["domain"] for r in rows})
# a 2-update smoke draws 16 of the 48 prompts: at least two domains are needed for a multiplier other than 1
print(f"{'yes' if ok and len(set(doms) & {'code', 'ifeval', 'math'}) >= 2 else 'no'} rows={len(rows)} domains={doms}")
PY
)
        if [ "${n_panel}" -ge 2 ] && [ "${n_applied}" -ge 2 ] && [[ "${audit_ok}" == yes* ]]; then
            record "4 DN multipliers logged+applied" PASS "panels=${n_panel} applied_lines=${n_applied} audit ${audit_ok#yes }"
        else
            record "4 DN multipliers logged+applied" FAIL "panels=${n_panel} applied_lines=${n_applied} audit=${audit_ok}"
        fi
    else
        n_applied=$(grep -c 'MOPD_DN_APPLIED' "${LOG}" || true)
        if [ "${n_applied}" -eq 0 ]; then record "4 Label applies no multiplier" PASS "MOPD_DN_APPLIED lines=0"
        else record "4 Label applies no multiplier" FAIL "MOPD_DN_APPLIED lines=${n_applied}"; fi
    fi
done
stop_if_failed

# ---- 5. HF export of the DN-MOPD student
EXPORT="${CKPT_ROOT}/smoke_dn_mopd_hf_iter2"
if bash "${RECIPE_DIR}/export_hf.sh" "${CKPT_ROOT}/smoke_dn_mopd" 2 "${SIZE}" "${EXPORT}" > "${SMOKE}/5_export.log" 2>&1 \
        && "${PYTHON}" - "${EXPORT}" "${BASE}" >> "${SMOKE}/5_export.log" 2>&1 <<'PY'
import json, os, sys
from safetensors import safe_open
from transformers import AutoConfig
out, base = sys.argv[1], sys.argv[2]
AutoConfig.from_pretrained(out)
wm = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
bm = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"] \
    if os.path.exists(os.path.join(base, "model.safetensors.index.json")) else None
missing = sorted(set(bm) - set(wm)) if bm else []
assert all(k.startswith("mtp.") for k in missing), missing[:5]
with safe_open(os.path.join(out, sorted(set(wm.values()))[0]), "pt") as h:
    k = next(iter(h.keys()))
    t = h.get_tensor(k)
print(f"EXPORT_CHECK tensors={len(wm)} missing_vs_base={len(missing)} (mtp only) first={k} dtype={t.dtype}")
PY
then
    record "5 HF export" PASS "$(grep -h EXPORT_CHECK "${SMOKE}/5_export.log" | tail -1)"
else
    record "5 HF export" FAIL "see ${SMOKE}/5_export.log"
fi
[ "${FAILED}" -eq 0 ]
