#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GPU smoke test of the evaluation pipeline: fetch the pinned graders, prepare the catalogs, generate a few questions of
# every suite with a small model on one GPU, grade them, aggregate and compare. Prints PASS/FAIL per step and exits
# non-zero on any failure.
#
#   CUDA_VISIBLE_DEVICES=0 bash eval/smoke_test.sh
#
# Environment (all optional):
#   PYTHON          interpreter with vLLM 0.18 and the grading requirements (default: python)
#   SMOKE_MODEL     HF directory or hub id (default: Qwen/Qwen3.5-2B, the paper's 2B initial student)
#   SMOKE_LIMIT     questions per suite (default: 2)
#   SMOKE_CAP       generation cap, 8192 or 16384 (default: 8192)
#   SMOKE_SUITES    comma-separated suites (default: all seven)
#   SMOKE_PYTEST    1 = also run the CPU test suite tests/eval first (default: 1)
#   SMOKE_REQUIRE_NONZERO  1 = fail if MATH-500 or IFEval scores exactly zero (default: 1; set 0 for other models)
#   SMOKE_PREPARE   1 = run eval.data.prepare (default: 1). Offline nodes: set 0 and point DN_MOPD_EVAL_DATA at a
#                   directory prepared elsewhere (generation still checks every catalog's sha256); the fetch step
#                   downloads nothing when DN_MOPD_THIRD_PARTY already holds the verified files.
#   DN_MOPD_ROOT, OUT_ROOT, DN_MOPD_THIRD_PARTY, DN_MOPD_EVAL_DATA  as in eval/README.md; a node without internet can
#                   point the last two at directories fetched and prepared elsewhere.
set -uo pipefail
ROOT="${DN_MOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PYTHON:-python}"
MODEL="${SMOKE_MODEL:-Qwen/Qwen3.5-2B}"
LIMIT="${SMOKE_LIMIT:-2}"
CAP="${SMOKE_CAP:-8192}"
SUITES="${SMOKE_SUITES:-aime25,aime26,lcb_v5,lcb_v6,ifeval,ifbench,math500}"
RUNS="${OUT_ROOT:-$ROOT/outputs}/eval_smoke"
NAME=smoke_model
cd "$ROOT" || exit 2
export TOKENIZERS_PARALLELISM=false
FAILED=0
step() {  # name command...
    local name="$1"; shift
    echo "== $name"
    if "$@"; then echo "PASS $name"; else echo "FAIL $name"; FAILED=$((FAILED + 1)); fi
}

step "python environment" "$PY" - <<'EOF'
import importlib.metadata as m
for p in ('vllm', 'torch', 'transformers', 'math-verify', 'nltk', 'langdetect', 'numpy'):
    print(f'  {p} {m.version(p)}')
if m.version('vllm') != '0.18.0':
    print('  WARNING: the paper used vLLM 0.18.0')
EOF
step "fetch pinned graders" "$PY" -m eval.external.fetch
if [ "${SMOKE_PYTEST:-1}" = "1" ]; then
    step "CPU tests (tests/eval)" "$PY" -m pytest -q -p no:cacheprovider tests/eval
fi
if [ "${SMOKE_PREPARE:-1}" = "1" ]; then
    step "prepare catalogs" "$PY" -m eval.data.prepare --suites "$SUITES"
fi

rm -rf "$RUNS/$NAME"
IFS=',' read -r -a SUITE_LIST <<< "$SUITES"
for s in "${SUITE_LIST[@]}"; do
    out="$RUNS/$NAME/$s/cap$CAP"
    step "generate $s" "$PY" -m eval.generate --model "$MODEL" --name "$NAME" --suite "$s" --max-tokens "$CAP" \
        --out "$out" --limit "$LIMIT"
    step "grade $s" "$PY" -m eval.grade --gen "$out"
done

step "check outputs" "$PY" - "$RUNS/$NAME" "$CAP" "$LIMIT" "$SUITES" "${SMOKE_REQUIRE_NONZERO:-1}" <<'EOF'
import json, sys
from pathlib import Path
from eval.common import records, released_dir
from eval.protocol import SUITES
root, cap, limit, suites, nonzero = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(','), sys.argv[5] == '1'
ok = True
for s in suites:
    d = root / s / f'cap{cap}'
    m, summ = json.loads((d / 'MANIFEST.json').read_text()), json.loads((d / 'summary.json').read_text())
    rows = records(d / 'graded.jsonl')
    good = (len(rows) == limit == summ['problems'] and all(len(r['correct']) == SUITES[s].n for r in rows)
            and all(max(r['token_counts']) <= cap for r in rows) and m['sampling']['max_tokens'] == cap)
    if s == 'math500':
        good = good and all(x == x.strip() for r in rows for x in r['responses'])
    ref = released_dir() / ('math500' if s == 'math500' else f'scores/{s}') / f'cap{cap}' / 'q35_2b_base.jsonl'
    paper = ''
    if ref.is_file():
        rec = {r['id']: r['correct'] for r in records(ref)}
        vals = [rec[r['id']].count('1') / len(rec[r['id']]) for r in rows if r['id'] in rec]
        paper = f'; paper 2B initial student on these questions {100 * sum(vals) / len(vals):.1f}' if vals else ''
    print(f'  {s:8s} {100 * summ["mean_pass1"]:6.1f} avg@{SUITES[s].n} over {summ["problems"]} questions; '
          f'cap-hit {summ["cap_hit_rate"]:.3f}; template {str(m["chat_template_sha256"])[:12]}{paper}'
          + ('' if good else '  <-- malformed output'))
    ok = ok and good
    if nonzero and s in ('math500', 'ifeval') and summ['mean_pass1'] == 0:
        print(f'  {s}: score is exactly zero; the pipeline is probably broken'); ok = False
sys.exit(0 if ok else 1)
EOF
step "aggregate" "$PY" -m eval.aggregate --records "$RUNS" --cap "$CAP"
step "compare (a model against itself is exactly 0)" "$PY" - "$RUNS" "$NAME" "$CAP" "${SUITE_LIST[0]}" <<'EOF'
import sys
from eval.stats import Records, contrast
rec = Records.detect(sys.argv[1])
d, ci, _ = contrast(rec, sys.argv[2], rec, sys.argv[2], int(sys.argv[3]), (sys.argv[4],))
print(f'  delta {d} ci {ci}')
sys.exit(0 if d == 0 and ci == [0.0, 0.0] else 1)
EOF

if [ "$FAILED" -eq 0 ]; then echo "SMOKE PASS"; else echo "SMOKE FAIL ($FAILED step(s) failed)"; fi
exit $(( FAILED > 0 ))
