#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Export one FSDP checkpoint of a Qwen3.5 run to a Hugging Face directory (the "passthrough" export of
# miles/tools/convert_fsdp_to_hf.py: the origin repo's key names and the checkpoint's dtype; Qwen3_5ForConditionalGeneration
# without the 15 multi-token-prediction tensors, which the trainer never holds). CPU only.
#
#   bash recipes/qwen3.5/export_hf.sh RUN_DIR ITER SIZE [OUT_DIR]
#     RUN_DIR  the training directory (holds iter_XXXXXXX/)      ITER  the iteration to export
#     OUT_DIR  default: <RUN_DIR>_hf_iter<ITER>
# A single-file export (2B, 4B) also gets a model.safetensors.index.json generated from its own tensor names.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR="${1:?usage: export_hf.sh RUN_DIR ITER SIZE [OUT_DIR]}"
ITER="${2:?usage: export_hf.sh RUN_DIR ITER SIZE [OUT_DIR]}"
SIZE="${3:?usage: export_hf.sh RUN_DIR ITER SIZE [OUT_DIR]}"
check_size "${SIZE}"
OUT="${4:-${RUN_DIR}_hf_iter${ITER}}"
SRC="${RUN_DIR}/iter_$(printf %07d "${ITER}")"
ORIGIN="$(base_model "${SIZE}")" || die "no base model for ${SIZE}"

[ -f "${SRC}/meta.json" ] || die "no complete checkpoint at ${SRC}"
if [ -f "${OUT}/model.safetensors.index.json" ] && [ -f "${OUT}/config.json" ]; then
    log "export exists: ${OUT}"; exit 0
fi
TMP="${OUT}.partial_$$"
rm -rf "${TMP}"
mkdir -p "$(dirname "${OUT}")"
log "exporting ${SRC} -> ${OUT}"
PYTHONPATH="${MILES_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" "${MILES_DIR}/tools/convert_fsdp_to_hf.py" \
    --input-dir "${SRC}" --output-dir "${TMP}" --origin-hf-dir "${ORIGIN}" -f > "${TMP}.log" 2>&1 \
    || { tail -n 30 "${TMP}.log" >&2; rm -rf "${TMP}"; die "conversion failed (log: ${TMP}.log)"; }
grep -aE "passthrough|NOTE|Refusing" "${TMP}.log" | sed 's/^/  /' || true
if [ -f "${TMP}/model.safetensors" ] && [ ! -f "${TMP}/model.safetensors.index.json" ]; then
    "${PYTHON}" - "${TMP}" <<'PY'
import json, os, sys
from safetensors import safe_open
d = sys.argv[1]
f = os.path.join(d, "model.safetensors")
with safe_open(f, "pt") as h:
    keys = list(h.keys())
json.dump({"metadata": {"total_size": os.path.getsize(f)}, "weight_map": {k: "model.safetensors" for k in sorted(keys)}},
          open(os.path.join(d, "model.safetensors.index.json"), "w"), indent=1)
print(f"  single-file export: index written for {len(keys)} tensors")
PY
fi
[ -f "${TMP}/model.safetensors.index.json" ] && [ -f "${TMP}/config.json" ] || { rm -rf "${TMP}"; die "incomplete export"; }
rm -rf "${OUT}"
mv "${TMP}" "${OUT}"
mv "${TMP}.log" "${OUT}/export.log"
log "EXPORT_OK ${OUT}"
