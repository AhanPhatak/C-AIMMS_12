#!/usr/bin/env bash
# IterRet + delta-mem on LongBench QA (default: Qasper, first 50 documents).
# One run per OSAM arm; each restarts vLLM through run_pipeline.sh. The first
# arm builds the document graphs (cached, shared by every later arm/adapter).
#
#   bash scripts/run_longbench.sh                                # combined + hybrid, released adapter
#   ARMS="combined" N=200 bash scripts/run_longbench.sh
#   CAIMMS_ADAPTER_DIR=$CAIMMS_WORKSPACE/models/delta-mem-iterret TAG=iterret ARMS=combined \
#       bash scripts/run_longbench.sh                            # retrained adapter
#   LB_MAX_EVIDENCE_TOKENS=1536 TAG=released_cap1536 ...         # cap evidence to the training budget
#   VLLM_PORT=8002 bash scripts/run_longbench.sh
#
# Arms: combined (S = prompt = IterRet evidence), hybrid (S = whole document,
# prompt = evidence), vanilla (S = prompt = whole document).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

export LB_TASK="${LB_TASK:-qasper}"
export LB_MAX_SAMPLES="${N:-50}"
ARMS="${ARMS:-combined hybrid}"
TAG="${TAG:-released}"
export LB_DATA="${LB_DATA:-${CAIMMS_OUTPUT_DIR}/longbench_data}"

if [ ! -s "${LB_DATA}/${LB_TASK}.jsonl" ]; then
    caimms_activate
    python3 -m deltamem.workmem.longctx_data fetch --out-dir "${LB_DATA}" --tasks "${LB_TASK}" \
        || { echo "LongBench download failed"; exit 1; }
fi

for ARM in ${ARMS}; do
    case "${ARM}" in combined|hybrid|vanilla) ;; *) echo "unknown arm '${ARM}'"; exit 1 ;; esac
    OUT="${CAIMMS_OUTPUT_DIR}/lb_${LB_TASK}_${ARM}_${TAG}_n${LB_MAX_SAMPLES}.jsonl"
    echo "================ ${LB_TASK} arm=${ARM} adapter=${CAIMMS_ADAPTER_DIR} -> ${OUT}"
    EVAL_MODULE=deltamem.workmem.eval_longbench_iterret WORKMEM_OSAM_MODE="${ARM}" \
    WORKMEM_OUTPUT_FILE="${OUT}" \
        bash "${HERE}/run_pipeline.sh" || echo "!! arm ${ARM} exited non-zero -- continuing"
done

echo
echo "================ summary (${LB_TASK}, ${TAG})"
for ARM in ${ARMS}; do
    OUT="${CAIMMS_OUTPUT_DIR}/lb_${LB_TASK}_${ARM}_${TAG}_n${LB_MAX_SAMPLES}.jsonl"
    [ -s "${OUT}" ] || { echo "${ARM}: no rows"; continue; }
    python3 - "${OUT}" "${ARM}" <<'PY'
import json, sys
rows = {}
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get("prediction") == "" and not r.get("skipped"):
        continue
    rows[r["idx"]] = r
rows = list(rows.values())
f1 = sum(r["score"] for r in rows) / max(1, len(rows))
sk = sum(1 for r in rows if r.get("skipped"))
ev = [r["n_evidence"] for r in rows if not r.get("skipped")]
print(f"{sys.argv[2]:>9}: n={len(rows)}  F1={f1:.4f}  skipped={sk}  "
      f"evidence/q={sum(ev) / max(1, len(ev)):.1f}")
PY
done
