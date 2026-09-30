#!/usr/bin/env bash
# Small OSAM-mode comparison: combined vs vanilla vs hybrid (+ vanilla S-only),
# judge ON (each row carries both token-F1 and judge_correct, so one run per mode
# gives both metrics -- predictions are identical with the judge off).
#
#   bash scripts/run_osam_modes.sh                 # 1 conversation, all 4 arms
#   N=4 bash scripts/run_osam_modes.sh             # 4 conversations
#   ARMS="vanilla_Sonly hybrid" bash scripts/run_osam_modes.sh
#   VLLM_PORT=8002 bash scripts/run_osam_modes.sh
#
# Arms:
#   combined       S = IterRet evidence,   prompt = IterRet evidence (default system)
#   vanilla        S = full conversation,  prompt = full conversation
#   vanilla_Sonly  S = full conversation,  prompt = none (paper-faithful delta-mem)
#   hybrid         S = full conversation,  prompt = IterRet evidence
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

N="${N:-1}"
ARMS="${ARMS:-combined vanilla vanilla_Sonly hybrid}"

for ARM in ${ARMS}; do
    case "${ARM}" in
        combined|vanilla|hybrid) MODE="${ARM}"; EIP=1 ;;
        vanilla_Sonly)           MODE="vanilla"; EIP=0 ;;
        *) echo "unknown arm '${ARM}'"; exit 1 ;;
    esac
    OUT="${CAIMMS_OUTPUT_DIR}/mode_${ARM}_n${N}.jsonl"
    echo "================ arm=${ARM} (mode=${MODE}, evidence_in_prompt=${EIP}) -> ${OUT}"
    WORKMEM_MAX_SAMPLES="${N}" WORKMEM_JUDGE=1 \
    WORKMEM_OSAM_MODE="${MODE}" OSAM_EVIDENCE_IN_PROMPT="${EIP}" \
    WORKMEM_OUTPUT_FILE="${OUT}" \
        bash "${HERE}/run_pipeline.sh" || echo "!! arm ${ARM} exited non-zero -- continuing"
done

echo
echo "================ summary"
for ARM in ${ARMS}; do
    OUT="${CAIMMS_OUTPUT_DIR}/mode_${ARM}_n${N}.jsonl"
    [ -s "${OUT}" ] || { echo "${ARM}: no rows"; continue; }
    python3 - "${OUT}" "${ARM}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
# drop failed-generation rows (empty prediction), keep the last row per question
rows = list({(r["sample_idx"], r["q_idx"]): r for r in rows
             if not (r.get("prediction") == "" and not r.get("skipped"))}.values())
rows = [r for r in rows if r.get("category") != 5]
f1 = [r.get("score") or 0.0 for r in rows]
j = [r["judge_correct"] for r in rows if "judge_correct" in r]
msg = f"{sys.argv[2]:>14}: n={len(rows)}  F1={sum(f1)/max(len(f1),1):.4f}"
if j:
    msg += f"  judge={sum(bool(x) for x in j)/len(j):.4f}"
print(msg)
PY
done
