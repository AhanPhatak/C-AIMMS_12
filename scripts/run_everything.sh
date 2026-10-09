#!/usr/bin/env bash
# One script, from a fresh clone to scores: environment, assets, dry-run check,
# the LoCoMo delta-Mem + IterRet eval, scoring, and the HyperMem hypergraph build.
#
#   bash scripts/run_everything.sh --smoke        # 1 conversation / 152 Q   (~1h)
#   bash scripts/run_everything.sh --subset 4     # 4 conversations / 584 Q
#   bash scripts/run_everything.sh --full         # 10 conversations / 1540 Q (~30-35h)
#   add --skip-hypermem / --skip-eval to run only one half
#
# Reference token-F1 (docs/HANDOFF_DELMEM_FORK.md): full 0.4142 (standard prompt),
# 584-set 0.4053, conv-0 (= --smoke) ~0.371. All measured before relative-date
# resolution (9c6f840) was ported, which upstream measured as a large temporal gain.
#
# Layout: models/ and outputs/ go in the repo's PARENT dir (override with
# CAIMMS_WORKSPACE). Conda env name defaults to "workmem" (CONDA_ENV_NAME).
#
# GPUs: picks the two cards with the most free memory (VLLM_GPU / EVAL_GPU to
# force), falls back to single-GPU co-location on a 1-card box, and sizes vLLM's
# --gpu-memory-utilization to what is actually free (CAIMMS_VLLM_GPU_MEM_UTIL to
# force). vLLM v1 counts other users' memory on the card against its budget, so
# on a shared card 0.85 fails with "KV cache is needed ... larger than available".
#
# NOTE: the HyperMem code (adaptive_memory_structures/) is NOT wired into the
# LoCoMo eval -- nothing under delta-Mem/ or IterRet/ imports it -- so it cannot
# change the F1. Step 7 validates it on its own: builds a hypergraph from LoCoMo
# conv-26 with surprise segmentation (small in-process LM) + fact/topic
# extraction by Qwen3-4B via vLLM, and runs two retrieval queries.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

MODE=smoke; SUBSET_N=""; DO_EVAL=1; DO_HYPERMEM=1
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) MODE=smoke ;;
        --full) MODE=full ;;
        --subset) MODE=subset; SUBSET_N="${2:?--subset needs N}"; shift ;;
        --skip-eval) DO_EVAL=0 ;;
        --skip-hypermem) DO_HYPERMEM=0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
    shift
done

source "${ROOT}/env.sh"
LOG_DIR="${CAIMMS_OUTPUT_DIR}"; mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ── 1. environment ────────────────────────────────────────────────────────────
echo "### [1/7] conda env '${CONDA_ENV_NAME}'"
if ! caimms_activate 2>/dev/null; then
    bash "${HERE}/setup_env.sh"
    caimms_activate
fi

# ── 2. assets ─────────────────────────────────────────────────────────────────
echo "### [2/7] model weights, adapter, LoCoMo, MiniLM"
bash "${HERE}/download_assets.sh"
# The hypergraph tools read LoCoMo from <repo>/data/ (gitignored duplicate).
mkdir -p "${ROOT}/data"
cp -n "${CAIMMS_DATA_FILE}" "${ROOT}/data/locomo10.json"

# ── 3. cached CTC graphs (saves ~400-600 vLLM calls per conversation) ─────────
echo "### [3/7] seeding graph cache"
mkdir -p "${CAIMMS_OUTPUT_DIR}/graph_cache"
cp -n "${ROOT}"/cached_graphs/locomo/*.json "${CAIMMS_OUTPUT_DIR}/graph_cache/"

# ── 4. no-GPU static check ────────────────────────────────────────────────────
echo "### [4/7] dry run"
(cd "${HERE}" && PYTHONPATH=".:${PYTHONPATH}" python3 dryrun_pipeline.py | tail -3)

# ── 5. GPU selection ──────────────────────────────────────────────────────────
echo "### [5/7] GPU selection"
mapfile -t BY_FREE < <(nvidia-smi --query-gpu=index,memory.free,memory.total \
    --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -nr)
NGPU="${#BY_FREE[@]}"
[ "${NGPU}" -ge 1 ] || { echo "ERROR: no GPU visible"; exit 1; }
if [ "${NGPU}" -ge 2 ] && [ "${CAIMMS_SINGLE_GPU:-0}" != "1" ]; then
    export VLLM_GPU="${VLLM_GPU:-$(cut -d, -f1 <<<"${BY_FREE[0]}")}"
    export EVAL_GPU="${EVAL_GPU:-$(cut -d, -f1 <<<"${BY_FREE[1]}")}"
    export CAIMMS_SINGLE_GPU=0
else
    export CAIMMS_SINGLE_GPU=1
    export CAIMMS_GPU_INDEX="${CAIMMS_GPU_INDEX:-$(cut -d, -f1 <<<"${BY_FREE[0]}")}"
    export VLLM_GPU="${CAIMMS_GPU_INDEX}" EVAL_GPU="${CAIMMS_GPU_INDEX}"
fi
if [ -z "${CAIMMS_VLLM_GPU_MEM_UTIL:-}" ]; then
    read -r FREE TOTAL < <(nvidia-smi --id="${VLLM_GPU}" --query-gpu=memory.free,memory.total \
        --format=csv,noheader,nounits | tr -d ',')
    if [ "${CAIMMS_SINGLE_GPU}" = "1" ]; then
        FREE=$((FREE - 11000))   # leave room for the eval's own model copy + adapter
    fi
    CAIMMS_VLLM_GPU_MEM_UTIL="$(python3 -c "print(min(0.85, round(($FREE - 600) / $TOTAL - 0.005, 2)))")"
    export CAIMMS_VLLM_GPU_MEM_UTIL
    python3 -c "import sys; sys.exit(0 if ${CAIMMS_VLLM_GPU_MEM_UTIL} >= 0.45 else 1)" || {
        echo "ERROR: only ${CAIMMS_VLLM_GPU_MEM_UTIL} of GPU ${VLLM_GPU} is free -- not enough for vLLM"
        echo "       (8GB weights + KV cache). Wait for other jobs, or force CAIMMS_VLLM_GPU_MEM_UTIL."
        exit 1; }
fi
echo "  vLLM on GPU ${VLLM_GPU} (util ${CAIMMS_VLLM_GPU_MEM_UTIL}), eval on GPU ${EVAL_GPU}, single=${CAIMMS_SINGLE_GPU}"

# ── 6. LoCoMo eval + score ────────────────────────────────────────────────────
if [ "${DO_EVAL}" = "1" ]; then
    echo "### [6/7] LoCoMo eval (${MODE})"
    case "${MODE}" in
        smoke)  ARGS=(--smoke); OUT="${CAIMMS_OUTPUT_DIR}/smoke_results.jsonl" ;;
        subset) export WORKMEM_MAX_SAMPLES="${SUBSET_N}"; ARGS=()
                OUT="${CAIMMS_OUTPUT_DIR}/workmem_iterret_n${SUBSET_N}.jsonl" ;;
        full)   ARGS=(); OUT="${CAIMMS_OUTPUT_DIR}/workmem_iterret_full.jsonl" ;;
    esac
    # Full/subset runs RESUME from their checkpoint -- archive it, or you get
    # the previous run's scores back (HANDOFF.md §8 trap 1). Smoke clears its own.
    if [ "${MODE}" != "smoke" ] && [ -f "${OUT}" ]; then
        mv "${OUT}" "${OUT%.jsonl}_archived_${STAMP}.jsonl"
        echo "  archived previous checkpoint"
    fi
    bash "${HERE}/run_pipeline.sh" "${ARGS[@]}"
    echo
    python3 "${HERE}/score_calculator.py" "${OUT}" | tee "${LOG_DIR}/score_${MODE}_${STAMP}.txt"
fi

# ── 7. HyperMem hypergraph build (separate from the eval) ─────────────────────
if [ "${DO_HYPERMEM}" = "1" ]; then
    echo "### [7/7] HyperMem: build hypergraph from LoCoMo conv-26 (3 sessions)"
    HM_LOG="${LOG_DIR}/hypermem_server_${STAMP}.log"
    # run_pipeline.sh shuts its vLLM down on exit, so start one for this step.
    CUDA_VISIBLE_DEVICES="${VLLM_GPU}" python3 -m vllm.entrypoints.openai.api_server \
        --model "${CAIMMS_MODEL_PATH}" --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
        --port "${VLLM_PORT}" --dtype bfloat16 --max-model-len 8192 \
        --gpu-memory-utilization "${CAIMMS_VLLM_GPU_MEM_UTIL}" --enforce-eager \
        --disable-log-requests > "${HM_LOG}" 2>&1 &
    HM_PID=$!
    trap 'kill ${HM_PID} 2>/dev/null || true' EXIT INT TERM
    for _ in $(seq 1 120); do
        curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && break
        kill -0 "${HM_PID}" 2>/dev/null || { echo "ERROR: vLLM died, see ${HM_LOG}"; tail -20 "${HM_LOG}"; exit 1; }
        sleep 5
    done
    curl -sf "http://localhost:${VLLM_PORT}/v1/models" | grep -q Qwen3-4B \
        || { echo "ERROR: vLLM not serving Qwen3-4B on port ${VLLM_PORT}"; exit 1; }
    export ITERRET_LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

    # Surprise segmentation + embeddings run in-process; 0.5B fits next to
    # anything. Set HYPERMEM_MODEL=${CAIMMS_MODEL_PATH} if a card has ~10GB spare.
    SEG_GPU="${EVAL_GPU}"
    [ "${CAIMMS_SINGLE_GPU}" = "1" ] && SEG_GPU="${VLLM_GPU}"
    (cd "${ROOT}/adaptive_memory_structures" && CUDA_VISIBLE_DEVICES="${SEG_GPU}" \
        python3 build_locomo_hypergraph.py --sample-index 0 --max-sessions 3 \
            --model "${HYPERMEM_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" --device cuda \
            --query "What happened at the LGBTQ support group?" \
            --query "What does Melanie like to paint?" \
            -o "hypergraph_output/conv-26_${STAMP}.json") \
        2>&1 | tee "${LOG_DIR}/hypermem_build_${STAMP}.log"
    kill "${HM_PID}" 2>/dev/null || true
    echo "  open adaptive_memory_structures/hypergraph_visualizer.html and load"
    echo "  adaptive_memory_structures/hypergraph_output/conv-26_${STAMP}.json"
fi

echo "### done. Logs and results in ${CAIMMS_OUTPUT_DIR}"
