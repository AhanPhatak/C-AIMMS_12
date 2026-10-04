#!/usr/bin/env bash
# Build delta-mem training episodes from IterRet evidence on Qasper TRAIN papers
# (deltamem/workmem/build_iterret_sft_data.py), then prepare the trainer file.
# Needs vLLM (graph building + IterRet routing), so it runs through
# run_pipeline.sh. Resumable: re-run the same command after a kill.
#
#   bash scripts/build_sft_data.sh                      # 300 papers (~900 questions)
#   SFT_MAX_PAPERS=0 bash scripts/build_sft_data.sh     # all ~880 train papers
#   MAX_WRITE_TOKENS=1536 bash scripts/build_sft_data.sh   # evidence budget per episode
#   VLLM_PORT=8002 bash scripts/build_sft_data.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
caimms_activate

QASPER_DIR="${CAIMMS_OUTPUT_DIR}/qasper_raw"
export SFT_QASPER_TRAIN="${SFT_QASPER_TRAIN:-${QASPER_DIR}/qasper-train-v0.3.json}"
if [ ! -s "${SFT_QASPER_TRAIN}" ]; then
    mkdir -p "${QASPER_DIR}"
    echo "Downloading Qasper train/dev (allenai) ..."
    curl -fsSL https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz \
        | tar xz -C "${QASPER_DIR}" || { echo "Qasper download failed"; exit 1; }
fi

# The LongBench Qasper eval set: papers found in it are excluded from training.
export LB_DATA="${LB_DATA:-${CAIMMS_OUTPUT_DIR}/longbench_data}"
if [ ! -s "${LB_DATA}/qasper.jsonl" ]; then
    python3 -m deltamem.workmem.longctx_data fetch --out-dir "${LB_DATA}" --tasks qasper \
        || { echo "LongBench download failed"; exit 1; }
fi

export SFT_MAX_PAPERS="${SFT_MAX_PAPERS:-300}"
export SFT_WORKERS="${SFT_WORKERS:-4}"
export SFT_OUT="${SFT_OUT:-${CAIMMS_OUTPUT_DIR}/sft_iterret_qasper.jsonl}"
MAX_WRITE_TOKENS="${MAX_WRITE_TOKENS:-1536}"
TRAIN_FILE="${TRAIN_FILE:-${CAIMMS_OUTPUT_DIR}/sft_iterret_qasper_train_w${MAX_WRITE_TOKENS}.jsonl}"

EVAL_MODULE=deltamem.workmem.build_iterret_sft_data WORKMEM_OUTPUT_FILE="${SFT_OUT}" \
    bash "${HERE}/run_pipeline.sh" || { echo "episode build exited non-zero"; exit 1; }

python3 -m deltamem.workmem.build_iterret_sft_data prepare \
    --in "${SFT_OUT}" --out "${TRAIN_FILE}" \
    --tokenizer "${CAIMMS_MODEL_PATH}" --max-write-tokens "${MAX_WRITE_TOKENS}"
echo "Training file: ${TRAIN_FILE}"
