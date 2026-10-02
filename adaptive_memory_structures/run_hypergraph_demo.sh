#!/usr/bin/env bash
# Runs run_hypergraph_demo.py under this repo's standard env setup.
#
#   bash run_hypergraph_demo.sh                    # real small model, real segmentation
#   bash run_hypergraph_demo.sh --dry-run           # no GPU / no download, fallback paths only
#   bash run_hypergraph_demo.sh --model Qwen/Qwen3-4B-Instruct-2507 --gamma 1.5
#
# Any extra arguments are forwarded to run_hypergraph_demo.py as-is (see
# `python3 run_hypergraph_demo.py --help` for the full list).
#
# Env setup mirrors scripts/run_pipeline.sh: sources ../env.sh, then
# caimms_activate's the CONDA_ENV_NAME conda env (default "workmem", per
# env.sh). If your box doesn't have that env, point CONDA_ENV_NAME at one
# that does -- e.g. this was validated against a "C-AIMMS" env on the
# resiliente-2003 workstation:
#
#   CONDA_ENV_NAME=C-AIMMS bash run_hypergraph_demo.sh --dry-run
#
# That env needs numpy, torch, transformers, accelerate (installed
# separately -- it wasn't already present) at minimum; `openai` is optional
# (only used when a real vLLM server + ITERRET_LLM_BASE_URL is configured,
# see README_HYPERGRAPH.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
caimms_activate

cd "${HERE}"
python3 run_hypergraph_demo.py "$@"
