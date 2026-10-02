#!/usr/bin/env bash
# Builds a HypergraphMemory index from a real LoCoMo conversation sample
# (data/locomo10.json) and writes it to hypergraph_output/<sample_id>.json
# for hypergraph_visualizer.html to render.
#
#   bash build_locomo_hypergraph.sh --list                       # list the 10 samples
#   bash build_locomo_hypergraph.sh --sample-index 0              # default: first 3 sessions
#   bash build_locomo_hypergraph.sh --sample-index 0 --max-sessions 0   # ALL sessions (slow, many LLM calls)
#   bash build_locomo_hypergraph.sh --dry-run                     # no GPU / no download, fallback paths only
#
# Any extra arguments are forwarded to build_locomo_hypergraph.py as-is (see
# `python3 build_locomo_hypergraph.py --help` for the full list).
#
# Env setup mirrors scripts/run_pipeline.sh / run_hypergraph_demo.sh: sources
# ../env.sh, then caimms_activate's the CONDA_ENV_NAME conda env (default
# "workmem"). Override if yours is named differently, e.g.:
#   CONDA_ENV_NAME=C-AIMMS bash build_locomo_hypergraph.sh --sample-index 0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
caimms_activate

cd "${HERE}"
python3 build_locomo_hypergraph.py "$@"
