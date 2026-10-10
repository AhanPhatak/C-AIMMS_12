# Cached CTC graphs

`locomo/sample_{0..9}.json` -- the LoCoMo CTC graphs the eval otherwise rebuilds
with ~400-600 vLLM calls per conversation. Built on resiliente-2003 (the graphs
behind the Sep 2026 runs); relative dates are resolved in memory on load.

The eval reads them from `$CAIMMS_OUTPUT_DIR/graph_cache/`, so after cloning:

    source env.sh && mkdir -p "$CAIMMS_OUTPUT_DIR/graph_cache" && \
      cp -n cached_graphs/locomo/*.json "$CAIMMS_OUTPUT_DIR/graph_cache/"

`qasper_surprise_g1.5_m64_w400/` -- LongBench Qasper document graphs built with
surprise segmentation (gamma 1.5, min block 64 tokens, events capped at 400
words), first 50 rows. Docs 15 and 16 are deliberately ABSENT: on resiliente the
surprise pass OOM'd on them and the eval fell back to fixed chunking (fixed in
b8f775e), so those two must be rebuilt. The matching partial results (46/50 per
arm, doc 16 rows removed) are in `../cached_results/`. Restore both with:

    source env.sh && mkdir -p "$CAIMMS_OUTPUT_DIR/lb_graph_cache/qasper_surprise_g1.5_m64_w400" && \
      cp -n cached_graphs/qasper_surprise_g1.5_m64_w400/*.json "$CAIMMS_OUTPUT_DIR/lb_graph_cache/qasper_surprise_g1.5_m64_w400/" && \
      cp -n cached_results/lb_qasper_*_surprise_n50.jsonl "$CAIMMS_OUTPUT_DIR/"

then `LB_SEGMENTATION=surprise ARMS="combined hybrid vanilla" bash scripts/run_longbench.sh`
answers only the 4 missing docs per arm.

`hypermem/sample_{0..9}.json` -- the LoCoMo conversation hypergraphs for
`WORKMEM_RETRIEVER=hypermem` (topics / episodes / facts / hyperedges / dated
turns; see `adaptive_memory_structures/README_HYPERGRAPH.md`). Built
2026-10-09/10 with Qwen3-4B via vLLM for extraction and Qwen2.5-0.5B for
surprise segmentation (conv 0 on GPU, 1-9 on CPU); ~6h of vLLM calls to
rebuild. The eval reads them from `$CAIMMS_OUTPUT_DIR/hypermem_cache/`:

    source env.sh && mkdir -p "$CAIMMS_OUTPUT_DIR/hypermem_cache" && \
      cp -n cached_graphs/hypermem/*.json "$CAIMMS_OUTPUT_DIR/hypermem_cache/"

The matching conv-0 result rows are `../cached_results/locomo_c0_{iterret,hypermem_strict,hypermem_soft}.jsonl`.
