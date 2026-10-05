# Cached CTC graphs

`locomo/sample_{0..9}.json` -- the LoCoMo CTC graphs the eval otherwise rebuilds
with ~400-600 vLLM calls per conversation. Built on resiliente-2003 (the graphs
behind the Sep 2026 runs); relative dates are resolved in memory on load.

The eval reads them from `$CAIMMS_OUTPUT_DIR/graph_cache/`, so after cloning:

    source env.sh && mkdir -p "$CAIMMS_OUTPUT_DIR/graph_cache" && \
      cp -n cached_graphs/locomo/*.json "$CAIMMS_OUTPUT_DIR/graph_cache/"
