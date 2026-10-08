"""Remove specific LongBench documents from a graph cache and from result files,
so the next run rebuilds and re-answers them.

  python3 scripts/lb_purge_docs.py --idx 15 16 \
      --cache-dir "$CAIMMS_OUTPUT_DIR/lb_graph_cache/qasper_surprise_g1.5_m64_w400" \
      --results "$CAIMMS_OUTPUT_DIR"/lb_qasper_*_surprise_n50.jsonl

Each results file is backed up to <file>.bak-<timestamp> before rewriting.
Document ids are LongBench row indices (the "idx" field); cache files are found
from the row's context hash, exactly as eval_longbench_iterret keys them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--task", default="qasper")
    ap.add_argument("--lb-data", default=os.path.join(os.environ.get("CAIMMS_OUTPUT_DIR", "."), "longbench_data"))
    a = ap.parse_args()

    with open(os.path.join(a.lb_data, f"{a.task}.jsonl")) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    drop = set(a.idx)
    for i in sorted(drop):
        key = hashlib.md5(rows[i]["context"].encode("utf-8")).hexdigest()[:16]  # = longctx_data.doc_key
        path = os.path.join(a.cache_dir, f"{key}.json")
        if os.path.exists(path):
            os.remove(path)
            print(f"removed graph idx={i} {path}")
        else:
            print(f"no graph for idx={i} ({path})")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for f in a.results:
        shutil.copy2(f, f"{f}.bak-{stamp}")
        kept, removed = [], 0
        with open(f) as fh:
            for line in fh:
                try:
                    if json.loads(line).get("idx") in drop:
                        removed += 1
                        continue
                except json.JSONDecodeError:
                    pass
                kept.append(line)
        with open(f, "w") as fh:
            fh.writelines(kept)
        print(f"{f}: removed {removed} row(s), kept {len(kept)} (backup .bak-{stamp})")


if __name__ == "__main__":
    main()
