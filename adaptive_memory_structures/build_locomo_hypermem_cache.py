"""
Pre-build the conversation hypergraphs the LoCoMo eval reads
(WORKMEM_RETRIEVER=hypermem) into <outputs>/hypermem_cache/sample_<i>.json,
so the eval itself never has to stop and build. Needs the Qwen3-4B vLLM
server up (fact/topic extraction) and one small in-process LM for surprise
segmentation.

    python3 build_locomo_hypermem_cache.py --samples 0-9
    python3 build_locomo_hypermem_cache.py --samples 1,2,3 --base-url http://localhost:8000/v1

Existing cache files are kept (delete one to rebuild it).
"""

from __future__ import annotations

import argparse
import json
import os

from locomo_hypermem import load_or_build


def _parse_samples(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default="0-9")
    p.add_argument("--data", default=os.environ.get("CAIMMS_DATA_FILE"))
    p.add_argument("--cache-dir", default=os.path.join(os.environ.get("CAIMMS_OUTPUT_DIR", "."), "hypermem_cache"))
    p.add_argument("--base-url", default=os.environ.get("CAIMMS_VLLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = p.parse_args()

    with open(args.data) as f:
        samples = json.load(f)
    for i in _parse_samples(args.samples):
        load_or_build(
            os.path.join(args.cache_dir, f"sample_{i}.json"), samples[i]["conversation"],
            args.base_url, args.model, log=lambda m, i=i: print(f"[sample {i}] {m}", flush=True),
        )


if __name__ == "__main__":
    main()
