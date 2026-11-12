"""LLM-judge saved LoCoMo eval rows after the run, with iterret.llm_judge.

The default HyperMem run skips WORKMEM_JUDGE (the GPUs hold the answer and
retrieval models, with no room for a vLLM server), so this scores its rows
afterwards against any OpenAI-compatible server, e.g. vLLM serving Qwen3-4B:

  python -m vllm.entrypoints.openai.api_server --model $CAIMMS_MODEL_PATH \
      --served-model-name Qwen/Qwen3-4B-Instruct-2507 --port 8007 --max-model-len 8192
  python3 scripts/judge_rows.py outputs/workmem_hypermem.jsonl [more.jsonl ...]

Writes <rows>.judge.jsonl next to each input (sample_idx, q_idx, category, score,
judge) and prints overall and per-category accuracy. `judge` is True, False, or
an "ERR:..." string when the call failed; errors count as wrong.
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "IterRet"))
from iterret.llm_client import OpenAICompatibleLLMClient  # noqa: E402
from iterret.llm_judge import judge_answer  # noqa: E402

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+", help="eval output .jsonl files")
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8007/v1"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    llm = OpenAICompatibleLLMClient(base_url=args.base_url, model=args.model)

    def judge(row):
        try:
            return judge_answer(row["question"], str(row["gold_answer"]), str(row["prediction"]), llm)
        except Exception as e:  # keep going; the row is reported as an error
            return "ERR:" + str(e)[:80]

    for path in args.rows:
        rows = [r for r in map(json.loads, open(path)) if not r.get("skipped")]
        with ThreadPoolExecutor(args.workers) as ex:
            verdicts = list(ex.map(judge, rows))
        out = os.path.splitext(path)[0] + ".judge.jsonl"
        with open(out, "w") as f:
            for r, v in zip(rows, verdicts):
                f.write(json.dumps({"sample_idx": r["sample_idx"], "q_idx": r["q_idx"], "category": r["category"],
                                    "score": r["score"], "judge": v}) + "\n")
        by_cat = collections.defaultdict(list)
        for r, v in zip(rows, verdicts):
            by_cat[int(r["category"])].append(v is True)
        errors = sum(isinstance(v, str) for v in verdicts)
        acc = sum(v is True for v in verdicts) / max(len(verdicts), 1)
        cats = "  ".join(f"{CATEGORIES.get(c, c)} {sum(v) / len(v):.3f} (n={len(v)})" for c, v in sorted(by_cat.items()))
        print(f"{path}: n={len(rows)} judge {acc:.4f} errors {errors}\n  {cats}\n  -> {out}", flush=True)


if __name__ == "__main__":
    main()
