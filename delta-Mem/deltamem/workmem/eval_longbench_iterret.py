"""IterRet + delta-mem on LongBench QA (default: Qasper).

The LongBench counterpart of eval_locomo_iterret_mock.py. Per row: build (or
load) the document's CTC graph, run IterRet, then answer in one OSAM mode:

  combined : S = IterRet evidence,  prompt = IterRet evidence
  hybrid   : S = whole document,    prompt = IterRet evidence
  vanilla  : S = whole document,    prompt = whole document

The document is split into ~180-word passages (iterret.doc_memory_builder); in
vanilla/hybrid "whole document" is those passages in order. Prompt = LongBench's
official template split at {context}; metric = LongBench qa_f1_score.

Environment (all optional except the paths env.sh already sets):
  LB_TASK              qasper | narrativeqa | multifieldqa_en | hotpotqa | 2wikimqa | musique
  LB_DATA              <task>.jsonl or a dir of them (else the Hub data.zip)
  LB_MAX_SAMPLES       first N rows (default 50; 0 = all)
  WORKMEM_OSAM_MODE    combined | hybrid | vanilla (default combined)
  WORKMEM_OUTPUT_FILE  results JSONL (resumable)
  LB_GRAPH_CACHE_DIR   default <output dir>/lb_graph_cache/<task>
  CAIMMS_ADAPTER_DIR   delta-mem adapter (e.g. the IterRet-retrained one)
  LB_SEGMENTATION      fixed (~180-word passages, default) | surprise (EM-LLM
                       surprise boundaries from CAIMMS_MODEL_PATH; events capped
                       at LB_SURPRISE_MAX_WORDS). Knobs: LB_SURPRISE_GAMMA (1.5),
                       LB_SURPRISE_MIN_BLOCK (64 tokens), LB_SURPRISE_MAX_WORDS (400).

Graphs are built in a PRE-PASS before the delta-mem model loads, so the surprise
model (its own Qwen3-4B copy) and the delta-mem model never share the GPU.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.eval.common import attach_delta_adapter_in_place
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.longctx_data import (
    MAX_NEW_TOKENS, doc_key, format_query, load_longbench, qa_f1_score,
)
from deltamem.workmem.longctx_retrieval import (
    cap_evidence_by_tokens, get_or_build_doc_graph, graph_cache_path, make_backend, retrieve_evidence,
)
from deltamem.workmem.osam_workmem import answer_with_modes
from iterret.llm_client import OpenAICompatibleLLMClient

_ROOT = os.environ.get("CAIMMS_ROOT", ".")
MODEL_PATH = os.environ.get("CAIMMS_MODEL_PATH", f"{_ROOT}/models/Qwen3-4B-Instruct-2507")
ADAPTER_DIR = os.environ.get("CAIMMS_ADAPTER_DIR", f"{_ROOT}/models/delta-mem-adapter")
VLLM_BASE_URL = os.environ.get("CAIMMS_VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

TASK = os.environ.get("LB_TASK", "qasper")
LB_DATA = os.environ.get("LB_DATA") or None
MAX_SAMPLES = int(os.environ.get("LB_MAX_SAMPLES", "50"))
OSAM_MODE = os.environ.get("WORKMEM_OSAM_MODE", "combined")
OUTPUT_FILE = os.environ.get("WORKMEM_OUTPUT_FILE", f"{_ROOT}/outputs/lb_{TASK}_{OSAM_MODE}.jsonl")
SEGMENTATION = os.environ.get("LB_SEGMENTATION", "fixed")
SURPRISE_GAMMA = float(os.environ.get("LB_SURPRISE_GAMMA", "1.5"))
SURPRISE_MIN_BLOCK = int(os.environ.get("LB_SURPRISE_MIN_BLOCK", "64"))
SURPRISE_MAX_WORDS = int(os.environ.get("LB_SURPRISE_MAX_WORDS", "400"))
# One cache per segmentation (the cache is keyed by document only). "fixed"
# keeps the original location so existing caches are reused.
_SEG_SUFFIX = ("" if SEGMENTATION == "fixed"
               else f"_surprise_g{SURPRISE_GAMMA:g}_m{SURPRISE_MIN_BLOCK}_w{SURPRISE_MAX_WORDS}")
GRAPH_CACHE_DIR = Path(os.environ.get(
    "LB_GRAPH_CACHE_DIR", str(Path(OUTPUT_FILE).parent / "lb_graph_cache" / f"{TASK}{_SEG_SUFFIX}")))
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("WORKMEM_MAX_CONSECUTIVE_FAILURES", "5"))
# Optional cap on IterRet evidence (tokens, most relevant first). 0 = no cap.
# Set it to the training write budget when comparing a retrained adapter, and
# use the SAME cap for the released adapter so the comparison stays fair.
MAX_EVIDENCE_TOKENS = int(os.environ.get("LB_MAX_EVIDENCE_TOKENS", "0"))

if OSAM_MODE not in ("combined", "hybrid", "vanilla"):
    raise SystemExit(f"[FATAL] unknown WORKMEM_OSAM_MODE={OSAM_MODE!r} (expected combined | hybrid | vanilla)")
if SEGMENTATION not in ("fixed", "surprise"):
    raise SystemExit(f"[FATAL] unknown LB_SEGMENTATION={SEGMENTATION!r} (expected fixed | surprise)")
if TASK not in MAX_NEW_TOKENS:
    raise SystemExit(f"[FATAL] unsupported LB_TASK={TASK!r}")


def _load_done(path: str) -> dict:
    done = {}
    if Path(path).exists():
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written last line from a killed run
                # failed generations are never final (see the LoCoMo eval)
                if row.get("prediction") == "" and not row.get("skipped"):
                    continue
                done[row["idx"]] = row
    return done


def _write(row: dict) -> None:
    with open(OUTPUT_FILE, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _summary(rows: list) -> None:
    if not rows:
        print("No results.", flush=True)
        return
    f1 = [r["score"] for r in rows]
    answered = [r for r in rows if not r.get("skipped")]
    unans = [r for r in rows if any(a.strip().lower() == "unanswerable" for a in r["answers"])]
    ans = [r for r in rows if r not in unans]
    print("=" * 60, flush=True)
    print(f"{TASK} | mode={OSAM_MODE} | segmentation={SEGMENTATION} | adapter={ADAPTER_DIR}", flush=True)
    print(f"F1 (all {len(rows)}):            {sum(f1) / len(f1):.4f}", flush=True)
    if answered:
        print(f"F1 (answered {len(answered)}):        {sum(r['score'] for r in answered) / len(answered):.4f}", flush=True)
    if unans:
        print(f"  gold unanswerable ({len(unans)}): {sum(r['score'] for r in unans) / len(unans):.4f}", flush=True)
    if ans:
        print(f"  gold answerable   ({len(ans)}): {sum(r['score'] for r in ans) / len(ans):.4f}", flush=True)
    print(f"skipped (no evidence): {len(rows) - len(answered)}", flush=True)
    print("=" * 60, flush=True)


def _build_missing_graphs(rows: list, llm) -> None:
    """Build every uncached graph before the delta-mem model is loaded. For
    surprise segmentation this is the only time the surprise model is on the
    GPU; it is freed before generation starts."""
    missing, seen = [], set()
    for r in rows:
        key = doc_key(r["context"])
        if key not in seen and not graph_cache_path(GRAPH_CACHE_DIR, key).exists():
            seen.add(key)
            missing.append(r)
    if not missing:
        return
    print(f"[graphs] building {len(missing)} {SEGMENTATION} graphs -> {GRAPH_CACHE_DIR}", flush=True)
    segment_fn = None
    segmenter = None
    if SEGMENTATION == "surprise":
        from iterret.doc_segmenter import DocumentSurpriseSegmenter
        segmenter = DocumentSurpriseSegmenter(MODEL_PATH, gamma=SURPRISE_GAMMA,
                                              min_block_size=SURPRISE_MIN_BLOCK, device="cuda:0")

        def segment_fn(text: str):
            return segmenter.segment_document(text, max_words=SURPRISE_MAX_WORDS)
    try:
        for i, r in enumerate(missing):
            try:
                graph, passages, _ = get_or_build_doc_graph(
                    r["context"], doc_key(r["context"]), GRAPH_CACHE_DIR, llm,
                    segment_fn=segment_fn, segmentation=SEGMENTATION)
                words = [len(p.split()) for p in passages]
                print(f"[graphs] {i + 1}/{len(missing)} idx={r['idx']}: {len(passages)} units "
                      f"(words/unit mean {sum(words) / max(1, len(words)):.0f}, max {max(words, default=0)}), "
                      f"{graph.meta.get('n_semantic', 0)} facts", flush=True)
            except Exception as exc:  # noqa: BLE001 -- retried in the main loop / next run
                print(f"[graphs] idx={r['idx']} FAILED: {exc}", flush=True)
    finally:
        if segmenter is not None:
            del segmenter
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    print(f"[init] task={TASK} mode={OSAM_MODE} max_samples={MAX_SAMPLES} out={OUTPUT_FILE}", flush=True)
    print(f"[init] adapter={ADAPTER_DIR} graph_cache={GRAPH_CACHE_DIR} "
          f"max_evidence_tokens={MAX_EVIDENCE_TOKENS or 'off'}", flush=True)
    rows = load_longbench(TASK, LB_DATA)
    if MAX_SAMPLES > 0:
        rows = rows[:MAX_SAMPLES]
    done = _load_done(OUTPUT_FILE)
    todo = [r for r in rows if r["idx"] not in done]
    print(f"[init] {len(rows)} rows, {len(done)} already done, {len(todo)} to run", flush=True)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    if todo:
        llm = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)
        _build_missing_graphs(todo, llm)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True)
        attach_delta_adapter_in_place(model, Path(ADAPTER_DIR))
        model.eval()
        backend = make_backend()

    consecutive_failures = 0
    for row in todo:
        idx, question = row["idx"], row["question"]
        try:
            graph, passages, cached = get_or_build_doc_graph(row["context"], doc_key(row["context"]),
                                                             GRAPH_CACHE_DIR, llm, segmentation=SEGMENTATION)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}] graph build FAILED: {exc}", flush=True)
            continue  # not checkpointed -> retried on resume
        diag: dict = {}
        try:
            evidence = retrieve_evidence(question, graph, backend, llm, diag=diag)
            diag["n_evidence_uncapped"] = len(evidence)
            evidence = cap_evidence_by_tokens(evidence, tokenizer, MAX_EVIDENCE_TOKENS)
        except Exception as exc:  # noqa: BLE001
            # e.g. vLLM down: NOT "no evidence" -- leave the row for the next run.
            print(f"[{idx}] IterRet FAILED: {exc}", flush=True)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(f"[FATAL] {consecutive_failures} consecutive failures -- aborting; "
                                 "check the vLLM server and re-run (finished rows are kept).")
            continue

        base = {"idx": idx, "id": row["id"], "task": TASK, "question": question,
                "answers": row["answers"], "osam_mode": OSAM_MODE, "adapter_dir": ADAPTER_DIR,
                "segmentation": SEGMENTATION,
                "max_evidence_tokens": MAX_EVIDENCE_TOKENS,
                "n_passages": len(passages), "n_evidence": len(evidence), "graph_cached": cached,
                "retrieval": diag}

        if not evidence and OSAM_MODE == "combined":
            _write({**base, "prediction": "", "score": 0.0, "skipped": True, "reason": "no_evidence"})
            print(f"[{idx}] no evidence, skipped", flush=True)
            continue

        if OSAM_MODE == "combined":
            s_content, prompt_content = evidence, evidence
        elif OSAM_MODE == "hybrid":
            s_content, prompt_content = passages, evidence
        else:
            s_content, prompt_content = passages, passages

        session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
        session.reset()
        try:
            out = answer_with_modes(session, question, s_content=s_content, prompt_content=prompt_content,
                                    formatted_query=format_query(TASK, question),
                                    max_new_tokens=MAX_NEW_TOKENS[TASK])
            prediction = str(out.get("response") or out.get("assistant") or out.get("text") or "").strip() \
                if isinstance(out, dict) else ""
            if not prediction:
                prediction = next((m.get("content", "") for m in reversed(session.messages)
                                   if m.get("role") == "assistant"), "").strip()
            osam = (out.get("prompt_output_ratio_stats") or {}) if isinstance(out, dict) else {}
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}] generation FAILED: {exc}", flush=True)
            del session
            torch.cuda.empty_cache()
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(f"[FATAL] {consecutive_failures} consecutive generation failures -- "
                                 "aborting; fix the cause and re-run (finished rows are kept).")
            continue
        consecutive_failures = 0
        del session
        torch.cuda.empty_cache()

        score = qa_f1_score(prediction, row["answers"])
        _write({**base, "prediction": prediction, "score": score, "skipped": False, "osam_contribution": osam})
        ratio = osam.get("mean_delta_o_ratio")
        print(f"[{idx}] F1={score:.3f} n_ev={len(evidence)}/{len(passages)} "
              f"osam={ratio if ratio is None else round(ratio, 4)} pred={prediction[:70]!r} "
              f"gold={row['answers'][0][:40]!r}", flush=True)
        gc.collect()

    final = _load_done(OUTPUT_FILE)
    _summary([final[r["idx"]] for r in rows if r["idx"] in final])


if __name__ == "__main__":
    main()
