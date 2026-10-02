#!/usr/bin/env python3
"""
Build a HypergraphMemory index from a real LoCoMo conversation sample and
serialize the resulting topic/episode/fact hypergraph to JSON for the
visualizer (hypergraph_visualizer.html).

Usage
-----
    # List the 10 samples with their size, to help pick one
    python3 build_locomo_hypergraph.py --list

    # Build from sample 0 (Caroline/Melanie), its first 3 sessions
    python3 build_locomo_hypergraph.py --sample-index 0 --max-sessions 3

    # Real repo model instead of the small default
    python3 build_locomo_hypergraph.py --model Qwen/Qwen3-4B-Instruct-2507 --gamma 1.5

Prefer the wrapper script (build_locomo_hypergraph.sh) for this repo's
conda-env / env.sh setup; call this file directly once your own environment
is already active.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from qwen_client import QwenClient, QwenConfig, get_client
from memory_structures import EpisodicSession, HypergraphMemory
from locomo_loader import list_samples, load_sample

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "locomo10.json")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "hypergraph_output")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA_PATH, help=f"Path to locomo10.json (default: {DEFAULT_DATA_PATH})")
    p.add_argument("--list", action="store_true", help="List the 10 samples (id, speakers, size) and exit.")
    p.add_argument("--sample-index", type=int, default=0, help="Which of the 10 conversations to load (default: 0).")
    p.add_argument("--sample-id", default=None, help="Load by sample_id (e.g. conv-26) instead of --sample-index.")
    p.add_argument("--max-sessions", type=int, default=3,
                    help="Only load this many of the sample's sessions, in order (default: 3). "
                         "A full sample has 19-32 sessions / 369-689 turns -- unset (0) processes all "
                         "of them, which means many more LLM calls and a much longer run.")
    p.add_argument("--max-turns", type=int, default=None, help="Additionally cap total turns loaded.")

    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF model id or local path to load in-process (default: a small model; pass "
                         "Qwen/Qwen3-4B-Instruct-2507 to match this repo's intended model).")
    p.add_argument("--device", default="auto", help="cuda / cuda:1 / cpu / auto (default: auto)")
    p.add_argument("--dry-run", action="store_true", help="Skip loading a real model -- fallback paths only.")
    p.add_argument("--gamma", type=float, default=0.5,
                    help="Surprise threshold multiplier for episode segmentation (default: 0.5, lower "
                         "than HypergraphMemory's own 1.5 default -- see README_HYPERGRAPH.md's gamma "
                         "caveat for why).")
    p.add_argument("--min-block-size", type=int, default=1)

    p.add_argument("--query", action="append", default=[],
                    help="Optional: run a retrieval query against the built hypergraph and print the "
                         "top pages. Repeatable.")
    p.add_argument("--top-k", type=int, default=3)

    p.add_argument("-o", "--output", default=None,
                    help=f"Output JSON path (default: {DEFAULT_OUTPUT_DIR}/<sample_id>.json)")
    return p


def serialize_hypergraph(session_info: dict, session: EpisodicSession) -> dict:
    hg = session.hypergraph
    topics = hg.get("topics", {})
    episodes = hg.get("episodes", {})
    facts = hg.get("facts", {})
    episode_hyperedges = hg.get("episode_hyperedges", {})
    fact_hyperedges = hg.get("fact_hyperedges", {})
    page_to_episode = hg.get("page_to_episode", {})

    episode_topic: dict[str, str] = {}
    for he in episode_hyperedges.values():
        for eid in he.relation:
            episode_topic[eid] = he.topic_id

    return {
        "sample_id": session_info["sample_id"],
        "speakers": session_info["speakers"],
        "sessions_included": session_info["sessions_included"],
        "num_pages": len(session.pages),
        "topics": [
            {"id": t.id, "title": t.title, "summary": t.summary, "episode_ids": t.episode_ids}
            for t in topics.values()
        ],
        "episodes": [
            {
                "id": e.id,
                "subject": e.subject,
                "summary": e.summary,
                "page_ids": e.page_ids,
                "topic_id": episode_topic.get(e.id),
            }
            for e in episodes.values()
        ],
        "facts": [
            {
                "id": f.id,
                "content": f.content,
                "confidence": f.confidence,
                "episode_id": f.episode_id,
                "keywords": f.keywords,
                "temporal": f.temporal,
                "spatial": f.spatial,
                "role": next(iter(f.hyperedge.values()), None),
            }
            for f in facts.values()
        ],
        "episode_hyperedges": [
            {
                "id": he.id,
                "topic_id": he.topic_id,
                "relation": he.relation,
                "weights": he.weights,
                "coherence_score": he.coherence_score,
            }
            for he in episode_hyperedges.values()
        ],
        "fact_hyperedges": [
            {"id": he.id, "episode_id": he.episode_id, "relation": he.relation, "weights": he.weights}
            for he in fact_hyperedges.values()
        ],
        "pages": [
            {"id": p.page_id, "text": p.to_text(), "episode_id": page_to_episode.get(p.page_id)}
            for p in session.pages
        ],
    }


def main() -> None:
    args = build_argparser().parse_args()

    if args.list:
        for s in list_samples(args.data):
            print(f"[{s['index']}] {s['sample_id']}: {s['speakers'][0]}/{s['speakers'][1]} "
                  f"-- {s['num_sessions']} sessions, {s['total_turns']} turns")
        return

    max_sessions = args.max_sessions if args.max_sessions and args.max_sessions > 0 else None
    print(f"Loading LoCoMo sample (index={args.sample_index}, id={args.sample_id}, "
          f"max_sessions={max_sessions}) from {args.data} ...")
    session_info = load_sample(
        args.data,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
        max_sessions=max_sessions,
        max_turns=args.max_turns,
    )
    pages = session_info["pages"]
    print(f"Loaded sample {session_info['sample_id']!r} ({session_info['speakers'][0]}/"
          f"{session_info['speakers'][1]}): {len(session_info['sessions_included'])} sessions, "
          f"{len(pages)} pages")

    print(f"Loading QwenClient (dry_run={args.dry_run}, model={args.model!r}, device={args.device!r}) ...")
    QwenClient.load(model_path=args.model, config=QwenConfig(device=args.device), dry_run=args.dry_run, force_reload=True)
    qwen = get_client()

    session = EpisodicSession(pages=pages, structure_type="hypergraph")
    hgm = HypergraphMemory(gamma=args.gamma, min_block_size=args.min_block_size)

    print("Building hypergraph index (this makes several LLM calls per episode) ...")
    t0 = time.time()
    hgm.build_index(session)
    elapsed = time.time() - t0

    hg = session.hypergraph
    print(f"Done in {elapsed:.1f}s -- topics={len(hg.get('topics', {}))} "
          f"episodes={len(hg.get('episodes', {}))} facts={len(hg.get('facts', {}))}")

    for query in args.query:
        print(f"\n=== retrieve: {query!r} ===")
        query_emb = qwen.embed(query)
        for page in hgm.retrieve(session, query_emb, top_k=args.top_k):
            print(" -", page.user_text)

    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"{session_info['sample_id']}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = serialize_hypergraph(session_info, session)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote hypergraph JSON to {output_path}")
    print("Open hypergraph_visualizer.html and load this file to visualize it.")


if __name__ == "__main__":
    main()
