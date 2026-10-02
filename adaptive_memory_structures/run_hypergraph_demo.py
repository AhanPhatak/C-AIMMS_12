#!/usr/bin/env python3
"""
Demo / smoke-test driver for HypergraphMemory.

Builds a small synthetic two-topic conversation (a Japan trip + a dog),
indexes it with HypergraphMemory (surprise segmentation -> LLM topic
clustering -> LLM fact extraction -> embedding propagation), prints the
resulting topic/episode/fact structure, then runs two differently-themed
retrieval queries against it so you can see the coarse-to-fine
topic -> episode -> fact traversal actually discriminate between them.

Usage
-----
    # Real model (downloads weights on first run if not cached)
    python3 run_hypergraph_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --gamma 0.5

    # No GPU / no model download -- exercises every fallback path only
    python3 run_hypergraph_demo.py --dry-run

Prefer the wrapper script (run_hypergraph_demo.sh) for the conda-env /
env.sh setup this repo expects; call this file directly once your own
environment is already active.
"""

from __future__ import annotations

import argparse

from qwen_client import QwenClient, QwenConfig, get_client
from memory_structures import EpisodicSession, HypergraphMemory, Page

DEMO_PAGES = [
    ("Hi, I'm planning a trip to Japan next spring.", "That sounds exciting! Tokyo or Kyoto?"),
    ("Mostly Kyoto, I love temples.", "Kyoto has Kinkaku-ji and Fushimi Inari, both great picks."),
    ("I also need to renew my passport before I go.", "Passport renewal usually takes 4-6 weeks in the US."),
    ("Switching gears -- my dog Max just turned 5 years old.", "Happy birthday to Max! What breed is he?"),
    ("He's a golden retriever, loves chasing squirrels in the park.", "Classic golden retriever behavior, very playful."),
    ("Max also knows how to fetch the newspaper every morning.", "That's an impressive trick for a dog to learn."),
    ("Back to Japan -- what's the best time to see cherry blossoms in Kyoto?", "Late March to early April is typically peak bloom in Kyoto."),
]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF model id or local path to load in-process (default: a small model "
                         "so the demo is runnable without heavy GPU/download cost; pass "
                         "Qwen/Qwen3-4B-Instruct-2507 to match this repo's intended model).")
    p.add_argument("--device", default="auto", help="cuda / cuda:1 / cpu / auto (default: auto)")
    p.add_argument("--dry-run", action="store_true",
                    help="Skip loading a real model entirely -- exercises every graceful-degradation "
                         "fallback path (no segmentation, deterministic summaries/facts) instead of "
                         "real model behaviour. Useful as a fast, GPU-free smoke test.")
    p.add_argument("--gamma", type=float, default=0.5,
                    help="Surprise threshold multiplier for episode segmentation (default: 0.5). "
                         "HypergraphMemory's own default is 1.5 (matching the paper's "
                         "CAIMMSBoundaryEmitter), but that was tuned for much longer sessions and "
                         "Qwen3-4B; on this demo's 7-page session it produces zero splits with a "
                         "small model, so this driver defaults lower to actually show segmentation. "
                         "Pass --gamma 1.5 to see that conservative behaviour yourself.")
    p.add_argument("--topic-top-k", type=int, default=2)
    p.add_argument("--episode-top-k", type=int, default=3)
    p.add_argument("--fact-top-k", type=int, default=5)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    print(f"Loading QwenClient (dry_run={args.dry_run}, model={args.model!r}, device={args.device!r}) ...")
    QwenClient.load(
        model_path=args.model,
        config=QwenConfig(device=args.device),
        dry_run=args.dry_run,
        force_reload=True,
    )
    qwen = get_client()

    pages = [Page(user_text=u, agent_text=a) for u, a in DEMO_PAGES]
    session = EpisodicSession(pages=pages, structure_type="hypergraph")
    hgm = HypergraphMemory(
        topic_top_k=args.topic_top_k,
        episode_top_k=args.episode_top_k,
        fact_top_k=args.fact_top_k,
        gamma=args.gamma,
        min_block_size=1,
    )

    print("\n=== build_index ===")
    hgm.build_index(session)
    hg = session.hypergraph
    print(f"topics={len(hg.get('topics', {}))} episodes={len(hg.get('episodes', {}))} "
          f"facts={len(hg.get('facts', {}))}")

    for topic in hg.get("topics", {}).values():
        print(f"\nTopic: {topic.title!r}")
        print(f"  summary: {topic.summary!r}")
        print(f"  episodes: {len(topic.episode_ids)}")

    for episode in hg.get("episodes", {}).values():
        print(f"\nEpisode: {episode.subject!r}")
        print(f"  summary: {episode.summary!r}")
        print(f"  pages: {len(episode.page_ids)}")

    for fact in hg.get("facts", {}).values():
        print(f"  fact: {fact.content!r} (confidence={fact.confidence})")

    for label, query in [
        ("Japan trip", "When should I visit Kyoto to see cherry blossoms?"),
        ("the dog", "Tell me about Max the dog."),
    ]:
        print(f"\n=== retrieve: query about {label} ===")
        query_emb = qwen.embed(query)
        for page in hgm.retrieve(session, query_emb, top_k=3):
            print(" -", page.user_text)

    print("\nDone.")


if __name__ == "__main__":
    main()
