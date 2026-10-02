# HypergraphMemory

Replaces `GraphMemory` (a plain pairwise page-similarity graph) with a real
3-layer hypergraph memory modeled on the **HyperMem** paper
(`Hypergraph Memory for Long-Term Conversations`), adapted to this repo's
per-session (`EpisodicSession`) architecture. This is a **swap, not an
addition** — `GraphMemory` no longer exists; every place that used to
dispatch on `structure_type == "graph"` now dispatches on `"hypergraph"`.

## Why

Before this change, "hypergraph" was a dead label — a `0 is hypergraph`
comment in three classifier/dataset scripts that no code path ever acted on.
The only real structure resembling a graph was `GraphMemory`: nodes = pages,
edges = pairs with cosine similarity > 0.6, retrieval = seed node + 1-hop
expansion. It had no notion of hyperedges, topics, or facts.

## Architecture

```
Session pages  --[surprise segmentation]-->  Episodes  --[LLM streaming topic match]-->  Topics
                                                  |
                                          [LLM fact extraction]
                                                  v
                                                Facts
```

- **Topic** (L3) — an LLM-formed cluster of episodes. A session can contain
  several topics. Formed by *streaming* over episodes in temporal order:
  each new episode is LLM-matched against topics formed so far; on a match
  the topic is LLM-updated and its `EpisodeHyperedge` (role + importance
  weight per member episode) is rebuilt over the full new membership; on no
  match a new topic is seeded. This mirrors HyperMem's live-arrival
  `topic_extractor.py` algorithm even though here the "arrivals" are
  already-known episodes from one batch segmentation pass, per explicit
  request rather than a simpler one-shot clustering call.
- **Episode** (L2) — a contiguous span of `Page`s, found by the surprise
  mechanism (see below). Each episode gets an LLM-written subject/summary
  and its own `FactHyperedge` (role + weight per fact extracted from it).
- **Fact** (L1) — an atomic, LLM-extracted claim from one episode's text
  (content, confidence, temporal/spatial hints, keywords, query patterns).

Retrieval is coarse-to-fine, entirely inside the one session `MTEM` already
selected (`memory_layers.py`'s existing cross-session ranking is unchanged
and acts as the coarser pre-filter above all of this):

1. Score every topic by cosine(query, topic embedding); keep top `topic_top_k`.
2. **Hard-filter** to episodes whose `EpisodeHyperedge.topic_id` is among the
   selected topics (a connectivity filter, not a soft score); score by
   cosine(query, propagated episode embedding); keep top `episode_top_k`.
3. **Hard-filter** to facts whose `FactHyperedge.episode_id` is among the
   selected episodes; score by cosine(query, propagated fact embedding); keep
   top `fact_top_k`.
4. Map facts back to their episode's member pages, dedupe, return `top_k`
   `Page`s (backfilling from the next-best episode if short).

Embeddings are updated once before scoring via HyperMem's attention-weighted
hyperedge propagation: `edge_embedding = Σ softmax(hyperedge.weights)[i] *
member_embedding[i]`, then `node.embedding += alpha * edge_embedding`
(`alpha = 0.5` default) — applied per fact hyperedge and per episode
hyperedge.

### Episode segmentation: the surprise mechanism, not HyperMem's own

HyperMem's own episode detector (`conv_episode_extractor.py`) is an LLM
streaming boundary detector. Per explicit request, that's swapped out for
this repo's own **surprise + graph-modularity** mechanism (originally in
`Suprise Boundary Creator Module/`, which was orphaned — no working code path
called it before this change). The core math (token surprisal via next-token
cross-entropy, a rolling mean+std spike threshold, then a modularity-based
boundary-snapping pass over transformer Key-state similarity) is ported
essentially unchanged into `surprise_episode_segmenter.py`; the one real
adaptation is granularity — since a `Page` is one atomic turn that can't be
split mid-page, token-level surprisal and Key-states are pooled to
*per-page* values from a single forward pass over the whole session
transcript, rather than run through the original streaming, multi-chunk
history-buffer wrapper (`StatefulSurpriseBoundary`), which exists to carry
state across chunks this batch use case doesn't have.

## Files

New, in `adaptive_memory_structures/`:

| File | What it is |
|---|---|
| `hypergraph_types.py` | `TopicNode` / `EpisodeNode` / `FactNode` + `EpisodeHyperedge` / `FactHyperedge` dataclasses, plus `FactRole` / `EpisodeRole` enums |
| `surprise_episode_segmenter.py` | The surprise/modularity boundary math (ported) + `PageEpisodeSegmenter` (the page-granularity adaptation) |
| `hypergraph_extraction.py` | LLM-driven fact extraction and streaming topic formation, all JSON-prompted with retry-then-fallback parsing |
| `hypergraph_embedding.py` | Numpy port of the attention-weighted hyperedge embedding propagation formula |
| `vllm_llm_client.py` | Minimal OpenAI-compatible client for the vLLM server this repo already runs, shaped like `QwenClient.chat()` so either client works interchangeably |
| `run_hypergraph_demo.py` / `.sh` | Demo/smoke-test driver on a small synthetic conversation (see **Running it**) |
| `locomo_loader.py` | Loads a sample from `data/locomo10.json`, converts its turns into `Page`s |
| `build_locomo_hypergraph.py` / `.sh` | Builds a hypergraph from a real LoCoMo sample, writes `hypergraph_output/<sample_id>.json` (see **Building a hypergraph from real LoCoMo data**) |
| `hypergraph_visualizer.html` | Interactive topic/episode/fact visualization of a built hypergraph (see **Visualizing it**) |

Changed:

| File | What changed |
|---|---|
| `memory_structures.py` | `GraphMemory` removed, `HypergraphMemory` added (`build_index`/`retrieve`, same signatures as the other structures); `EpisodicSession.graph_index` replaced with `EpisodicSession.hypergraph` |
| `memory_layers.py` | `DefaultSelector` cycle, `MTEM._rebuild_structure_index`, `MTEM.retrieve` — `"graph"`/`GraphMemory` → `"hypergraph"`/`HypergraphMemory` |
| `best_structure_evaluator.py` | Same dispatch swap, plus `_build_memory_layout_string`'s hypergraph branch (`[Topic t / Episode e -> facts f1,f2] page text`) |
| `pipeline.py` | `_count_structures` stats dict key |
| `qwen_client.py` | Added `.model` / `.tokenizer` read-only properties so the segmenter reuses the already-loaded LM instead of loading a second copy |

## Config knobs

All on `HypergraphMemory.__init__`: `topic_top_k` (2), `episode_top_k` (3),
`fact_top_k` (5), `alpha` (0.5), `topic_match_batch_size` (10), plus the
segmenter's `gamma` (1.5), `n_local` (4096), `n_init` (128),
`min_block_size` (8), `similarity_refinement` (True) — all much smaller than
HyperMem's corpus-scale defaults since these traverse one session, not a
whole conversation history.

## Scope decisions

- **Retrieval scoring** is pure dense cosine at each layer plus the hard
  hyperedge connectivity filter — no BM25/RRF fusion, no reranker. This repo
  has no `rank_bm25` dependency, and that machinery is retrieval-quality
  plumbing orthogonal to "is it a hypergraph." Can be added later.
- **Rebuild cost**: like the other structures, `build_index` fully rebuilds
  every time it's called (`memory_layers.py` calls it on every page ingested
  into a session). For `HypergraphMemory` that means re-running segmentation
  + several LLM calls each time — materially more expensive than
  `GraphMemory`'s pure-cosine rebuild. Accepted for now (this is an
  eval/research pipeline, not a low-latency product).
- **LLM/embedding client**: embeddings always go through the existing
  `QwenClient.embed()`. LLM-JSON calls (fact extraction, role/weight
  assignment, topic matching) prefer a vLLM server (`vllm_llm_client.py`,
  pointed at `$ITERRET_LLM_BASE_URL` / the server this repo already
  launches) when the `openai` package is importable, else fall back to
  `QwenClient.chat()`. No native JSON mode is assumed either way — every call
  prompts for JSON and parses with one retry, and every extraction function
  has a deterministic fallback so a flaky/offline/weak LLM degrades
  hypergraph *quality*, never breaks indexing outright.

## Running it

```bash
# Real small model (downloads ~1GB on first run), real segmentation + LLM calls
bash run_hypergraph_demo.sh

# No GPU / no download -- exercises every fallback path only
bash run_hypergraph_demo.sh --dry-run

# Match this repo's actual model
bash run_hypergraph_demo.sh --model Qwen/Qwen3-4B-Instruct-2507 --gamma 1.5

# This repo's canonical conda env is "workmem" (see env.sh); override if
# yours is named differently
CONDA_ENV_NAME=C-AIMMS bash run_hypergraph_demo.sh
```

It builds a small synthetic two-topic conversation (a Japan trip + a dog),
indexes it, prints the resulting topic/episode/fact structure, and runs two
differently-themed queries so you can see the coarse-to-fine traversal
actually discriminate between them. See `run_hypergraph_demo.py --help` for
all flags.

## Building a hypergraph from real LoCoMo data

`build_locomo_hypergraph.py` / `.sh` load a real conversation sample from
`data/locomo10.json`, convert its turns into `Page`s (`locomo_loader.py` --
LoCoMo turns are single-speaker and don't strictly alternate, so consecutive
turns are paired two at a time into one `Page` each, with the real speaker
name prefixed onto the text), build the hypergraph with `HypergraphMemory`,
and write the result to `hypergraph_output/<sample_id>.json` for the
visualizer.

```bash
# List the 10 samples (id, speakers, size) to help pick one
bash build_locomo_hypergraph.sh --list

# Build from sample 0 (Caroline/Melanie), its first 3 sessions (default)
bash build_locomo_hypergraph.sh --sample-index 0

# A whole sample has 19-32 sessions / 369-689 turns -- ALL of them means
# many more LLM calls and a much longer run
bash build_locomo_hypergraph.sh --sample-index 0 --max-sessions 0

# Also run a retrieval query against the freshly-built hypergraph
bash build_locomo_hypergraph.sh --sample-index 0 --query "What happened at the LGBTQ support group?"
```

The one already checked into the repo
(`hypergraph_output/conv-26.json`) is Caroline/Melanie's first 3 sessions
(29 pages) under the same small `Qwen/Qwen2.5-0.5B-Instruct` model used for
testing, `gamma=0.5` -- built in 27s into 2 topics, 7 episodes, 11 facts.
That's also the dataset baked into `hypergraph_visualizer.html` by default.

## Visualizing it

`hypergraph_visualizer.html` renders the topic/episode/fact structure as
three horizontal strata (topics → episodes → facts, like a core sample),
with hyperedge connections drawn between them. Click any card to trace its
full lineage -- ancestors and descendants light up with a connecting line,
everything else dims -- and an inspector panel shows the full
title/summary/content plus the underlying source pages for whatever's
selected. A "Load JSON…" control swaps in any other
`build_locomo_hypergraph.py` output (or a hand-built one matching the same
shape) without needing to touch the file.

Open it by publishing/hosting the HTML file, or directly as a local file in
a browser (`file://…/hypergraph_visualizer.html` -- everything is
self-contained, no server needed beyond what a file load requires for the
Google Fonts request). To point it at a different sample, rebuild with
`build_locomo_hypergraph.sh`, then use "Load JSON…" in the page and pick the
new `hypergraph_output/<sample_id>.json`.

**JSON schema** it expects (matches `serialize_hypergraph()` in
`build_locomo_hypergraph.py`): `{sample_id, speakers, sessions_included,
num_pages, topics: [{id, title, summary, episode_ids}], episodes: [{id,
subject, summary, page_ids, topic_id}], facts: [{id, content, confidence,
episode_id, keywords, temporal, spatial, role}], episode_hyperedges: [{id,
topic_id, relation, weights, coherence_score}], fact_hyperedges: [{id,
episode_id, relation, weights}], pages: [{id, text, episode_id}]}`.

## Verified

- Full pytest-style unit coverage doesn't exist yet (none did before this
  change either) — verification so far is targeted smoke tests, run both in
  dry-run mode and against a real, in-process-loaded model
  (`Qwen/Qwen2.5-0.5B-Instruct`, on a shared GPU box):
  segmenter tensor-shape correctness, happy-path JSON extraction/streaming
  topic formation (multi-topic, hyperedge rebuild-and-cleanup), the
  coarse-to-fine retrieval hard filter, full `MTEM` dispatch integration, and
  `best_structure_evaluator`'s build/retrieve/layout-string paths.
- **Bug found and fixed during real-model testing**: newer `transformers`
  versions (5.x) replaced `DynamicCache.to_legacy_cache()` with a
  `.layers[i].keys/.values` API. The original `_to_legacy_kv` helper only
  checked for `to_legacy_cache`, so on transformers 5.x it silently left the
  Key-states as an unsubscriptable `Cache` object; the resulting `TypeError`
  was then swallowed by `PageEpisodeSegmenter.segment()`'s own defensive
  `except Exception` fallback, making broken similarity-refinement look
  identical to "this session just isn't surprising enough to split." Fixed
  with `_extract_layer_keys()`, a version-tolerant accessor (`.layers`,
  `.to_legacy_cache()`, `.key_cache`, or a plain legacy tuple), and the
  fallback now logs a warning (`exc_info=True`) instead of failing silently.
- `build_locomo_hypergraph.py` was run end-to-end against real LoCoMo data
  (sample `conv-26`, sessions 1-3, 29 pages) and produced a genuine
  multi-topic hypergraph (2 topics, 7 episodes, 11 facts) in 27s.
  `hypergraph_visualizer.html` was checked by running its actual inline
  script (not a reimplementation) under jsdom against that real output:
  correct card counts per band, and clicking a topic card then a fact card
  then re-clicking to clear all produced the expected selection state and
  inspector content with no runtime errors. jsdom has no real layout engine
  (`getBoundingClientRect` always returns zeros), so the connector-line
  *positioning* math and overall visual layout were not verified in a real
  browser — no way to do that in this environment (the Claude-in-Chrome
  extension wasn't available). Worth a quick look in an actual browser
  before trusting the line placement pixel-for-pixel.

### Known caveats

- **`gamma` needs recalibrating for small sessions.** The default `gamma =
  1.5` (matching HyperMem's own `CAIMMSBoundaryEmitter`) produced *zero*
  episode splits on a 7-page demo session under a 0.5B model — the
  mean+std threshold check only has 7 samples to work with, and a small
  model's surprisal signal is less discriminative than the repo's intended
  Qwen3-4B. A gamma sweep (1.5 → 0.0) confirmed the underlying threshold
  math itself is working and sensitive to the setting — it split into 2, 3,
  then 4 episodes as gamma dropped — so this is a tuning question, not a
  bug. Re-tune against real session lengths and the actual Qwen3-4B model
  before trusting the 1.5 default in production.
- **Never tested against the repo's actual Qwen3-4B model or its own vLLM
  server.** Testing used a much smaller stand-in model
  (`Qwen/Qwen2.5-0.5B-Instruct`) loaded in-process, specifically to avoid
  competing for GPU memory/compute with another user's unrelated job already
  saturating both GPUs on the shared box this was tested on, and to avoid
  needing the credentials for the vLLM server already running there for a
  different pipeline. Fact/topic quality was visibly rough (a small model's
  weaker instruction-following — e.g. it sometimes extracted a question as
  a "fact" instead of a declarative claim) — expected from the smaller
  model, not a sign of a pipeline defect, but worth re-validating against
  the real model before drawing conclusions about hypergraph *quality*.
