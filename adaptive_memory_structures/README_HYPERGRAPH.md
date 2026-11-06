# HypergraphMemory

Replaces `GraphMemory` (a plain pairwise page-similarity graph) with a real
3-layer hypergraph memory modeled on the **HyperMem** paper
(`Hypergraph Memory for Long-Term Conversations`), adapted to this repo's
per-session (`EpisodicSession`) architecture. This is a **swap, not an
addition** — `GraphMemory` no longer exists; every place that used to
dispatch on `structure_type == "graph"` now dispatches on `"hypergraph"`.

It is also wired into the main LoCoMo delta-Mem eval as an alternative
retriever to IterRet (`WORKMEM_RETRIEVER=hypermem`) -- see
**[LoCoMo eval integration](#locomo-eval-integration-workmem_retrieverhypermem)**
for how, the retrieval redesign that made it competitive, and results.

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
| `locomo_hypermem.py` | Whole-conversation hypergraph build + `HyperMemRetriever` (soft hybrid / strict), the eval's `WORKMEM_RETRIEVER=hypermem` backend (see **LoCoMo eval integration**) |
| `build_locomo_hypermem_cache.py` | Batch-builds the eval's hypergraph cache (`<outputs>/hypermem_cache/sample_<i>.json`) against a running vLLM |

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
  (This still describes `HypergraphMemory`; the LoCoMo eval's
  `HyperMemRetriever` does add BM25 fusion -- see **LoCoMo eval integration**.)
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
- **Fact/topic extraction has now been run with the repo's Qwen3-4B via
  vLLM** (2026-10-09; previously only `Qwen/Qwen2.5-0.5B-Instruct`
  in-process). The 0.5B model's rough output (questions extracted as
  "facts") is gone: facts are clean declarative claims ("Melanie painted a
  lake sunrise last year.") and topics are sensible ("Caroline's LGBTQ+
  Adoption Journey"). Surprise segmentation still uses the 0.5B model in
  every build so far.
- **Mean-pooled causal-LM embeddings retrieve poorly.** `QwenClient.embed()`
  (mean of the last hidden state) is what `HypergraphMemory.retrieve` scores
  with; on conv-26 it missed "What does Melanie like to paint?" even though
  the fact was in the graph. The eval integration below scores with MiniLM
  (IterRet's sentence encoder) instead.
- **Topics come out very broad on whole conversations.** Streaming topic
  matching yields 4-11 topics per LoCoMo conversation, one of which often
  holds a third to half of all episodes (conv 0: 7 topics, one with 26 of
  54 episodes). A hard top-k topic filter over that is the main reason the
  strict traversal underperforms (below).

## LoCoMo eval integration (`WORKMEM_RETRIEVER=hypermem`)

The LoCoMo eval (`delta-Mem/deltamem/workmem/eval_locomo_iterret_mock.py`)
can take its evidence from HyperMem instead of IterRet. **Only the evidence
source changes**: the relevance filter, the delta-mem OSAM write, the answer
prompt, the generation settings and the scoring are identical, so the two
retrievers are directly comparable on the same questions.

```bash
source env.sh
# hypergraphs ship pre-built (cached_graphs/hypermem/, ~6h of vLLM calls to rebuild)
mkdir -p "$CAIMMS_OUTPUT_DIR/hypermem_cache" && cp -n cached_graphs/hypermem/*.json "$CAIMMS_OUTPUT_DIR/hypermem_cache/"

# conv 0 (152 Q), same set as run_pipeline.sh --smoke
WORKMEM_RETRIEVER=hypermem WORKMEM_MAX_SAMPLES=1 \
  WORKMEM_OUTPUT_FILE="$CAIMMS_OUTPUT_DIR/workmem_hypermem_n1.jsonl" bash scripts/run_pipeline.sh
# or: bash scripts/run_everything.sh --smoke --retriever hypermem

# With every hypergraph cached the eval never calls vLLM, so it can also run
# without run_pipeline.sh, on one GPU or even CPU (WORKMEM_DEVICE=cpu, ~70s/question):
cd delta-Mem && WORKMEM_RETRIEVER=hypermem WORKMEM_SAMPLES=0 WORKMEM_DEVICE=cpu \
  WORKMEM_OUTPUT_FILE="$CAIMMS_OUTPUT_DIR/c0_hypermem.jsonl" python3 -m deltamem.workmem.eval_locomo_iterret_mock
```

### Building the conversation hypergraph (`locomo_hypermem.py`)

`HypergraphMemory.build_index` works on one session; a LoCoMo question can
need any of a conversation's 19-32 sessions, so the eval builds one
hypergraph per **conversation**:

- each session's turns are paired into `Page`s (as `locomo_loader.py` does),
  with relative dates resolved against the session timestamp by IterRet's
  `time_resolution.resolve_relative_time` -- the same transformation the
  IterRet graph applies -- and the session date prefixed: `[8 May, 2023] Caroline: ...`
- **segmentation is per session.** `PageEpisodeSegmenter` does one forward
  pass over its whole input; a full conversation (~30k tokens x ~150k-vocab
  logits, ~9GB) does not fit
- episode summaries, facts and fact hyperedges come from the same
  `hypergraph_extraction` functions as `build_index`, via Qwen3-4B on vLLM
- **topics are formed across the whole conversation**: all episodes are
  streamed through `build_topics_for_session` in temporal order, so a topic
  can span sessions -- HyperMem's own setting

Defaults: segmenter `Qwen/Qwen2.5-0.5B-Instruct` (`HYPERMEM_SEG_MODEL`,
`HYPERMEM_SEG_DEVICE`), `gamma=0.5` (`HYPERMEM_GAMMA`), `min_block_size=1`.
Cost: 20-67 min per conversation (~500 vLLM calls). Built for LoCoMo:

| conv | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| topics | 7 | 4 | 9 | 11 | 10 | 5 | 11 | 7 | 8 | 11 |
| episodes | 54 | 42 | 79 | 80 | 96 | 96 | 89 | 81 | 71 | 74 |
| facts | 242 | 184 | 321 | 332 | 359 | 378 | 362 | 337 | 286 | 315 |

Conv 0 was segmented on GPU (bf16), convs 1-9 on CPU (fp32) after the GPUs
became unavailable; boundaries can differ slightly between the two.

The eval looks for `<outputs>/hypermem_cache/sample_<i>.json` and builds any
missing one in-process (segmenter on the eval GPU, extraction through the
vLLM `run_pipeline.sh` starts). To pre-build all of them against a running
vLLM: `python3 adaptive_memory_structures/build_locomo_hypermem_cache.py --samples 0-9`.

Build speed (2026-10-26). The build is three stages: segmentation (in-process,
per session), per-episode extraction (summary, facts, fact roles: 3 calls per
episode) and topic streaming (a match call per episode plus a topic update
and hyperedge re-weighting on every match, strictly serial). By call count
the topic stage is most of the ~500 calls (conv 1: 42 episodes = 126
extraction calls). Two build knobs:

| var / flag | default | effect |
|---|---|---|
| `HYPERMEM_BUILD_WORKERS` / `--workers` | 8 | episodes extracted concurrently against vLLM; output order and content are unchanged (each episode's own calls stay in order) |
| `HYPERMEM_FACT_SCOPE` | `episode` | `topic` = facts from a topic's episodes read together, in place of per-episode facts; `both` = added to them. Not adopted, see "Topic facts and query patterns" |
| `HYPERMEM_TOPIC_STRICT` | 0 | 1 = stricter topic matching: separate people's similar subjects stay separate topics. Not adopted |
| `HYPERMEM_BUILD_TOPICS` / `--topics` | `llm` | `none` skips the LLM topic stage; the retriever then clusters episodes into ~4-episode topics itself (`HYPERMEM_RETOPIC` defaults to 4 for a topic-less graph) |

`--topics none` costs no recall (see "Topic layer" below); it is not the
default yet because end-to-end F1 on topic-less graphs has not been run.
Write topic-less builds to their own `--cache-dir`: the cache key is only
the sample index, so they would otherwise replace the LLM-topic graphs.

### Retrieval (`HyperMemRetriever`)

**strict** (`HYPERMEM_MODE=strict`) is `HypergraphMemory.retrieve`'s
coarse-to-fine traversal at conversation scale: top-3 topics -> hard-filtered
top-8 episodes -> hard-filtered top-20 facts -> those facts' episodes' pages,
up to 40 turns; dense cosine only. Its failure mode on LoCoMo: when the
evidence sits under a topic outside the top 3 -- frequent, given one topic
can hold half the conversation -- it is unreachable.

**soft** (default) keeps the hypergraph but turns the hierarchy from a filter
into evidence aggregation:

1. every topic, episode, fact and turn is ranked against the question by
   **hybrid dense + BM25 reciprocal-rank fusion** (MiniLM + a Porter-stemmed
   BM25). HyperMem's own retrieval is hybrid; the port had dropped the
   lexical half, which matters for names and dates
2. hyperedge embedding propagation (alpha 0.5) is still applied on the dense side
3. an episode's score fuses its own rank, its topic's rank (through the
   episode hyperedge), its best fact's rank (through the fact hyperedge) and
   its best turn's rank -- nothing is excluded
4. turns are ranked individually: own hybrid rank fused with their episode's
   rank
5. a second pass (**PRF**) repeats 1-4 with the question expanded by the
   top facts of the first pass, and the two passes are rank-fused
6. a cross-encoder **reranks** the top 120 candidate turns, fused with the
   hypergraph order; the top 40 are the evidence
7. the top 10 extracted **facts**, dated, are prepended to those turns

Steps 5-7 are the default since 2026-10-12. "Soft" or "plain soft" in the
tables below means steps 1-4 only
(`HYPERMEM_PRF=0 HYPERMEM_RERANK=0 HYPERMEM_FACTS=0`), the earlier default.

Knobs (env, read when the retriever is built):

| var | default | effect |
|---|---|---|
| `HYPERMEM_MODE` | `soft` | `strict` = original traversal |
| `HYPERMEM_MAX_TURNS` | 40 | evidence budget, in turns (IterRet averages ~36) |
| `HYPERMEM_FACTS` | 10 (strict: 0) | prepend the top-N extracted facts, dated, as evidence |
| `HYPERMEM_RRF_K` | 20 | RRF constant (tuned; 60 is the textbook value) |
| `HYPERMEM_W_EPISODE` / `_TOPIC` / `_FACT` / `_ETURN` | 1 / 0.5 / 1 / 1 | episode-score signal weights |
| `HYPERMEM_W_TURN` / `_PRIOR` | 1 / 1 | turn score: own rank vs episode prior |
| `HYPERMEM_PER_EPISODE_CAP` | 0 | max turns from one episode (0 = no cap) |
| `HYPERMEM_PAGE_EXPAND` | 0 | 1 = take a selected turn's page partner too |
| `HYPERMEM_TOPIC_TOP_K` / `_EPISODE_TOP_K` / `_FACT_TOP_K` | 3 / 8 / 20 | strict mode only |
| `HYPERMEM_PRF` | 1 | second pass, question expanded with the top round-1 facts (`_PRF_FACTS` 5, `_PRF_BETA` 0.5, `HYPERMEM_W_PRF` 1) |
| `HYPERMEM_LINK` | 0 | 1 = entity hyperedges: facts sharing rare keywords/names with the top facts boost their episodes (`_LINK_SEEDS` 5, `HYPERMEM_W_LINK` 1) |
| `HYPERMEM_SPEAKER_BOOST` | 1 | w > 0 = when a question names exactly one speaker, their turns get +w/k in each scoring pass; 1 is the tested value (see "Multi-hop retrieval") |
| `HYPERMEM_RERANK` | 1 | cross-encoder rerank of the top candidates (`_RERANK_POOL` 120, `_RERANK_MODEL` `Qwen/Qwen3-Reranker-4B`, `_RERANK_DEVICE` last visible GPU, `HYPERMEM_W_CE` 1). The retriever raises if the model cannot be loaded (`scripts/download_assets.sh` fetches it) |
| `HYPERMEM_LAYER_RERANK` | 1 | also rerank each pass's top `HYPERMEM_LAYER_POOL` (100) topics, episodes and facts; off when `HYPERMEM_RERANK=0` |
| `HYPERMEM_EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B` | retrieval encoder (`HYPERMEM_EMBED_DEVICE` last visible GPU); empty = the caller's encoder (MiniLM in the evals) |
| `HYPERMEM_FACT_PATTERNS` | 0 | 1 = also index each fact's `query_patterns` for dense and BM25 search (graphs built before 2026-11-05 did not save them; see "Topic facts and query patterns") |
| `HYPERMEM_CE_CUTOFF` | unset | with rerank: drop turns whose cross-encoder logit is below this, keeping at least `HYPERMEM_MIN_TURNS` (8) |

### Tuning (held out: conv 0 and convs 4-9)

Tuned **only on convs 1-3**, by gold-evidence recall: the share of
questions whose LoCoMo gold evidence turns are *all* in the 40 retrieved
turns (`scripts/hypermem_recall.py`, CPU, minutes). Every single-knob change
moved recall by <= 0.035 except fusion sharpness; the top combinations were
within ~0.01 of each other (noise at n=432), so the simplest one was kept:
the defaults above with `RRF_K` 60 -> 20.

| all gold turns retrieved (40 turns) | convs 1-3 (tuning, n=432) | conv 0 (held out, n=150) | convs 4-9 (held out, n=954) |
|---|---|---|---|
| strict | 0.282 | 0.433 | 0.358 |
| soft, k=60 | 0.736 | 0.753 | 0.736 |
| **soft, k=20** | **0.773** | **0.767** | **0.777** |
| flat MiniLM over all turns | 0.632 | 0.627 | 0.609 |
| flat MiniLM + BM25 over all turns | 0.697 | 0.687 | 0.695 |

These numbers were re-measured after a fix to the harness's gold matching.
The cached graphs store date-resolved turn text ("tomorrow (20 June,
2023)"), so a gold turn with a relative date in its first 50 characters
could never match and was counted as a miss whatever was retrieved (7-27
questions per conversation). Every row was affected alike: the earlier
table read 0.05-0.09 lower for soft (0.718 / 0.678 / 0.715) with the same
ordering, and the tuning decision does not change.

Soft retrieval beats the same hybrid search without the hypergraph by
~0.08 on held-out conversations, so the structure contributes beyond
the lexical channel. Multi-hop is the weak category (~0.27-0.29 all-gold),
since its questions need evidence from several sessions at once.

### Multi-hop and precision steps (no LLM calls)

Three additions on top of plain soft. PRF and RERANK (pool 120) are on by
default; LINK and the cutoff are not:

- **PRF** (`HYPERMEM_PRF=1`): a second scoring pass whose query is the
  question expanded with the top round-1 facts (dense: Rocchio; lexical:
  their text + keywords), rank-fused with the first pass. Reaches the
  "second hop" that resembles what the first hop found, not the question
- **LINK** (`HYPERMEM_LINK=1`): entity hyperedges over facts (stemmed
  keywords + capitalised names, IDF-weighted); facts sharing an entity with
  the top facts boost their episodes
- **RERANK** (`HYPERMEM_RERANK=1`): a cross-encoder reranks the top
  candidate turns, fused with the hypergraph order; `HYPERMEM_CE_CUTOFF`
  optionally cuts the list short instead of always filling 40 turns

Same protocol as above: configs compared on convs 1-3, then checked on
the held-out conversations. Cells: all gold turns retrieved, multi-hop
only in brackets.

| config | convs 1-3 (tuning, n=432) | conv 0 (held out, n=150) | convs 4-9 (held out, n=954) | turns used |
|---|---|---|---|---|
| plain soft | 0.773 [0.29] | 0.767 [0.28] | 0.777 [0.27] | 40 |
| PRF | 0.796 [0.35] | 0.813 [0.44] | 0.796 [0.33] | 40 |
| LINK | 0.773 [0.30] | 0.800 [0.34] | 0.791 [0.30] | 40 |
| PRF + LINK | 0.799 [0.37] | 0.800 [0.38] | 0.804 [0.37] | 40 |
| RERANK | 0.808 [0.38] | 0.780 [0.31] | 0.798 [0.34] | 40 |
| PRF + RERANK | 0.826 [0.46] | 0.813 [0.47] | 0.820 [0.43] | 40 |
| PRF + LINK + RERANK | 0.817 [0.44] | 0.813 [0.47] | 0.816 [0.42] | 40 |
| **PRF + RERANK, pool 120 (default)** | **0.836 [0.49]** | **0.820 [0.47]** | **0.826 [0.43]** | 40 |
| PRF + LINK + RERANK, cutoff -3 | 0.764 [0.33] | 0.747 [0.38] | 0.744 [0.26] | ~20 |
| PRF + LINK + RERANK, cutoff 0 | 0.734 [0.30] | 0.700 [0.25] | 0.716 [0.20] | ~13 |

- PRF and RERANK each add ~0.02 on convs 4-9 and stack: PRF + RERANK with
  a 120-turn pool is +0.05 over soft on all three splits, and multi-hop
  all-gold recall goes ~0.28 -> 0.43-0.49
- LINK alone is +0.01 to +0.03 on held-out conversations and 0 on the
  tuning ones, and adds nothing once PRF + RERANK are on
- the cutoff halves the evidence (~3x the precision) but loses recall at
  every value tried (0, -3; -6 and -8 before the matching fix); not
  evaluated end-to-end
- rerank pool 80 / 120 / 160 and `W_CE` 1 / 2 were within ~0.01 of each
  other on the tuning convs (measured before the matching fix)
- what is left is mostly multi-hop: single-hop and temporal are at
  0.9+ all-gold recall on most conversations

### End-to-end results (conv 0, 152 Q, token-F1)

| category | n | IterRet (original pipeline) | HyperMem strict | **HyperMem soft** |
|---|---|---|---|---|
| **Overall** | 152 | 0.4680 | 0.3052 | **0.4555** |
| Multi-hop | 32 | 0.3946 | 0.2534 | 0.3231 |
| Temporal | 37 | 0.6901 | 0.3981 | 0.6638 |
| Open-domain | 13 | 0.1655 | 0.1567 | 0.1421 |
| Single-hop | 70 | 0.4403 | 0.3073 | 0.4641 |

- soft vs strict: **+0.150** per question, 95% bootstrap CI [+0.09, +0.21];
  better on 68, worse on 19
- soft vs IterRet: -0.013, CI [-0.06, +0.03] -- statistically tied; better
  on 35, worse on 32. Ahead on single-hop, behind on multi-hop
- all three: 152/152 answered, 0 skipped, delta-mem live on every row;
  HyperMem used 40 evidence turns/question vs IterRet's ~36
- IterRet and strict ran on GPU (RTX 4090, bf16). **Soft ran on CPU**
  (`WORKMEM_DEVICE=cpu`) after a host driver update left CUDA unusable:
  same weights and greedy decoding, but CPU/GPU numerics can flip a few
  answers. Re-run on GPU on 2026-10-12: 0.4574 (5 of 152 scores differ)
- these runs predate PRF, rerank and facts-as-evidence (`HYPERMEM_FACTS`
  was 0); the current default is evaluated two sections below
- raw rows: `cached_results/locomo_c0_{iterret,hypermem_strict,hypermem_soft}.jsonl`

### End-to-end, all 10 conversations: soft vs PRF + RERANK (1540 Q, token-F1)

`HYPERMEM_PRF=1 HYPERMEM_RERANK=1 HYPERMEM_RERANK_POOL=120` (no facts)
against plain soft, both on GPU (RTX 4090, bf16), 40 evidence turns, 0 skipped.
Convs 1-3 are the ones the retrieval configs were compared on, so the
held-out column is the one to trust. Diff = paired mean difference with a
95% bootstrap CI.

| category | n | soft | PRF + RERANK | diff, all 10 | diff, held out (convs 0, 4-9; n=1108) |
|---|---|---|---|---|---|
| **Overall** | 1540 | 0.4724 | 0.4836 | +0.011 [+0.000, +0.022] | +0.004 [-0.009, +0.017] |
| Multi-hop | 282 | 0.3527 | 0.4118 | **+0.059 [+0.029, +0.090]** | **+0.040 [+0.006, +0.075]** |
| Temporal | 321 | 0.5798 | 0.5974 | +0.018 [-0.004, +0.040] | +0.022 [-0.007, +0.052] |
| Open-domain | 96 | 0.0899 | 0.1067 | +0.017 [-0.006, +0.045] | +0.010 [-0.010, +0.029] |
| Single-hop | 841 | 0.5152 | 0.5073 | -0.008 [-0.022, +0.007] | -0.017 [-0.034, +0.001] |

- multi-hop improves, as the recall numbers predicted; single-hop, more
  than half the questions, does not and may get slightly worse, so overall
  is a tie on held-out conversations. On their own the two steps were not
  worth making the default; with the facts added (next section) they are
- conv 0 alone (152 Q): soft 0.4574, PRF + RERANK 0.4768 (IterRet 0.4680);
  +0.019 [-0.010, +0.051] vs soft, +0.009 [-0.037, +0.054] vs IterRet
- per-conversation diff ranges from -0.020 to +0.043 (6 up, 1 flat, 3 down)
- recall is no longer what limits single-hop and temporal: on conv 0, all
  gold evidence is retrieved for 67/70 single-hop and 36/37 temporal
  questions, yet F1 on those is 0.46 and 0.71. The misses there are
  answer-side (a relative phrase copied instead of the resolved date,
  "not mentioned" with the evidence present, long answers to two-word
  golds)
- HyperMem answers in ~2 s/question on GPU with the graphs cached (no
  vLLM needed), so the full 1540 Q takes under an hour
- raw rows are not in git (16 MB): `<outputs>/c0_hm_{soft,prf_rr120}_gpu.jsonl`
  and `<outputs>/c1-9_hm_{soft,prf_rr120}_gpu.jsonl`

### End-to-end: facts as evidence, and the current default (token-F1)

`HYPERMEM_FACTS=10` prepends the 10 best-ranked extracted facts, each with
its episode's date, to the 40 turns (50 evidence items). First tried on
convs 1-3 on top of PRF + RERANK, then checked on the held-out
conversations. All runs on GPU, 0 skipped.

Convs 1-3 (432 Q), variants of PRF + RERANK; diff vs the 40-turn row:

| evidence | overall | multi-hop | temporal | single-hop | diff [95% CI] |
|---|---|---|---|---|---|
| 40 turns | 0.498 | 0.450 | 0.639 | 0.494 | -- |
| 20 turns | 0.475 | 0.396 | 0.629 | 0.473 | -0.023 [-0.043, -0.003] |
| 12 turns | 0.466 | 0.350 | 0.611 | 0.480 | -0.031 [-0.055, -0.008] |
| cutoff -3 (~22 turns) | 0.484 | 0.417 | 0.638 | 0.480 | -0.013 [-0.034, +0.008] |
| **40 turns + 10 facts** | **0.540** | 0.439 | **0.744** | **0.533** | **+0.043 [+0.018, +0.067]** |

Held-out convs 0, 4-9 (1108 Q):

| config | overall | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|---|
| plain soft | 0.4746 | 0.3570 | 0.5587 | 0.1090 | 0.5294 |
| PRF + RERANK | 0.4781 | 0.3971 | 0.5806 | 0.1187 | 0.5127 |
| plain soft + 10 facts | 0.5202 | 0.3764 | 0.6577 | 0.1621 | 0.5625 |
| **PRF + RERANK + 10 facts (default)** | **0.5287** | **0.4042** | **0.6652** | 0.1440 | **0.5684** |

- facts over plain soft: +0.046 [+0.029, +0.062]; temporal +0.099,
  open-domain +0.053, single-hop +0.033, multi-hop +0.019 (n.s.)
- **default over plain soft: +0.054 [+0.037, +0.071]**, positive on all
  seven held-out conversations (+0.035 to +0.100)
- PRF + RERANK on top of the facts: +0.009 [-0.001, +0.019] overall,
  multi-hop +0.028 [+0.001, +0.055]; the single-hop dip they showed
  without facts is gone
- fewer turns is worse, not better: the 4B answerer is not being drowned
  by 40 turns
- all 10 conversations (1540 Q; convs 1-3 from the tuning run): plain soft
  0.4724 -> default **0.5320**, +0.060 [+0.045, +0.074]; multi-hop 0.4139,
  temporal 0.6880, open-domain 0.1284, single-hop 0.5581
- conv 0 (152 Q): default 0.4921 vs IterRet 0.4680, +0.024
  [-0.020, +0.069] -- still a statistical tie; temporal 0.833 vs 0.690,
  multi-hop 0.363 vs 0.395
- only 10 facts was tried (not 5 or 20)
- raw rows, not in git: `<outputs>/ho_{soft,new}_facts10.jsonl` (held out)
  and `<outputs>/t13_new_{t20,t12,cut-3,facts10}.jsonl` (convs 1-3)

Not yet done: a full 10-conversation IterRet run to compare against (~30h
on the shared box); other fact counts; keeping the retriever's order
instead of the eval's MiniLM re-sort of the evidence (`evidence_filter.py`).

### Topic layer: LLM topics vs embedding clusters vs none (recall, 2026-10-25)

The LLM-streamed topics are very uneven (one topic holds 23-56% of a
conversation's episodes). `HYPERMEM_RETOPIC=N` swaps them for Ward clusters
of ~N episodes each; `HYPERMEM_W_TOPIC=0` drops the topic signal from the
episode score (hyperedge propagation still runs). Same harness and protocol
as above (`scripts/hypermem_recall.py`, cross-encoder on GPU). Cells: all
gold turns retrieved, multi-hop only in brackets.

| config | convs 1-3 (n=432) | convs 0, 4-9 (held out, n=1104) |
|---|---|---|
| default (LLM topics, `W_TOPIC` 0.5) | 0.836 [0.49] | 0.825 [0.44] |
| no topic signal (`W_TOPIC=0`) | 0.838 [0.51] | 0.828 [0.45] |
| LLM topics, `W_TOPIC=1` | 0.838 [0.49] | -- |
| clusters of ~3 / 4 / 6 episodes | 0.843 / 0.845 / 0.840 | -- / 0.829 [0.44] / -- |
| clusters of ~8 / 12 / 16 episodes | 0.838 / 0.833 / 0.836 | 0.826 [0.44] / -- / -- |
| clusters of ~4 / 8, `W_TOPIC=1` | 0.840 / 0.838 | -- |
| plain soft / no topic / clusters ~4 | 0.773 / 0.769 / 0.769 | 0.775 / 0.774 / 0.777 |
| graph built with no topics (cached graphs with topics stripped; retriever clusters ~4) | 0.845 [0.52] | 0.829 [0.44] |
| graph with no topics, no clustering either (`HYPERMEM_RETOPIC=0`) | 0.838 [0.49] | 0.829 [0.45] |

- every variant is within +/-0.01 of the default on both splits (about 4
  questions on the tuning set): **the topic layer contributes nothing
  measurable to recall in soft mode**, whether the topics come from the LLM,
  from balanced clustering, or are left out of the score entirely
- so better topics are not a lever for recall. What does move it is
  episode/fact/turn fusion, PRF and the rerank (sections above)
- the LLM topic stage is the serial, most expensive part of the build
  (a model call per episode against the existing topics, plus updates and
  hyperedge re-weighting on every match). Dropping it, or replacing it with
  the free clustering, loses no recall: the last two rows are the
  topic-less graphs `HYPERMEM_BUILD_TOPICS=none` produces (with topics
  stripped from the cached ones, everything else identical). Not yet checked
  end-to-end on F1
- raw output: none kept beyond this table (re-run in ~5 min with the command
  below)

```bash
python3 scripts/hypermem_recall.py 1,2,3 default: notopic:HYPERMEM_W_TOPIC=0 rt4:HYPERMEM_RETOPIC=4
```

## Stemming fix, LLM judge and chain-of-thought (2026-11-03)

**Correction: every F1 and recall number above was measured without
stemming**, even where it says Porter-stemmed. `nltk` was missing from the env
and the requirements, and both `locomo_protocol.py` and `_tokenize` silently
fell back to raw tokens. `nltk` is now pinned and required, so a missing install
fails instead of changing the metric. The tables above are left as measured.

Rescoring saved predictions with the stemmer: all-10-conversation default
0.5320 -> **0.5460**; conv 0 IterRet 0.4680 -> 0.4873. Recall with stemmed BM25,
all gold turns, multi-hop in brackets:

| config | convs 1-3 (n=432) | convs 0, 4-9 (held out, n=1104) |
|---|---|---|
| default | 0.836 [0.49] -> 0.845 [0.52] | 0.825 [0.44] -> 0.830 [0.46] |
| plain soft | 0.773 -> 0.792 | 0.775 -> 0.797 |
| flat MiniLM + BM25 | 0.697 -> 0.748 | ~0.695 -> 0.722 |

Convs 1-3 (n=432), stemmed BM25 and scoring, 4B answerer, no prompt
engineering. "Judge" is `iterret/llm_judge.py` run on 4B; oracle is gold turns
only. Diffs are paired with 95% CIs:

| arm | F1 | judge | judge: multi / temp / open / single |
|---|---|---|---|
| current (40 turns + 10 facts) | 0.549 | 0.785 | 0.620 / 0.935 / 0.211 / 0.826 |
| current + CoT | 0.510 | 0.780 | 0.608 / 0.882 / 0.316 / 0.834 |
| oracle | 0.612 | 0.806 | 0.722 / 0.871 / 0.105 / 0.863 |
| oracle + CoT | 0.586 | 0.829 | 0.785 / 0.860 / 0.579 / 0.851 |

- CoT on current evidence: F1 -0.039 [-0.065, -0.012], judge -0.005
  [-0.035, +0.028]. Not adopted
- oracle over current: F1 +0.063 [+0.031, +0.095], judge +0.021 [-0.019, +0.060];
  temporal is worse with oracle evidence, F1 -0.077 [-0.136, -0.023]
- the prompt-engineered t13 run judges 0.817, +0.032 [+0.002, +0.065] over
  current, above oracle with the standard prompt. Its rows predate the stemmed
  BM25, and convs 1-3 are the tuning set
  (`OSAM_PROMPT_ENGINEERING=1` is partly LoCoMo-specific, including "every
  question here has an answer", so this is not a general gain)
- all 10 conversations, saved default run (unstemmed BM25): judge 0.783
- the 4B judge is noisy in both directions when read by hand (accepts "hiked,
  swam" for "camping", rejects "I joined a gym last week" for "joined a gym"),
  so it is not comparable to the paper's GPT-4o-mini judge or 92.73
- raw rows and scripts, not in git: `<outputs>/cot_test_20261103/`

General prompts, held-out convs 0, 4-9 (n=1108), stemmed BM25 and scoring.
"General" is no first person + yes/no + temporal instructions. "+ answer first"
also asks for the answer as a short phrase before any elaboration. Neither has
the "every question has an answer" rule or the worked examples. Prompts live in
a scratch patch (`arm.py`, saved with the raw rows), not in `osam_workmem.py`:

| prompt | F1 | judge | words | F1 diff [95% CI] | judge diff [95% CI] |
|---|---|---|---|---|---|
| standard | 0.541 | 0.781 | 14.3 | -- | -- |
| general | 0.534 | 0.785 | 10.7 | -0.007 [-0.019, +0.005] | +0.005 [-0.013, +0.022] |
| general + answer first | 0.557 | 0.768 | 7.2 | +0.015 [+0.002, +0.028] | -0.013 [-0.032, +0.007] |

- neither improves judged accuracy; the F1 gain comes with answers half as long
  and is confined to single-hop (+0.039 [+0.021, +0.057]; multi-hop -0.023
  [-0.048, +0.001]), so it looks like format fit rather than better answers
- not adopted

### Multi-hop retrieval (2026-11-04)

Missed multi-hop gold turns sit deep (median dense rank 211 of 369 turns on convs
1-3), and 94% are spoken by a person the question names. Two patches were tested;
only the boost is in `locomo_hypermem.py` (`HYPERMEM_SPEAKER_BOOST=1`, on by default since the section below):

- **speaker boost**: when a question names exactly one speaker, that speaker's
  turns get +1/k in each scoring pass. No model call
- **rewritten queries**: 4B writes 4 search queries from the question text alone;
  each gets its own scoring pass, rank-fused at weight 1. One model call per
  question; the 38 of 1529 that failed use the original question

Held-out convs 0, 4-9, standard prompt, stemmed BM25 and scoring. Recall is all
gold turns, n=1104; F1 and judge (4B) are n=1108, diffs paired against default:

| config | recall [multi] | F1 | judge | F1 diff [95% CI] | judge diff [95% CI] |
|---|---|---|---|---|---|
| default | 0.830 [0.46] | 0.541 | 0.781 | -- | -- |
| speaker boost | 0.847 [0.54] | 0.548 | 0.796 | +0.007 [-0.002, +0.015] | +0.015 [+0.002, +0.029] |
| boost + rewritten | 0.858 [0.54] | 0.551 | 0.794 | +0.009 [+0.000, +0.019] | +0.014 [-0.001, +0.028] |

- rewritten queries add nothing on top of the boost: F1 +0.003 [-0.005, +0.011],
  judge -0.002 [-0.014, +0.010]
- multi-hop recall +0.08 gives multi-hop F1 +0.012 to +0.020 and judge +0.020 to
  +0.034, all within noise: most of the recall gain does not reach the answer
- on convs 1-3 the same patches gave multi-hop recall 0.52 -> 0.59 (boost) and
  0.62 (both). Filtering to the speaker, stripping the name, fact-to-turn links
  and a 200-turn rerank pool did no better
- raw rows and scripts, not in git: `<outputs>/multihop_20261104/`

### Paper retrieval models (2026-11-04)

Now the default, with the speaker boost: Qwen3-Embedding-4B and Qwen3-Reranker-4B
in place of MiniLM and ms-marco, and each pass also reranks its top 100 topics,
episodes and facts (RRF, then rerank, as the paper's text describes). The paper
also retrieves top-10 topics/10 episodes/30 facts, answers with GPT-4.1-mini + CoT
and judges with GPT-4o-mini; none of that is copied here. Previous stack:
`HYPERMEM_EMBED_MODEL= HYPERMEM_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 HYPERMEM_LAYER_RERANK=0`.

Held-out convs 0, 4-9, same protocol as the table above:

| config | recall [multi] | F1 | judge | judge, multi-hop |
|---|---|---|---|---|
| MiniLM, ms-marco | 0.830 [0.46] | 0.541 | 0.781 | 0.675 |
| + speaker boost | 0.847 [0.54] | 0.548 | 0.796 | 0.695 |
| paper models | 0.883 [0.63] | 0.559 | 0.798 | 0.719 |
| paper models + boost (default) | 0.887 [0.67] | 0.557 | 0.801 | 0.759 |
| reranker only + boost (`HYPERMEM_EMBED_MODEL=`) | not measured | 0.551 | 0.792 | 0.714 |

- against MiniLM + boost: judge +0.005 [-0.011, +0.021], F1 +0.009 [-0.004, +0.021];
  multi-hop judge +0.064 [+0.010, +0.123], temporal judge -0.022 [-0.057, +0.013]
- recall rises far more than answers do, so the 4B answerer now looks like the limit
- reranker only fits on one GPU with the answer model (~18GB) but does not beat
  MiniLM + boost: judge -0.005 [-0.021, +0.012], temporal judge -0.035 [-0.070, -0.004]

**GPU memory**: the two models peak near 17GB and do not fit beside the answer
model (~10GB) on a 24GB card. For `WORKMEM_RETRIEVER=hypermem`, `scripts/run_pipeline.sh`
and `run_everything.sh` therefore give the eval both cards (answer model on `EVAL_GPU`,
retrieval on `VLLM_GPU`) and start no vLLM, which needs every graph cached and rules
out `WORKMEM_JUDGE=1`; they stop at startup if either condition fails, or on one GPU.
`HYPERMEM_EMBED_MODEL=` restores the original layout, with the previous stack above
for the "+ speaker boost" row. By hand: `CUDA_VISIBLE_DEVICES=0,1 WORKMEM_DEVICE=cuda:0`.

### Formation models (2026-11-04)

The paper names no formation LLM and uses LLM boundary detection rather than
surprise. Graphs for convs 1-3 rebuilt in scratch, then scored with the default
retrieval models above (n=432). "Rebuild" repeats the current recipe, so its gap to
the existing graphs is rebuild noise (fact extraction samples at temperature 0.2):

| graphs | episodes | recall [multi] | F1 | judge | judge diff vs existing [95% CI] |
|---|---|---|---|---|---|
| existing (0.5B surprise) | 201 | 0.887 [0.68] | 0.563 | 0.810 | -- |
| rebuild (0.5B surprise) | 201 | 0.887 [0.70] | 0.551 | 0.796 | -0.014 [-0.039, +0.012] |
| Qwen3-4B surprise | 249 | 0.880 [0.68] | 0.557 | 0.810 | +0.000 [-0.032, +0.032] |
| Qwen3-4B LLM boundaries | 237 | 0.889 [0.71] | 0.567 | 0.803 | -0.007 [-0.037, +0.023] |

No gain beyond rebuild noise, so the segmenter default is unchanged. Convs 1-3 only.

### Topic facts and query patterns (2026-11-05)

The paper extracts facts per topic and indexes the questions each fact answers.
Neither beat the per-episode facts, so both stay off. Episodes are unchanged in
every row; answers and judge use the default retrieval models and Qwen3-4B.

Convs 1-3 (n=432), patterns off -> on:

| facts | F1 | judge | multi-hop judge |
|---|---|---|---|
| existing graph | 0.563 | 0.810 | 0.658 |
| re-extracted per-episode | 0.568 -> 0.566 | 0.782 -> 0.799 | 0.570 -> 0.633 |
| topic, replacing them | 0.508 -> 0.505 | 0.759 -> 0.757 | 0.646 -> 0.671 |
| both | 0.572 -> 0.581 | 0.794 -> 0.810 | 0.608 -> 0.658 |
| strict topics, topic facts | 0.548 -> 0.536 | 0.803 -> 0.785 | 0.747 -> 0.671 |

Replacing loses about two-thirds of the facts and leaves 26-45% of episodes with
none (judge -0.051 [-0.090, -0.012]). Long topics are split between calls, so
facts combine only episodes in the same part.

Held-out convs 0, 4-9 (n=1108; recall n=1104), the one candidate and its controls:

| facts | recall [multi] | F1 | judge | multi-hop judge |
|---|---|---|---|---|
| existing graph | 0.887 [0.67] | 0.557 | 0.801 | 0.759 |
| re-extracted per-episode | 0.889 [0.67] | 0.564 | 0.813 | 0.754 |
| both + patterns | 0.892 [0.68] | 0.560 | 0.810 | 0.778 |

- both + patterns vs existing: judge +0.008 [-0.010, +0.026], F1 +0.003 [-0.010, +0.016]
- both + patterns vs re-extracted: judge -0.004 [-0.020, +0.012], F1 -0.004 [-0.017, +0.007]
- simply re-extracting moved judge by -0.028 on convs 1-3 and +0.012 here, so
  gaps under about 0.03 between separately extracted graphs are not evidence
