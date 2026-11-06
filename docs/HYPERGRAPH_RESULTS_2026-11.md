# HyperMem improvement results, 2026-11-03 to 2026-11-05

Goal: get the HyperMem retriever past its plateau on LoCoMo, using ideas from the
HyperMem paper (arXiv:2604.08256). The answer model stayed at Qwen3-4B + δ-mem adapter
throughout, so that results stay comparable with older runs.

## Outcome

| Held-out conversations | F1 | Judge | Multi-hop judge | Recall | Multi-hop recall |
|---|---|---|---|---|---|
| Start: MiniLM, ms-marco | 0.541 | 0.781 | 0.675 | 0.830 | 0.46 |
| **Now: paper models + speaker boost** | **0.557** | **0.801** | **0.759** | **0.887** | **0.67** |
| Change [95% CI] | +0.016 [+0.002, +0.029] | +0.021 [+0.003, +0.039] | +0.084 [+0.025, +0.148] | +0.057 | +0.21 |

- Only better retrieval and the speaker boost were adopted. Chain-of-thought, general
  prompts, query rewriting, paper-style formation, topic facts and query patterns
  were not.
- Most of the judge gain (+0.015 of +0.021) comes from the speaker boost, which costs nothing. Against
  old models + boost, the paper models add judge +0.005 [-0.011, +0.021], and
  multi-hop judge +0.064 [+0.010, +0.123], for about 17GB of extra GPU memory.
- Scores stay near 0.80-0.81 whatever changes. Gold evidence reached only 0.806 on
  convs 1-3 with the standard prompt, so the 4B answer model looks like the limit.
  This is an inference: a larger answer model was deliberately not tested.

## How to read the numbers

- **Sets:** tuning is convs 1-3 (n=432). Held-out is convs 0, 4-9 (n=1108 for answers,
  n=1104 for recall). Many comparisons were made on held-out, so it is not perfectly
  clean, and an interval that only just clears zero is weak evidence.
- **F1:** token F1 with Porter stemming. Scores before 2026-11-03 were unstemmed because
  `nltk` was missing; the full 1540-question default moves from 0.532 to 0.546 (judge 0.783).
- **Judge:** `iterret/llm_judge.py` on Qwen3-4B. It is noisy and not comparable with the
  paper's 92.73% (GPT-4.1-mini answers, GPT-4o-mini judge).
- **Recall:** the share of questions with every gold turn retrieved.
- **Intervals:** paired bootstrap 95% CIs.
- **Rebuild noise:** re-extracting the same facts moved judge by -0.028 on convs 1-3 and
  +0.012 on held-out. Gaps under about 0.03 between separately built graphs are not evidence.

## Results by experiment

### Answering: chain-of-thought and gold evidence (convs 1-3, not adopted)

| Evidence | F1 | Judge |
|---|---|---|
| Current | 0.549 | 0.785 |
| Current + chain-of-thought | 0.510 | 0.780 |
| Gold turns only | 0.612 | 0.806 |
| Gold turns + chain-of-thought | 0.586 | 0.829 |

The LoCoMo-tuned prompt scored judge 0.817 on this set, but it was excluded from
further testing because parts of it are specific to the dataset.

### General prompts (held-out, not adopted)

| Prompt | F1 | F1 change | Judge | Judge change | Words |
|---|---|---|---|---|---|
| Standard | 0.541 | -- | 0.781 | -- | 14.3 |
| General | 0.534 | -0.007 [-0.019, +0.005] | 0.785 | +0.005 [-0.013, +0.022] | 10.7 |
| General + answer first | 0.557 | +0.015 [+0.002, +0.028] | 0.768 | -0.013 [-0.032, +0.007] | 7.2 |

The F1 gain comes with much shorter answers and no gain in judged accuracy.

### Retrieval (held-out)

| Config | Recall | Multi-hop recall | F1 | Judge | Multi-hop judge | Status |
|---|---|---|---|---|---|---|
| Old models | 0.830 | 0.46 | 0.541 | 0.781 | 0.675 | previous default |
| + speaker boost | 0.847 | 0.54 | 0.548 | 0.796 | 0.695 | one-GPU option |
| + boost + rewritten queries | 0.858 | 0.54 | 0.551 | 0.794 | 0.709 | not adopted |
| Reranker only + boost | not measured | not measured | 0.551 | 0.792 | 0.714 | not adopted |
| Paper models | 0.883 | 0.63 | 0.559 | 0.798 | 0.719 | -- |
| **Paper models + boost** | **0.887** | **0.67** | **0.557** | **0.801** | **0.759** | **default** |

"Paper models" means Qwen3-Embedding-4B and Qwen3-Reranker-4B, with reranking of turns
and of the top 100 topics, episodes and facts.

### Graph formation (convs 1-3, default retrieval, not adopted)

| Graphs | Episodes | Recall | F1 | Judge | Judge vs existing |
|---|---|---|---|---|---|
| Existing (0.5B surprise) | 201 | 0.887 | 0.563 | 0.810 | -- |
| Rebuild, same recipe | 201 | 0.887 | 0.551 | 0.796 | -0.014 [-0.039, +0.012] |
| Qwen3-4B surprise | 249 | 0.880 | 0.557 | 0.810 | +0.000 [-0.032, +0.032] |
| Qwen3-4B LLM boundaries | 237 | 0.889 | 0.567 | 0.803 | -0.007 [-0.037, +0.023] |

The paper names no formation model, so these are the closest tests rather than a copy.

### Topic facts and query patterns (not adopted)

Convs 1-3, query patterns off -> on:

| Facts | F1 | Judge | Multi-hop judge |
|---|---|---|---|
| Existing graph | 0.563 | 0.810 | 0.658 |
| Re-extracted per-episode | 0.568 -> 0.566 | 0.782 -> 0.799 | 0.570 -> 0.633 |
| Topic, replacing them | 0.508 -> 0.505 | 0.759 -> 0.757 | 0.646 -> 0.671 |
| Both | 0.572 -> 0.581 | 0.794 -> 0.810 | 0.608 -> 0.658 |
| Strict topics, topic facts | 0.548 -> 0.536 | 0.803 -> 0.785 | 0.747 -> 0.671 |

Held-out:

| Facts | Recall | F1 | Judge | Multi-hop judge |
|---|---|---|---|---|
| Existing graph | 0.887 | 0.557 | 0.801 | 0.759 |
| Re-extracted per-episode | 0.889 | 0.564 | 0.813 | 0.754 |
| Both + patterns | 0.892 | 0.560 | 0.810 | 0.778 |

Both + patterns vs existing: judge +0.008 [-0.010, +0.026]. Vs re-extracted: judge
-0.004 [-0.020, +0.012]. Long topics were split across calls, so facts could combine
only episodes in the same part, which may have limited this test. Strict topics were
not run on held-out.

### Run scripts (conv 0 smoke, 152 questions, all exit 0)

| Run | Layout | F1 | Check |
|---|---|---|---|
| HyperMem, paper models | answer model on one GPU, retrieval on the other, no vLLM | 0.535 | 151/152 answers match the saved run |
| HyperMem, old models + boost | vLLM on one GPU, eval on the other | 0.519 | 152/152 answers match the saved run |
| IterRet | vLLM on one GPU, eval on the other | 0.492 | ran normally; no current baseline to match |

One conversation, so this checks the scripts and does not rank the setups. A full
10-conversation run through the scripts has not been done.

## Full end-to-end run (2026-11-05/06)

One run with the current defaults: LoCoMo through `scripts/run_everything.sh --full
--retriever hypermem`, then all six LongBench QA tasks, every question. Driver, logs
and rows: `outputs/full_run_20261105/`. About 8h20m, including 1h20m lost to a
driver bug (since fixed in its `run_all.sh`).

**LoCoMo (1540 questions, 0 skipped):** F1 0.5589, judge 0.804. Identical F1 to the
combined held-out + convs 1-3 rows above; 1522 of 1540 answers match word for word.

| LoCoMo, all 1540 | Repo at clone | Full run | Change [95% CI] |
|---|---|---|---|
| F1, unstemmed (the repo's 0.5320) | 0.5320 | 0.5430 | +0.011 [-0.001, +0.023] |
| F1, stemmed | 0.546 | 0.559 | +0.013 [+0.001, +0.024] |
| Judge | 0.783 | 0.804 | +0.021 [+0.005, +0.036] |
| Multi-hop F1 / judge | 0.439 / 0.660 | 0.504 / 0.730 | |
| Temporal F1 / judge | 0.693 / 0.838 | 0.690 / 0.832 | |

The unstemmed and CI figures are from the combined rows, which match this run's F1.

**LongBench (LongBench's own F1, no judge, 0 skipped):**

| Task | Questions | F1 | Earlier HyperMem, same questions |
|---|---|---|---|
| qasper | 200 | 0.4220 | first 50: 0.3728 -> 0.3671, -0.006 [-0.024, +0.008] |
| hotpotqa | 200 | 0.5909 | first 100: 0.5516 -> 0.5569, +0.005 [-0.034, +0.042] |
| multifieldqa_en | 150 | 0.4978 | none |
| 2wikimqa | 200 | 0.4862 | none |
| musique | 200 | 0.3047 | none |
| narrativeqa | 200 | 0.2573 | none |

The paper models do not change LongBench measurably where earlier rows exist. The
speaker boost has nothing to act on in documents. On the first 50 qasper questions,
HyperMem without PRF, rerank and facts scored 0.4275 and flat MiniLM + BM25 0.4048,
both above this run's 0.3671 (CIs include zero, n=50).

## Current defaults

- **On:** `HYPERMEM_SPEAKER_BOOST=1`, `HYPERMEM_EMBED_MODEL=Qwen/Qwen3-Embedding-4B`,
  `HYPERMEM_RERANK_MODEL=Qwen/Qwen3-Reranker-4B`, `HYPERMEM_LAYER_RERANK=1`.
- **Off:** `HYPERMEM_FACT_SCOPE=episode`, `HYPERMEM_FACT_PATTERNS=0`, `HYPERMEM_TOPIC_STRICT=0`.
- **Unchanged:** Qwen3-4B + δ-mem answers, Qwen2.5-0.5B surprise segmentation, existing graphs.
- **Requirements:** two GPUs and no vLLM during the eval, all graphs cached, no `WORKMEM_JUDGE=1`.
- **Previous stack, for one eval GPU beside vLLM:**
  `HYPERMEM_EMBED_MODEL= HYPERMEM_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 HYPERMEM_LAYER_RERANK=0`.
- **LongBench:** `LB_RETRIEVER=hypermem` through `run_pipeline.sh` must use the previous stack.

## Still open

- `How long` questions trigger the temporal instruction, which can turn a duration into a date.
- The default HyperMem run has no judge scores, and the separate judging script is not in the repo.
- The δ-mem adapter has not been re-tested on the new evidence.
- The IterRet-style gap step (idea 7) is untested and low priority.
- Why gold turns alone score lower on temporal questions is unexplained.

## Where the data is

- `outputs/hypermem_results_20261105/`: answer rows, judge rows, recall tables, scripts and
  test graphs. Every F1 and judge score above from these runs was recomputed from these rows.
- `outputs/cot_test_20261103/`: chain-of-thought, gold-evidence and general-prompt runs.
- `outputs/multihop_20261104/`: speaker boost and rewritten-query runs.
- `adaptive_memory_structures/README_HYPERGRAPH.md`: method notes and settings.
