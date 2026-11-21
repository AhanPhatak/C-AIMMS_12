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
  This is an inference: a larger answer model was deliberately not tested. Gold turns
  understate the ceiling on temporal questions, and cutting evidence or adding
  chain-of-thought did not help either (2026-11-12, below).

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

## Evidence size, chain-of-thought and duration questions (2026-11-12, not adopted)

Tested whether the 4B answer model does better with less, cleaner evidence. The
default retrieval was run once for all 1540 questions and saved with the reranker
score of every evidence line (it matches the full run's evidence on 1540/1540), then
each arm replayed a cut of it. Convs 1-3, n=432; the replayed default reproduces the
full run's answers and judge verdicts on 432/432.

| Evidence | Items | F1 | Judge | Judge vs current [95% CI] | Multi-hop judge |
|---|---|---|---|---|---|
| **Current: 10 facts + 40 turns** | 50 | 0.563 | **0.810** | -- | 0.658 |
| 20 turns | 30 | 0.570 | 0.803 | -0.007 [-0.030, +0.016] | 0.633 |
| 10 turns | 20 | 0.558 | 0.801 | -0.009 [-0.037, +0.019] | 0.684 |
| Turns with reranker score >= 0, at least 5 | 28.7 | 0.561 | 0.799 | -0.012 [-0.035, +0.012] | 0.696 |
| Turns with reranker score >= 2, at least 5 | 22.2 | 0.556 | 0.792 | -0.019 [-0.046, +0.012] | 0.658 |
| No facts, 40 turns | 40 | 0.517 | 0.750 | -0.060 [-0.093, -0.028] | 0.595 |
| Current + chain-of-thought | 50 | 0.515 | 0.775 | -0.035 [-0.069, +0.000] | 0.709 |
| 20 turns + chain-of-thought | 30 | 0.523 | 0.803 | -0.007 [-0.044, +0.030] | 0.671 |

- No cut beats the current evidence, so the extra turns are not what holds the plain
  answer step back. Facts carry real weight: dropping them costs 6 judge points.
- Chain-of-thought does better with 20 turns than with 40 (+0.028), which fits the idea
  that noise hurts reasoning, but it still does not beat the plain answer step.
  With no arm ahead on the tuning set, nothing was run on held-out.

**Duration questions.** 42 questions match `how long / how old / how many
days|weeks|months|years` (F1 0.454 on the full run). A one-line instruction after the
question ("answer with the amount itself, not a date") raised F1 0.440 -> 0.486
(+0.046 [-0.021, +0.114], n=41 distinct) by shortening answers, but judge fell 0.619 ->
0.571. The model still cannot turn two dates into a duration, so it was reverted. The
earlier note that the temporal instruction turns durations into dates applies only to
the opt-in `OSAM_PROMPT_ENGINEERING=1` prompt; the default prompt has no temporal
instruction.

**Gold turns on temporal questions.** The gold-only run (convs 1-3) scored temporal F1
0.635 against 0.712 for retrieved evidence. Of the 12 temporal questions where it was
clearly worse, 7 have a gold turn that does not mention the event (for "When was
Joanna's second movie script shown on the big screens?" the gold turn is "Hey Joanna,
what's been up since we last chatted?") and the model rightly says it is not
mentioned; 4 more have only a relative date ("I got a puppy two weeks ago") that it
repeats. So the gold-evidence score understates the ceiling on temporal questions.

Rows, judge files, the saved evidence and scripts: `outputs/evidence_20261112/`.

## LongBench evidence (2026-11-12)

HyperMem's LongBench retrieval (repo defaults, paper models) and flat MiniLM + BM25 were
run once for every question of the six tasks and saved, then replayed through the
unmodified LongBench eval with different evidence. The replayed default reproduces the
full run on 1150/1150 answers. The first half of each task (by row order) chose the
arm; the second half reports it. F1 is LongBench's own; "mean" averages the six tasks.

| Evidence (12 passages unless noted) | Tuning mean | Held-out mean | All 1150 vs default [95% CI] |
|---|---|---|---|
| **Default:** 10 facts + passages, re-sorted by the relevance filter | 0.404 | 0.449 | -- |
| No facts | 0.408 | 0.444 | -0.000 [-0.012, +0.011] |
| Reranker score >= 0, at least 4 passages | 0.401 | 0.437 | -0.011 (held-out) |
| Flat MiniLM + BM25 | 0.391 | 0.423 | -0.019 [-0.034, -0.005] |
| No re-sort (retriever's order) | 0.401 | 0.447 | -0.002 [-0.017, +0.012] |
| No re-sort, no facts | 0.405 | 0.436 | -0.006 [-0.021, +0.009] |
| Facts + passages in document order | 0.408 | 0.449 | +0.002 [-0.012, +0.017] |
| **No facts, passages in document order** | **0.423** | **0.464** | **+0.017 [+0.002, +0.032]** |
| No facts, 16 passages in document order | 0.424 | 0.452 | +0.012 [-0.004, +0.028] |
| Flat MiniLM + BM25, no re-sort | 0.385 | 0.425 | -0.021 [-0.037, -0.005] |

Per task, all 1150 (default -> no facts, document order): qasper 0.422 -> 0.459,
hotpotqa 0.591 -> 0.610, multifieldqa_en 0.498 -> 0.524, 2wikimqa 0.486 -> 0.479,
musique 0.305 -> 0.330, narrativeqa 0.257 -> 0.257. On the held-out half alone the gain
is +0.015 [-0.005, +0.037], so this is moderate evidence, not proof.

- `filter_evidence_by_relevance` (MiniLM cosine >= 0.30) dropped almost nothing here,
  but it re-sorts everything by MiniLM score, so the reranker's order never reached the
  answer model. Turning the re-sort off alone changes nothing.
- The gain needs both changes: document order with facts, or no facts in relevance
  order, are no better than the default. Read as passages of the source in their own
  order, the evidence keeps the document's flow; facts are short paraphrases that add
  nothing the passages do not already say.
- HyperMem does beat flat retrieval on LongBench overall (-0.019 for flat), despite
  trailing it on qasper (+0.02 for flat). The earlier first-50 qasper result was real
  but did not generalise to the other tasks.
- Now available as `LB_EVIDENCE_LAYERS=episodic LB_EVIDENCE_ORDER=document` with
  `LB_RETRIEVER=hypermem` (previously IterRet only). Checked live on qasper: 199/200
  answers match the replay, F1 0.459. Not the default yet.

Rows, saved retrieval and scripts: `outputs/longbench_evidence_20261112/`.

### Same re-sort on LoCoMo (2026-11-13, not adopted)

The LoCoMo eval runs the same MiniLM re-sort. `WORKMEM_EVIDENCE_ORDER=retriever` keeps
HyperMem's order instead (`HYPERMEM_TURN_ORDER`, `HYPERMEM_FACTS_POS`). Replayed on the
saved retrieval, convs 1-3 (n=432), stemmed F1 and judge, vs the default (F1 0.563, judge 0.810):

| Evidence order | F1 | Judge | Judge vs default |
|---|---|---|---|
| Reranker order, facts first | 0.563 | 0.812 | +0.002 [-0.023, +0.028] |
| Reranker order, facts last | 0.570 | 0.789 | -0.021 [-0.049, +0.007] |
| Chronological, facts first | 0.554 | 0.808 | -0.002 [-0.032, +0.028] |
| Chronological, facts last | 0.568 | 0.801 | -0.009 [-0.044, +0.025] |

No order beats the re-sort; facts last costs judge accuracy. Unlike LongBench, LoCoMo
needs the facts (dropping them was -0.060), and conversation turns read out of
order lose less than document passages do. Rows: `outputs/evidence_20261112/rows/ro_*`.

## δ-mem adapter on vs off (2026-11-20)

`OSAM_DELTA_GAIN=0` zeroes both δ-mem corrections (q and o; the adapter has no other
weights), leaving plain Qwen3-4B on the same evidence and prompt. Rows confirm it:
`mean_delta_o_ratio` 0.0 vs 0.034 with the adapter on. Replayed on the saved
retrieval; the LoCoMo control is the full run (`outputs/full_run_20261105/`), which
the replay reproduces 432/432 on convs 1-3. Only 537 of 1540 LoCoMo answers stay the
same without the adapter.

| Adapter off minus on | F1 | Judge |
|---|---|---|
| LoCoMo, all 1540 (on: F1 0.559, judge 0.804) | -0.031 [-0.040, -0.021] | -0.010 [-0.024, +0.003] |
| LongBench default, all 1150 (on: F1 0.426) | -0.036 [-0.048, -0.024] | -0.002 [-0.019, +0.015] |
| LongBench passages only, document order (on: F1 0.443) | -0.041 [-0.054, -0.029] | not judged |

- The adapter raises F1 on both benchmarks, but the judge barely moves: by
  correctness it is neutral to slightly positive. LoCoMo judge per category, on vs
  off: open-domain 0.385 vs 0.312, temporal 0.832 vs 0.813, multi-hop 0.734 vs
  0.727, single-hop 0.864 vs 0.863.
- On LongBench the gain is answer format. Without the adapter, answers grow longer
  (musique median 3 -> 16 words, often a full sentence around the answer), which
  LongBench's F1 penalises and the judge does not. The adapter was trained on Qasper
  and keeps answers short and extractive.
- On LoCoMo it is not length: answers are shorter without the adapter (12.7 vs 20.4
  words) and F1 still drops.
- So keep the adapter on. Its F1 gains are real, but they are not evidence that it
  supplies memory the prompt lacks. This agrees with the earlier handoff finding
  that δ-mem's guidance is about neutral on correctness.
- The LongBench judge here is Qwen3-4B with every reference answer joined by OR. It is
  lenient on the long free-form answers of qasper and narrativeqa.

Rows and scripts: `outputs/delta_off_20261120/`.

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

- The default HyperMem run has no judge scores of its own; judge its rows afterwards
  with `scripts/judge_rows.py`.
- The LongBench passages-only arm has no judge scores; its +0.017 is F1 only.
- The IterRet-style gap step (idea 7) is untested and low priority.
- LongBench `LB_EVIDENCE_LAYERS=episodic LB_EVIDENCE_ORDER=document` (+0.017) is
  opt-in, not yet the default for `LB_RETRIEVER=hypermem`.

## Where the data is

- `outputs/hypermem_results_20261105/`: answer rows, judge rows, recall tables, scripts and
  test graphs. Every F1 and judge score above from these runs was recomputed from these rows.
- `outputs/cot_test_20261103/`: chain-of-thought, gold-evidence and general-prompt runs.
- `outputs/multihop_20261104/`: speaker boost and rewritten-query runs.
- `outputs/evidence_20261112/`: evidence-size, chain-of-thought and duration runs.
- `outputs/longbench_evidence_20261112/`: LongBench evidence runs.
- `outputs/delta_off_20261120/`: δ-mem on vs off (adapter-off rows, LongBench judge rows).
- `adaptive_memory_structures/README_HYPERGRAPH.md`: method notes and settings.
