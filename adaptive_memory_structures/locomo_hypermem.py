"""
HyperMem as a drop-in retriever for the LoCoMo delta-Mem eval
(delta-Mem/deltamem/workmem/eval_locomo_iterret_mock.py, WORKMEM_RETRIEVER=hypermem).

Build (once per conversation, cached as JSON):
  - each LoCoMo session's turns are paired into Pages (as locomo_loader does),
    with relative dates resolved against the session timestamp exactly like
    the IterRet graph does, and the session date prefixed onto each Page
  - each session is surprise-segmented into episodes on its own: the segmenter
    runs one forward pass over its whole input, and a full conversation
    (~30k tokens x ~150k vocab logits) does not fit in memory
  - every episode gets an LLM subject/summary, LLM-extracted facts and a
    FactHyperedge (same functions HypergraphMemory.build_index uses)
  - topics are formed by streaming ALL episodes of the conversation in
    temporal order, so a topic can span sessions -- HyperMem's own setting,
    rather than HypergraphMemory's per-session one

Retrieve (per question): the same coarse-to-fine traversal as
HypergraphMemory.retrieve (topic -> hard-filtered episodes -> hard-filtered
facts -> their pages), with embedding propagation, but
  - embeddings come from Qwen3-Embedding-4B, HyperMem's own encoder
    (HYPERMEM_EMBED_MODEL; empty = the caller's sentence encoder, which the
    eval passes as MiniLM), not mean-pooled causal-LM states, which retrieved
    poorly in testing
  - top-k's are sized for a whole conversation, not one session
  - output is the selected pages' individual turns, rendered
    "[date] Speaker: text" like IterRet's evidence, so everything downstream
    (relevance filter, OSAM write, answer prompt) is unchanged
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hypergraph_embedding import propagate_episode_embeddings, propagate_fact_embeddings  # noqa: E402
from hypergraph_types import EpisodeHyperedge, EpisodeNode, FactHyperedge, FactNode, TopicNode  # noqa: E402
from memory_structures import Page  # noqa: E402

BUILD_VERSION = 1


def _session_nums(conv_block: dict) -> list[int]:
    nums = []
    for k in conv_block:
        if k.startswith("session_") and not k.endswith("_date_time"):
            try:
                nums.append(int(k.split("_")[1]))
            except ValueError:
                pass
    return sorted(nums)


def _render_turn(turn: dict, session_ts: str) -> str:
    from iterret.time_resolution import date_only, resolve_relative_time
    text, _ = resolve_relative_time(turn.get("text", ""), session_ts)
    return f"[{date_only(session_ts)}] {turn.get('speaker', 'Unknown')}: {text}"


def _pages_for_session(turns: list[dict], session_ts: str) -> list[tuple[Page, list[str]]]:
    """Pair consecutive turns into Pages; also return each Page's rendered turns."""
    out = []
    for i in range(0, len(turns), 2):
        rendered = [_render_turn(t, session_ts) for t in turns[i:i + 2]]
        page = Page(user_text=rendered[0], agent_text=rendered[1] if len(rendered) > 1 else "")
        out.append((page, rendered))
    return out


class _DocPage(Page):
    """A document passage as a Page: plain text, no User:/Agent: framing."""

    def to_text(self) -> str:
        return self.user_text


def _fact_json(f: FactNode, scope: str = "episode") -> dict:
    return {"id": f.id, "episode_id": f.episode_id, "content": f.content, "confidence": f.confidence,
            "temporal": f.temporal, "spatial": f.spatial, "keywords": f.keywords,
            "query_patterns": f.query_patterns, "scope": scope}


def add_topic_facts(data: dict, llm_client: Any, *, replace: bool, workers: int = 1) -> dict:
    """HyperMem's fact extraction: facts come from all of a topic's episodes read
    together, on top of (or, with ``replace``, instead of) the per-episode facts.
    Episodes, topics and pages are unchanged. Returns a new graph dict."""
    from concurrent.futures import ThreadPoolExecutor

    from hypergraph_extraction import assign_fact_roles, extract_facts_for_topic

    if not data["topics"]:
        raise ValueError("topic-level facts need LLM topics (HYPERMEM_BUILD_TOPICS=llm)")
    page_turns = {p["id"]: p["turns"] for p in data["pages"]}
    episodes = {e["id"]: e for e in data["episodes"]}
    order = {e["id"]: i for i, e in enumerate(data["episodes"])}

    def extract(topic):
        ids = sorted((e for e in topic["episode_ids"] if e in episodes), key=order.__getitem__)
        segments = [(e, "\n".join(t for p in episodes[e]["page_ids"] for t in page_turns[p])) for e in ids]
        return extract_facts_for_topic(topic["title"], topic["summary"], segments, llm_client)

    def roles(job):
        eid, ep_facts = job
        return assign_fact_roles(ep_facts, eid, episodes[eid]["summary"], llm_client)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        by_episode: dict[str, list[FactNode]] = {}
        for facts in pool.map(extract, data["topics"]):
            for f in facts:
                by_episode.setdefault(f.episode_id, []).append(f)
        jobs = sorted(by_episode.items(), key=lambda job: order[job[0]])
        edges = list(pool.map(roles, jobs))

    new_facts = [_fact_json(f, "topic") for _, facts in jobs for f in facts]
    new_edges = [{"id": h.id, "episode_id": h.episode_id, "relation": h.relation, "weights": h.weights} for h in edges]
    return {**data,
            "facts": new_facts if replace else data["facts"] + new_facts,
            "fact_hyperedges": new_edges if replace else data["fact_hyperedges"] + new_edges,
            "params": {**data["params"], "fact_scope": "topic" if replace else "both"}}


def _build_hypergraph(
    sessions: list[tuple[int, str, list[tuple[Page, list[str]]]]],
    llm_client: Any,
    seg_model: Any,
    seg_tokenizer: Any,
    *,
    gamma: float,
    min_block_size: int,
    topic_match_batch_size: int,
    log: Callable[[str], None],
    topics_mode: str = "llm",
    workers: int = 1,
) -> dict:
    """``sessions``: (number, timestamp, [(Page, rendered turns)]) in order. Each
    is surprise-segmented on its own; topics are streamed over all episodes.

    ``workers`` > 1 runs the per-episode LLM calls (summary, facts, fact roles)
    for different episodes concurrently; each episode's three calls stay in
    order, and the output keeps segmentation order. ``topics_mode`` "none"
    skips the LLM topic stage (the retriever then clusters episodes itself,
    see HYPERMEM_RETOPIC): it is the serial, most expensive part of the build
    and measured to add no recall in soft mode (README_HYPERGRAPH.md)."""
    from concurrent.futures import ThreadPoolExecutor

    from hypergraph_extraction import (
        assign_fact_roles,
        build_topics_for_session,
        extract_facts_for_episode,
        summarize_episode,
    )
    from surprise_episode_segmenter import PageEpisodeSegmenter

    if topics_mode not in ("llm", "none"):
        raise ValueError(f"topics_mode must be 'llm' or 'none', got {topics_mode!r}")
    segmenter = PageEpisodeSegmenter(gamma=gamma, min_block_size=min_block_size)
    t0 = time.time()

    # 1. segment every session (in-process LM, sequential)
    segmented: list[tuple[int, str, dict, list[list[Page]]]] = []
    for snum, session_ts, paged in sessions:
        pages = [p for p, _ in paged]
        groups = segmenter.segment(pages, seg_model, seg_tokenizer)
        segmented.append((snum, session_ts, {p.page_id: r for p, r in paged}, groups))
        log(f"[hypermem] session {snum}: {len(pages)} pages -> {len(groups)} episodes "
            f"({time.time() - t0:.0f}s)")

    # 2. per-episode LLM extraction, episodes in parallel
    jobs = [(snum, ts, rendered, group) for snum, ts, rendered, groups in segmented for group in groups]

    def extract(job):
        group = job[3]
        subject, summary = summarize_episode(group, llm_client)
        episode = EpisodeNode(page_ids=[p.page_id for p in group], subject=subject, summary=summary)
        ep_facts = extract_facts_for_episode(group, episode.id, llm_client)
        he = assign_fact_roles(ep_facts, episode.id, summary, llm_client)
        return episode, ep_facts, he

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(extract, jobs))

    pages_out: list[dict] = []
    episodes: list[EpisodeNode] = []
    episode_session: dict[str, int] = {}
    facts: dict[str, FactNode] = {}
    fact_hyperedges: dict[str, FactHyperedge] = {}
    for (snum, session_ts, rendered_by_id, group), (episode, ep_facts, he) in zip(jobs, results):
        episodes.append(episode)
        episode_session[episode.id] = snum
        facts.update((f.id, f) for f in ep_facts)
        fact_hyperedges[he.id] = he
        for p in group:
            pages_out.append({
                "id": p.page_id, "session": snum, "date": session_ts,
                "turns": rendered_by_id[p.page_id], "episode_id": episode.id,
            })
    log(f"[hypermem] extracted {len(episodes)} episodes / {len(facts)} facts in "
        f"{time.time() - t1:.0f}s ({workers} workers)")

    # 3. topics: streamed over all episodes in temporal order (inherently serial)
    if topics_mode == "llm":
        t2 = time.time()
        topics, episode_hyperedges = build_topics_for_session(
            episodes, llm_client, batch_size=topic_match_batch_size,
            strict=_env_int("HYPERMEM_TOPIC_STRICT", 0) == 1,
        )
        log(f"[hypermem] topics: {len(topics)} in {time.time() - t2:.0f}s")
    else:
        topics, episode_hyperedges = {}, {}
    log(f"[hypermem] built: {len(topics)} topics, {len(episodes)} episodes, "
        f"{len(facts)} facts in {time.time() - t0:.0f}s")

    data = {
        "version": BUILD_VERSION,
        "params": {"gamma": gamma, "min_block_size": min_block_size, "topics": topics_mode},
        "topics": [{"id": t.id, "title": t.title, "summary": t.summary, "episode_ids": t.episode_ids}
                   for t in topics.values()],
        "episodes": [{"id": e.id, "subject": e.subject, "summary": e.summary, "page_ids": e.page_ids,
                      "session": episode_session[e.id]} for e in episodes],
        "facts": [_fact_json(f) for f in facts.values()],
        "episode_hyperedges": [{"id": h.id, "topic_id": h.topic_id, "relation": h.relation,
                                "weights": h.weights} for h in episode_hyperedges.values()],
        "fact_hyperedges": [{"id": h.id, "episode_id": h.episode_id, "relation": h.relation,
                             "weights": h.weights} for h in fact_hyperedges.values()],
        "pages": pages_out,
    }
    fact_scope = os.environ.get("HYPERMEM_FACT_SCOPE", "episode")  # episode | topic | both
    if fact_scope != "episode":
        data = add_topic_facts(data, llm_client, replace=fact_scope == "topic", workers=workers)
        log(f"[hypermem] {fact_scope} facts: {len(data['facts'])}")
    return data


def build_conversation_hypergraph(
    conv_block: dict,
    llm_client: Any,
    seg_model: Any = None,
    seg_tokenizer: Any = None,
    *,
    gamma: float = 0.5,
    min_block_size: int = 1,
    topic_match_batch_size: int = 10,
    topics_mode: str = "llm",
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> dict:
    """Build the whole-conversation hypergraph. Returns a JSON-serialisable dict."""
    sessions = []
    for snum in _session_nums(conv_block):
        turns = conv_block.get(f"session_{snum}") or []
        if not turns:
            continue
        session_ts = conv_block.get(f"session_{snum}_date_time", f"Session {snum}")
        sessions.append((snum, session_ts, _pages_for_session(turns, session_ts)))
    return _build_hypergraph(sessions, llm_client, seg_model, seg_tokenizer, gamma=gamma,
                             min_block_size=min_block_size,
                             topic_match_batch_size=topic_match_batch_size,
                             topics_mode=topics_mode, workers=workers, log=log)


def build_document_hypergraph(
    passages: list[str],
    llm_client: Any,
    seg_model: Any = None,
    seg_tokenizer: Any = None,
    *,
    gamma: float = 0.5,
    min_block_size: int = 1,
    topic_match_batch_size: int = 10,
    window: int = 16,
    topics_mode: str = "llm",
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> dict:
    """The same hypergraph over a DOCUMENT (LongBench): one Page per passage,
    one "turn" per Page (the passage itself, undated). The passages are
    surprise-segmented ``window`` at a time, playing the role LoCoMo sessions
    play for a conversation (the segmenter runs one forward pass per input)."""
    sessions = []
    for n, start in enumerate(range(0, len(passages), window)):
        paged = [(_DocPage(user_text=t), [t]) for t in passages[start:start + window]]
        sessions.append((n, "", paged))
    return _build_hypergraph(sessions, llm_client, seg_model, seg_tokenizer, gamma=gamma,
                             min_block_size=min_block_size,
                             topic_match_batch_size=topic_match_batch_size,
                             topics_mode=topics_mode, workers=workers, log=log)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


_STOP = set("""a an the and or but of to in on at for with by from as is are was were be been being
do does did have has had i you he she it we they me him her them my your his its our their this that
these those what which who whom when where why how there here not no so if then than too very can will
just about into over after before again all any both each few more most other some such only own same
s t don should now""".split())


def _tokenize(text: str) -> list[str]:
    import re
    from nltk.stem import PorterStemmer
    stem = _tokenize._stem = getattr(_tokenize, "_stem", None) or PorterStemmer().stem
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


class _BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.toks = [_tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = (sum(map(len, self.toks)) / max(1, len(self.toks))) or 1.0
        df: dict[str, int] = {}
        for t in self.toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(self.toks)
        self.idf = {w: float(np.log(1 + (n - c + 0.5) / (c + 0.5))) for w, c in df.items()}
        self.tf = [{w: t.count(w) for w in set(t)} for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = set(_tokenize(query))
        out = np.zeros(len(self.toks), dtype=np.float32)
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            out[i] = sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return out


def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank of each item (0 = best)."""
    r = np.empty(len(scores), dtype=np.int64)
    r[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
    return r


_TURN_SPEAKER = re.compile(r"^\[[^\]]*\] ([^:]+):")  # the "[date] Name:" prefix from _render_turn


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _retrieval_device(env_name: str) -> str:
    """The retrieval models don't fit next to the answer model on a 24GB card, so
    unless told otherwise they go on the last visible GPU (answer model: cuda:0)."""
    device = os.environ.get(env_name)
    if device:
        return device
    import torch
    return f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu"


class _Qwen3Reranker:
    """Yes/no-logit scoring, as on the Qwen3-Reranker model card. Same predict()
    call as sentence-transformers' CrossEncoder."""

    INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
    SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct '
              'provided. Note that the answer can only be "yes" or "no".')
    MAX_LEN = 2048
    MAX_BATCH = 16  # a batch of 64 runs a 24GB card out of memory

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(f"<|im_start|>system\n{self.SYSTEM}<|im_end|>\n<|im_start|>user\n",
                                      add_special_tokens=False)
        self.suffix = self.tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                      add_special_tokens=False)

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = MAX_BATCH, **_ignored: Any) -> np.ndarray:
        import torch

        out = np.zeros(len(pairs), dtype=np.float32)
        batch_size = min(batch_size, self.MAX_BATCH)
        room = self.MAX_LEN - len(self.prefix) - len(self.suffix)
        with torch.no_grad():
            for b in range(0, len(pairs), batch_size):
                texts = [f"<Instruct>: {self.INSTRUCT}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs[b:b + batch_size]]
                enc = self.tok(texts, padding=False, truncation="longest_first", max_length=room,
                               return_attention_mask=False)
                ids = [self.prefix + x + self.suffix for x in enc["input_ids"]]
                batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self.device)
                # the head on the last position only: full logits are length x ~150k vocab
                logits = self.model.lm_head(self.model.model(**batch).last_hidden_state[:, -1, :])
                out[b:b + batch_size] = (logits[:, self.yes] - logits[:, self.no]).float().cpu().numpy()
        return out


# Loaded once per process: these take a minute and ~8GB each, and the eval builds
# one retriever per conversation.
_LOADED_MODELS: dict[str, Any] = {}


def _sentence_encoder(model_name: str) -> Any:
    if model_name not in _LOADED_MODELS:
        import torch
        from sentence_transformers import SentenceTransformer

        device = _retrieval_device("HYPERMEM_EMBED_DEVICE")
        kwargs = {} if device == "cpu" else {"model_kwargs": {"torch_dtype": torch.float16}}
        model = SentenceTransformer(model_name, device=device, processor_kwargs={"padding_side": "left"}, **kwargs)
        model.max_seq_length = min(model.max_seq_length or 1024, 1024)  # longer texts run out of memory
        _LOADED_MODELS[model_name] = model
    return _LOADED_MODELS[model_name]


def _reranker(model_name: str) -> Any:
    if model_name not in _LOADED_MODELS:
        device = _retrieval_device("HYPERMEM_RERANK_DEVICE")
        try:
            if "Qwen3-Reranker" in model_name:
                _LOADED_MODELS[model_name] = _Qwen3Reranker(model_name, device)
            else:
                from sentence_transformers import CrossEncoder
                _LOADED_MODELS[model_name] = CrossEncoder(model_name, device=device)
        except Exception as exc:  # noqa: BLE001
            # no silent fallback: a run without the reranker is a different retriever
            raise RuntimeError(
                f"HyperMem rerank model {model_name!r} failed to load ({exc}). Run "
                "scripts/download_assets.sh, or set HYPERMEM_RERANK=0 to retrieve without it.") from exc
    return _LOADED_MODELS[model_name]


class FlatHybridRetriever:
    """Control: MiniLM + BM25 rank fusion over the bare turns / passages -- no
    graph, no LLM. Same interface as HyperMemRetriever.retrieve."""

    def __init__(self, turns: list[str], encode_batch: Callable[[list[str]], Any], *, max_turns: int = 40):
        self.turns, self.max_turns = list(turns), max_turns
        self._encode_batch = encode_batch
        self.U = np.stack([_unit(np.asarray(v, dtype=np.float32)) for v in encode_batch(self.turns)])
        self.bm = _BM25(self.turns)

    def retrieve(self, question: str, diag: dict | None = None) -> list[str]:
        q = _unit(np.asarray(self._encode_batch([question])[0], dtype=np.float32))
        fused = 1 / (60 + _ranks(self.U @ q)) + 1 / (60 + _ranks(self.bm.scores(question)))
        chosen = [int(i) for i in np.argsort(-fused, kind="stable")[: self.max_turns]]
        if diag is not None:
            diag.update({"retriever": "flat-hybrid", "evidence_ids": chosen})
        return [self.turns[i] for i in chosen]


class HyperMemRetriever:
    """Retrieval over a built conversation hypergraph.

    mode "strict": the original coarse-to-fine traversal (HypergraphMemory.retrieve):
        top topics -> hard-filtered episodes -> hard-filtered facts -> their pages.
        On LoCoMo the hard topic filter is the bottleneck: one broad topic can
        hold half the conversation while the evidence sits under a topic that
        did not make the cut, so it is unreachable.

    mode "soft" (default): every layer is scored against the question with
        hybrid dense + BM25 reciprocal-rank fusion (HyperMem's own retrieval
        is hybrid; this port had dropped the lexical side), and the hierarchy
        becomes evidence aggregation instead of a filter: an episode's score
        fuses its own rank, its topic's rank (through the episode hyperedge),
        its best fact's rank (through the fact hyperedge) and its best turn's
        rank. Turns are then ranked individually, by their own hybrid score
        fused with their episode's rank, so a long episode contributes its
        relevant turns rather than all of them. On top of that, on by
        default: a second pass expanded with the top facts (HYPERMEM_PRF), a
        cross-encoder rerank of the candidate turns (HYPERMEM_RERANK), and the
        top facts, dated, prepended as evidence (HYPERMEM_FACTS).
    """

    def __init__(
        self,
        data: dict,
        encode: Callable[[str], Any],
        *,
        encode_batch: Callable[[list[str]], Any] | None = None,
        mode: str | None = None,
        alpha: float = 0.5,
        max_turns: int | None = None,
        n_facts: int | None = None,
    ):
        self.mode = mode or os.environ.get("HYPERMEM_MODE", "soft")
        self.max_turns = max_turns if max_turns is not None else _env_int("HYPERMEM_MAX_TURNS", 40)
        # top extracted facts, dated, prepended to the turns (soft mode default 10:
        # +0.05 token-F1 on held-out conversations, mostly temporal)
        self.n_facts = n_facts if n_facts is not None else _env_int(
            "HYPERMEM_FACTS", 10 if self.mode == "soft" else 0)
        # strict-mode knobs
        self.topic_top_k = _env_int("HYPERMEM_TOPIC_TOP_K", 3)
        self.episode_top_k = _env_int("HYPERMEM_EPISODE_TOP_K", 8)
        self.fact_top_k = _env_int("HYPERMEM_FACT_TOP_K", 20)
        # soft-mode weights (RRF, k = 20: tuned on LoCoMo convs 1-3 gold-evidence recall)
        self.rrf_k = _env_float("HYPERMEM_RRF_K", 20)
        self.w = {name: _env_float(f"HYPERMEM_W_{name.upper()}", dflt) for name, dflt in
                  [("episode", 1.0), ("topic", 0.5), ("fact", 1.0), ("eturn", 1.0), ("turn", 1.0), ("prior", 1.0)]}
        self.page_expand = _env_int("HYPERMEM_PAGE_EXPAND", 0)
        # also index the questions each fact was extracted to answer (HyperMem's
        # "potential" field), for dense and BM25 search; the evidence shows only the fact
        self.fact_patterns = _env_int("HYPERMEM_FACT_PATTERNS", 0)
        # max turns taken from any one episode (0 = no cap); spreads the budget
        # across episodes, which multi-hop questions need
        self.per_episode_cap = _env_int("HYPERMEM_PER_EPISODE_CAP", 0)
        # Step 1 (multi-hop, no LLM):
        #  PRF  -- a second pass whose query is the question expanded with the top
        #          round-1 facts (dense: Rocchio; lexical: their text + keywords),
        #          fused with the first pass. Finds the "second hop" that does not
        #          resemble the question but does resemble what the first hop found.
        #  LINK -- entity hyperedges: facts sharing a (rare) keyword or name with
        #          the top facts boost their episodes, so evidence about the same
        #          entity in other sessions becomes reachable.
        self.prf = _env_int("HYPERMEM_PRF", 1)
        self.prf_facts = _env_int("HYPERMEM_PRF_FACTS", 5)
        self.prf_beta = _env_float("HYPERMEM_PRF_BETA", 0.5)
        self.w_prf = _env_float("HYPERMEM_W_PRF", 1.0)
        self.link = _env_int("HYPERMEM_LINK", 0)
        self.link_seeds = _env_int("HYPERMEM_LINK_SEEDS", 5)
        self.w_link = _env_float("HYPERMEM_W_LINK", 1.0)
        # SPEAKER BOOST -- when the question names exactly one speaker, that speaker's
        # turns get +w/k in each pass. 0 = off. Missed multi-hop gold turns sit deep in
        # the ranking and are mostly spoken by the person the question names.
        self.speaker_boost = _env_float("HYPERMEM_SPEAKER_BOOST", 1.0)
        # Step 2 (precision): cross-encoder rerank of the top candidates, then an
        # adaptive cut instead of always filling max_turns.
        self.rerank = _env_int("HYPERMEM_RERANK", 1)
        self.rerank_pool = _env_int("HYPERMEM_RERANK_POOL", 120)
        self.rerank_model = os.environ.get("HYPERMEM_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
        self.w_ce = _env_float("HYPERMEM_W_CE", 1.0)        # CE rank vs hypergraph rank in the final order
        self.ce_cutoff = os.environ.get("HYPERMEM_CE_CUTOFF")  # None = no cut; else min CE logit kept
        self.min_turns = _env_int("HYPERMEM_MIN_TURNS", 8)
        # also rerank each pass's top topics, episodes and facts, HyperMem's own
        # retrieval (the turn rerank above is this fork's addition)
        self.layer_rerank = _env_int("HYPERMEM_LAYER_RERANK", 1) if self.rerank else 0
        self.layer_pool = _env_int("HYPERMEM_LAYER_POOL", 100)
        self._query = ""
        self._ce_memo: dict[str, float] = {}
        # Experimental: the LLM-streamed topics are very uneven on LoCoMo (the
        # largest holds 23-56% of a conversation's episodes, next to several
        # singletons). N > 0 regroups the episodes into ~N-episode topics by
        # embedding similarity (Ward clustering) and uses those instead, for both
        # the topic signal and the episode-hyperedge propagation. 0 = off.
        # A graph built without topics (HYPERMEM_BUILD_TOPICS=none) defaults to
        # ~4-episode clusters, the best of the sizes compared on recall.
        self.retopic = _env_int("HYPERMEM_RETOPIC", 0 if data.get("topics") else 4)
        # Order of the returned evidence (only matters when the caller keeps it,
        # WORKMEM_EVIDENCE_ORDER=retriever): turns by rank or chronologically,
        # facts before or after the turns.
        self.turn_order = os.environ.get("HYPERMEM_TURN_ORDER", "rank")    # rank | chrono
        self.facts_pos = os.environ.get("HYPERMEM_FACTS_POS", "first")     # first | last

        self._encode1 = encode
        self._encode_batch = encode_batch
        self._encode_query = None
        self.embed_model = os.environ.get("HYPERMEM_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
        if self.embed_model:
            st = _sentence_encoder(self.embed_model)
            query_prompt = "query" if "query" in st.prompts else None
            self._encode_batch = lambda texts: st.encode(texts, batch_size=8, show_progress_bar=False)
            self._encode_query = lambda text: st.encode([text], prompt_name=query_prompt, show_progress_bar=False)[0]

        self.topics = {t["id"]: TopicNode(id=t["id"], title=t["title"], summary=t["summary"],
                                          episode_ids=t["episode_ids"]) for t in data["topics"]}
        self.episodes = {e["id"]: EpisodeNode(id=e["id"], page_ids=e["page_ids"], subject=e["subject"],
                                              summary=e["summary"]) for e in data["episodes"]}
        self.facts = {f["id"]: FactNode(id=f["id"], episode_id=f["episode_id"], content=f["content"],
                                        confidence=f["confidence"], temporal=f.get("temporal"),
                                        spatial=f.get("spatial"), keywords=f.get("keywords") or [],
                                        query_patterns=f.get("query_patterns") or [])
                      for f in data["facts"]}
        self.episode_hyperedges = {h["id"]: EpisodeHyperedge(id=h["id"], topic_id=h["topic_id"],
                                                             relation=h["relation"], weights=h["weights"])
                                   for h in data["episode_hyperedges"]}
        self.fact_hyperedges = {h["id"]: FactHyperedge(id=h["id"], episode_id=h["episode_id"],
                                                       relation=h["relation"], weights=h["weights"])
                                for h in data["fact_hyperedges"]}
        self.pages = {p["id"]: p for p in data["pages"]}

        page_date = {p["id"]: p["date"] for p in data["pages"]}
        self.episode_date = {e.id: _date_of(page_date.get(e.page_ids[0])) if e.page_ids else ""
                             for e in self.episodes.values()}
        self.episode_topic = {eid: he.topic_id for he in self.episode_hyperedges.values()
                              for eid in he.relation}

        # flat views, index-aligned
        self.topic_list = list(self.topics.values())
        self.episode_list = list(self.episodes.values())
        self.fact_list = list(self.facts.values())
        self.turns: list[str] = []
        self.turn_page: list[str] = []
        self.turn_episode: list[str] = []
        for p in data["pages"]:
            for t in p["turns"]:
                self.turns.append(t)
                self.turn_page.append(p["id"])
                self.turn_episode.append(p["episode_id"])
        self.turn_speaker = [m.group(1) if (m := _TURN_SPEAKER.match(t)) else "" for t in self.turns]
        self.speakers = sorted(set(self.turn_speaker) - {""})
        self.ep_index = {e.id: i for i, e in enumerate(self.episode_list)}
        self.topic_index = {t.id: i for i, t in enumerate(self.topic_list)}
        self.fact_episode_idx = np.array([self.ep_index.get(f.episode_id, -1) for f in self.fact_list])
        self.turn_episode_idx = np.array([self.ep_index.get(e, -1) for e in self.turn_episode])
        self.episode_topic_idx = np.array([self.topic_index.get(self.episode_topic.get(e.id), -1)
                                           for e in self.episode_list])

        topic_txt = [t.to_text() for t in self.topic_list]
        ep_txt = [e.to_text() for e in self.episode_list]
        fact_txt = [f.to_text() + "".join(" " + q for q in f.query_patterns) if self.fact_patterns else f.to_text()
                    for f in self.fact_list]
        T, E, F, U = (self._embed(x) for x in (topic_txt, ep_txt, fact_txt, self.turns))
        if self.retopic and len(self.episode_list) > self.retopic:
            topic_txt, T = self._retopic(E, ep_txt)
        for node, v in zip(self.topic_list, T):
            node.embedding = v
        for node, v in zip(self.episode_list, E):
            node.embedding = v
        for node, v in zip(self.fact_list, F):
            node.embedding = v
        # HyperMem's hyperedge propagation, on the dense side
        propagate_fact_embeddings(self.facts, self.fact_hyperedges, alpha=alpha)
        propagate_episode_embeddings(self.episodes, self.episode_hyperedges, alpha=alpha)
        self.T = self._stack(self.topic_list)
        self.E = self._stack(self.episode_list)
        self.F = self._stack(self.fact_list)
        self.U = U
        self.bm = {"topic": _BM25(topic_txt), "episode": _BM25(ep_txt),
                   "fact": _BM25(fact_txt), "turn": _BM25(self.turns)}
        self._build_entity_index()

    def _retopic(self, E: np.ndarray, ep_txt: list[str]) -> tuple[list[str], np.ndarray]:
        """Replace the topics (and their episode hyperedges) with balanced
        embedding clusters of the episodes. A topic's text is its members' text
        (for BM25) and its vector their mean."""
        from sklearn.cluster import AgglomerativeClustering
        n = len(self.episode_list)
        labels = AgglomerativeClustering(n_clusters=max(2, round(n / self.retopic)), linkage="ward").fit_predict(E)
        self.topics, self.episode_hyperedges, self.episode_topic = {}, {}, {}
        topic_txt, T = [], []
        for c in sorted(set(labels)):
            members = [i for i in range(n) if labels[i] == c]
            ids = [self.episode_list[i].id for i in members]
            topic = TopicNode(title=f"cluster {c}", summary=" ".join(ep_txt[i] for i in members), episode_ids=ids)
            he = EpisodeHyperedge(topic_id=topic.id, relation={i: "member" for i in ids},
                                  weights={i: 1.0 for i in ids})
            topic.episode_hyperedge_id = he.id
            self.topics[topic.id] = topic
            self.episode_hyperedges[he.id] = he
            self.episode_topic.update({i: topic.id for i in ids})
            topic_txt.append(topic.summary)
            v = E[members].mean(axis=0)
            T.append(v / max(np.linalg.norm(v), 1e-9))
        self.topic_list = list(self.topics.values())
        self.topic_index = {t.id: i for i, t in enumerate(self.topic_list)}
        self.episode_topic_idx = np.array([self.topic_index[self.episode_topic[e.id]] for e in self.episode_list])
        return topic_txt, np.stack(T).astype(np.float32)

    def _build_entity_index(self) -> None:
        """Entity hyperedges over facts: stemmed keyword tokens + capitalised names
        in the fact text, IDF-weighted so generic ones ("support", the two
        speakers' names) count for little and very common ones not at all."""
        import re
        ents: list[set[str]] = []
        for f in self.fact_list:
            e = set(_tokenize(" ".join(f.keywords)))
            e |= {"N:" + w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", f.content)}
            ents.append(e)
        n = max(1, len(ents))
        df: dict[str, int] = {}
        for e in ents:
            for x in e:
                df[x] = df.get(x, 0) + 1
        self.ent_idf = {x: float(np.log(n / c)) for x, c in df.items() if c <= 0.2 * n}
        self.fact_ents = [{x for x in e if x in self.ent_idf} for e in ents]

    # -- embedding helpers --------------------------------------------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self._encode_batch is not None:
            M = np.asarray(self._encode_batch(texts), dtype=np.float32)
        else:
            M = np.stack([np.asarray(self._encode1(t), dtype=np.float32) for t in texts])
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)

    @staticmethod
    def _stack(nodes: list) -> np.ndarray:
        if not nodes:
            return np.zeros((0, 1), dtype=np.float32)
        M = np.stack([n.embedding for n in nodes]).astype(np.float32)
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)

    def _hybrid(self, layer: str, M: np.ndarray, q: np.ndarray, question: str) -> np.ndarray:
        """RRF of dense and BM25 ranks -> fused rank (0 = best)."""
        if len(M) == 0:
            return np.zeros(0, dtype=np.int64)
        k = self.rrf_k
        fused = 1.0 / (k + _ranks(M @ q)) + 1.0 / (k + _ranks(self.bm[layer].scores(question)))
        rank = _ranks(fused)
        if self.layer_rerank and layer != "turn":
            nodes = {"topic": self.topic_list, "episode": self.episode_list, "fact": self.fact_list}[layer]
            pool = np.argsort(rank, kind="stable")[: self.layer_pool]
            ce = self._rerank_scores([nodes[int(i)].to_text() for i in pool])
            reranked = 1.0 / (k + np.arange(len(pool))) + self.w_ce / (k + _ranks(ce))
            rank[pool[np.argsort(-reranked, kind="stable")]] = np.arange(len(pool))
        return rank

    # -- retrieval ------------------------------------------------------------
    def retrieve(self, question: str, diag: dict | None = None) -> list[str]:
        q = _unit(np.asarray(self._encode_query(question), dtype=np.float32)) if self._encode_query \
            else self._embed([question])[0]
        self._query, self._ce_memo = question, {}
        if self.mode == "strict":
            evidence, info = self._retrieve_strict(q)
        else:
            evidence, info = self._retrieve_soft(q, question)
        if diag is not None:
            diag.update({"retriever": f"hypermem-{self.mode}", **info})
        return evidence

    def _fact_lines(self, fact_idx) -> list[str]:
        out = []
        for i in fact_idx[: self.n_facts]:
            f = self.fact_list[i]
            date = self.episode_date.get(f.episode_id, "")
            out.append(f"[{date}] (memory) {f.to_text()}" if date else f"(memory) {f.to_text()}")
        return out

    def _link_rank(self, seed_facts) -> np.ndarray:
        """Rank of every fact by IDF-weighted entity overlap with the seed facts."""
        seed_ents: set[str] = set()
        for i in seed_facts:
            seed_ents |= self.fact_ents[i]
        score = np.array([sum(self.ent_idf[x] for x in (fe & seed_ents)) for fe in self.fact_ents],
                         dtype=np.float32)
        r = _ranks(score)
        r[score <= 0] = 10 ** 6
        return r

    def _speaker_mask(self, question: str) -> np.ndarray | None:
        """Turns by the one speaker the question names; None if it names zero or several."""
        named = [s for s in self.speakers if re.search(rf"\b{re.escape(s)}\b", question, re.IGNORECASE)]
        return np.array([s == named[0] for s in self.turn_speaker]) if len(named) == 1 else None

    def _soft_pass(self, q: np.ndarray, question: str, link_rank: np.ndarray | None,
                   speaker_mask: np.ndarray | None = None):
        """One scoring pass. Returns (turn_score, ep_rank, r_fact, r_topic)."""
        k, w = self.rrf_k, self.w
        r_topic = self._hybrid("topic", self.T, q, question)
        r_ep = self._hybrid("episode", self.E, q, question)
        r_fact = self._hybrid("fact", self.F, q, question)
        r_turn = self._hybrid("turn", self.U, q, question)

        n_ep = len(self.episode_list)
        big = 10 ** 6
        fe = self.fact_episode_idx >= 0

        def best_per_episode(rank, idx, mask):
            out = np.full(n_ep, big)
            np.minimum.at(out, idx[mask], rank[mask])
            return out

        best_fact = best_per_episode(r_fact, self.fact_episode_idx, fe)
        best_turn = best_per_episode(r_turn, self.turn_episode_idx, self.turn_episode_idx >= 0)
        topic_rank = np.where(self.episode_topic_idx >= 0,
                              r_topic[np.maximum(self.episode_topic_idx, 0)] if len(r_topic) else big, big)

        ep_score = (w["episode"] / (k + r_ep) + w["topic"] / (k + topic_rank)
                    + w["fact"] / (k + best_fact) + w["eturn"] / (k + best_turn))
        if link_rank is not None:
            ep_score = ep_score + self.w_link / (k + best_per_episode(link_rank, self.fact_episode_idx, fe))
        ep_rank = _ranks(ep_score)

        turn_score = w["turn"] / (k + r_turn) + w["prior"] / (k + ep_rank[np.maximum(self.turn_episode_idx, 0)])
        if speaker_mask is not None:
            turn_score = turn_score + self.speaker_boost / k * speaker_mask
        return turn_score, ep_rank, r_fact, r_topic

    def _rerank_scores(self, texts: list[str]) -> np.ndarray:
        """Reranker scores against the current question, always the original one:
        each text is scored once per question, however many passes rank it."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._ce_memo]
        if todo:
            scores = _reranker(self.rerank_model).predict([(self._query, t) for t in todo],
                                                          batch_size=64, show_progress_bar=False)
            self._ce_memo.update(zip(todo, (float(x) for x in scores)))
        return np.array([self._ce_memo[t] for t in texts], dtype=np.float32)

    def _retrieve_soft(self, q: np.ndarray, question: str) -> tuple[list[str], dict]:
        k = self.rrf_k
        # round 1 (fact ranks seed the entity links and the feedback query)
        r_fact0 = self._hybrid("fact", self.F, q, question)
        seeds = np.argsort(r_fact0, kind="stable")
        link_rank = self._link_rank(seeds[: self.link_seeds]) if self.link else None
        speaker_mask = self._speaker_mask(question) if self.speaker_boost else None
        turn_score, ep_rank, r_fact, r_topic = self._soft_pass(q, question, link_rank, speaker_mask)

        if self.prf and len(self.fact_list):
            top = seeds[: self.prf_facts]
            q2 = q + self.prf_beta * self.F[top].mean(axis=0)
            q2 = q2 / max(np.linalg.norm(q2), 1e-9)
            facts_txt = " ".join(self.fact_list[i].content + " " + " ".join(self.fact_list[i].keywords) for i in top)
            ts2, _, _, _ = self._soft_pass(q2, question + " " + facts_txt, link_rank, speaker_mask)
            # fuse the two passes by rank
            turn_score = 1.0 / (k + _ranks(turn_score)) + self.w_prf / (k + _ranks(ts2))

        order = np.argsort(-turn_score, kind="stable")
        ce_scores = None
        if self.rerank:
            pool = [int(i) for i in order[: self.rerank_pool]]
            ce = self._rerank_scores([self.turns[i] for i in pool])
            fused = 1.0 / (k + np.arange(len(pool))) + self.w_ce / (k + _ranks(ce))
            perm = np.argsort(-fused, kind="stable")
            order = [pool[j] for j in perm]
            ce_scores = {pool[j]: float(ce[j]) for j in range(len(pool))}

        chosen: list[int] = []
        seen: set[int] = set()
        per_ep: dict[int, int] = {}
        cutoff = float(self.ce_cutoff) if (self.ce_cutoff not in (None, "") and ce_scores is not None) else None
        for i in order:
            i = int(i)
            if len(chosen) >= self.max_turns:
                break
            if cutoff is not None and len(chosen) >= self.min_turns and ce_scores.get(i, -1e9) < cutoff:
                continue
            e = int(self.turn_episode_idx[i])
            if self.per_episode_cap and per_ep.get(e, 0) >= self.per_episode_cap:
                continue
            per_ep[e] = per_ep.get(e, 0) + 1
            group = [i]
            if self.page_expand:
                pid = self.turn_page[i]
                group = [j for j in range(max(0, i - 1), min(len(self.turns), i + 2)) if self.turn_page[j] == pid]
            for j in group:
                if j not in seen and len(chosen) < self.max_turns:
                    seen.add(j)
                    chosen.append(j)

        if self.turn_order == "chrono":
            chosen = sorted(chosen)
        fact_order = np.argsort(r_fact, kind="stable")
        fact_lines, turn_lines = self._fact_lines(fact_order), [self.turns[i] for i in chosen]
        facts_last = self.facts_pos == "last"
        evidence = turn_lines + fact_lines if facts_last else fact_lines + turn_lines
        info = {
            "topics": [self.topic_list[i].title for i in np.argsort(r_topic)[:3]],
            "episode_ids": [self.episode_list[i].id for i in np.argsort(ep_rank)[:8]],
            "facts": [self.fact_list[i].content for i in fact_order[:10]],
            "evidence_ids": ([self.turn_page[i] for i in chosen] + ["fact"] * len(fact_lines) if facts_last
                             else ["fact"] * len(fact_lines) + [self.turn_page[i] for i in chosen]),
            "steps": {"prf": self.prf, "link": self.link, "rerank": self.rerank, "layer_rerank": self.layer_rerank,
                      "ce_cutoff": self.ce_cutoff,
                      "speaker_boost": self.speaker_boost},
        }
        return evidence, info

    def _retrieve_strict(self, q: np.ndarray) -> tuple[list[str], dict]:
        def top(M, idx, n):
            idx = list(idx)
            return [idx[j] for j in np.argsort(-(M[idx] @ q), kind="stable")[:n]] if idx else []

        topics = top(self.T, range(len(self.topic_list)), self.topic_top_k)
        sel_t = set(topics)
        cand_e = [i for i in range(len(self.episode_list))
                  if self.episode_topic_idx[i] in sel_t or self.episode_topic_idx[i] < 0]
        eps = top(self.E, cand_e, self.episode_top_k)
        sel_e = set(eps)
        cand_f = [i for i in range(len(self.fact_list)) if self.fact_episode_idx[i] in sel_e]
        facts = top(self.F, cand_f, self.fact_top_k)

        page_ids: list[str] = []
        for ei in [self.fact_episode_idx[i] for i in facts] + eps:
            for pid in self.episode_list[ei].page_ids:
                if pid not in page_ids and pid in self.pages:
                    page_ids.append(pid)
        turns: list[str] = []
        used: list[str] = []
        for pid in page_ids:
            if len(turns) >= self.max_turns:
                break
            turns.extend(self.pages[pid]["turns"])
            used.extend([pid] * len(self.pages[pid]["turns"]))
        turns, used = turns[: self.max_turns], used[: self.max_turns]
        evidence = self._fact_lines(facts) + turns
        info = {
            "topics": [self.topic_list[i].title for i in topics],
            "episode_ids": [self.episode_list[i].id for i in eps],
            "facts": [self.fact_list[i].content for i in facts[:10]],
            "evidence_ids": ["fact"] * min(self.n_facts, len(facts)) + used,
        }
        return evidence, info


def _date_of(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        from iterret.time_resolution import date_only
        return date_only(ts) or ts
    except Exception:  # noqa: BLE001
        return ts


def load_segmenter(
    model_name: str = os.environ.get("HYPERMEM_SEG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    device: str = os.environ.get("HYPERMEM_SEG_DEVICE", "cuda:0"),
) -> tuple[Any, Any]:
    """The small causal LM whose surprise segments pages into episodes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval(), tok


def load_or_build(
    cache_path: str,
    conv_block: dict | None,
    llm_base_url: str,
    llm_model: str,
    *,
    passages: list[str] | None = None,
    segmenter: tuple[Any, Any] | None = None,
    seg_model_name: str = os.environ.get("HYPERMEM_SEG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    seg_device: str = os.environ.get("HYPERMEM_SEG_DEVICE", "cuda:0"),
    gamma: float = float(os.environ.get("HYPERMEM_GAMMA", "0.5")),
    topics_mode: str = os.environ.get("HYPERMEM_BUILD_TOPICS", "llm"),
    workers: int = int(os.environ.get("HYPERMEM_BUILD_WORKERS", "8")),
    log: Callable[[str], None] = print,
) -> dict:
    """Cached hypergraph of a LoCoMo conversation (``conv_block``) or, when
    ``passages`` is given, of a document split into those passages.
    ``segmenter``: an already loaded (model, tokenizer) to reuse across calls.
    ``topics_mode``: "llm" (streamed LLM topics) or "none" (skip them; the
    retriever clusters episodes instead). ``workers``: concurrent episodes
    during LLM extraction."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        if data.get("version") == BUILD_VERSION:
            log(f"[hypermem] loaded cached hypergraph {cache_path}")
            return data

    from vllm_llm_client import VLLMClient

    log(f"[hypermem] building hypergraph -> {cache_path} (segmenter {seg_model_name})")
    llm = VLLMClient(base_url=llm_base_url, model=llm_model)
    # The extraction functions fall back to raw text when an LLM call fails, so
    # without a server the build "succeeds" with no real summaries or facts and
    # caches that. Refuse to start instead.
    try:
        llm.chat("Reply with OK.", max_new_tokens=4)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"HyperMem build needs the extraction LLM at {llm_base_url} ({llm_model}), "
                           f"which did not answer: {exc}") from exc
    seg, tok = segmenter or load_segmenter(seg_model_name, seg_device)
    try:
        if passages is not None:
            data = build_document_hypergraph(passages, llm, seg, tok, gamma=gamma,
                                             topics_mode=topics_mode, workers=workers, log=log)
        else:
            data = build_conversation_hypergraph(_conv_or_self(conv_block), llm, seg, tok, gamma=gamma,
                                                 topics_mode=topics_mode, workers=workers, log=log)
    finally:
        if segmenter is None:
            import torch
            del seg
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, cache_path)
    return data


def _conv_or_self(block: dict) -> dict:
    return block.get("conversation", block)
