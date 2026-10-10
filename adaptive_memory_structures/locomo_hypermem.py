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
  - embeddings come from the caller's sentence encoder (the eval passes
    MiniLM, the encoder IterRet uses), not mean-pooled causal-LM states,
    which retrieved poorly in testing
  - top-k's are sized for a whole conversation, not one session
  - output is the selected pages' individual turns, rendered
    "[date] Speaker: text" like IterRet's evidence, so everything downstream
    (relevance filter, OSAM write, answer prompt) is unchanged
"""

from __future__ import annotations

import json
import os
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


def build_conversation_hypergraph(
    conv_block: dict,
    llm_client: Any,
    seg_model: Any = None,
    seg_tokenizer: Any = None,
    *,
    gamma: float = 0.5,
    min_block_size: int = 1,
    topic_match_batch_size: int = 10,
    log: Callable[[str], None] = print,
) -> dict:
    """Build the whole-conversation hypergraph. Returns a JSON-serialisable dict."""
    from hypergraph_extraction import (
        assign_fact_roles,
        build_topics_for_session,
        extract_facts_for_episode,
        summarize_episode,
    )
    from surprise_episode_segmenter import PageEpisodeSegmenter

    segmenter = PageEpisodeSegmenter(gamma=gamma, min_block_size=min_block_size)
    t0 = time.time()

    pages_out: list[dict] = []
    episodes: list[EpisodeNode] = []
    episode_session: dict[str, int] = {}
    facts: dict[str, FactNode] = {}
    fact_hyperedges: dict[str, FactHyperedge] = {}

    for snum in _session_nums(conv_block):
        turns = conv_block.get(f"session_{snum}") or []
        if not turns:
            continue
        session_ts = conv_block.get(f"session_{snum}_date_time", f"Session {snum}")
        paged = _pages_for_session(turns, session_ts)
        rendered_by_id = {p.page_id: r for p, r in paged}
        pages = [p for p, _ in paged]

        groups = segmenter.segment(pages, seg_model, seg_tokenizer)
        for group in groups:
            subject, summary = summarize_episode(group, llm_client)
            episode = EpisodeNode(page_ids=[p.page_id for p in group], subject=subject, summary=summary)
            episodes.append(episode)
            episode_session[episode.id] = snum
            ep_facts = extract_facts_for_episode(group, episode.id, llm_client)
            facts.update((f.id, f) for f in ep_facts)
            he = assign_fact_roles(ep_facts, episode.id, summary, llm_client)
            fact_hyperedges[he.id] = he
            for p in group:
                pages_out.append({
                    "id": p.page_id, "session": snum, "date": session_ts,
                    "turns": rendered_by_id[p.page_id], "episode_id": episode.id,
                })
        log(f"[hypermem] session {snum}: {len(pages)} pages -> {len(groups)} episodes "
            f"({len(episodes)} episodes / {len(facts)} facts so far, {time.time() - t0:.0f}s)")

    topics, episode_hyperedges = build_topics_for_session(
        episodes, llm_client, batch_size=topic_match_batch_size,
    )
    log(f"[hypermem] built: {len(topics)} topics, {len(episodes)} episodes, "
        f"{len(facts)} facts in {time.time() - t0:.0f}s")

    return {
        "version": BUILD_VERSION,
        "params": {"gamma": gamma, "min_block_size": min_block_size},
        "topics": [{"id": t.id, "title": t.title, "summary": t.summary, "episode_ids": t.episode_ids}
                   for t in topics.values()],
        "episodes": [{"id": e.id, "subject": e.subject, "summary": e.summary, "page_ids": e.page_ids,
                      "session": episode_session[e.id]} for e in episodes],
        "facts": [{"id": f.id, "episode_id": f.episode_id, "content": f.content,
                   "confidence": f.confidence, "temporal": f.temporal, "spatial": f.spatial,
                   "keywords": f.keywords} for f in facts.values()],
        "episode_hyperedges": [{"id": h.id, "topic_id": h.topic_id, "relation": h.relation,
                                "weights": h.weights} for h in episode_hyperedges.values()],
        "fact_hyperedges": [{"id": h.id, "episode_id": h.episode_id, "relation": h.relation,
                             "weights": h.weights} for h in fact_hyperedges.values()],
        "pages": pages_out,
    }


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
    try:
        from nltk.stem import PorterStemmer
        stem = _tokenize._stem = getattr(_tokenize, "_stem", None) or PorterStemmer().stem
    except Exception:  # noqa: BLE001
        stem = lambda w: w  # noqa: E731
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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


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
        relevant turns rather than all of them. The top facts, dated, can be
        prepended as evidence too (HYPERMEM_FACTS).
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
        self.n_facts = n_facts if n_facts is not None else _env_int("HYPERMEM_FACTS", 0)
        # strict-mode knobs
        self.topic_top_k = _env_int("HYPERMEM_TOPIC_TOP_K", 3)
        self.episode_top_k = _env_int("HYPERMEM_EPISODE_TOP_K", 8)
        self.fact_top_k = _env_int("HYPERMEM_FACT_TOP_K", 20)
        # soft-mode weights (RRF, k = 20: tuned on LoCoMo convs 1-3 gold-evidence recall)
        self.rrf_k = _env_float("HYPERMEM_RRF_K", 20)
        self.w = {name: _env_float(f"HYPERMEM_W_{name.upper()}", dflt) for name, dflt in
                  [("episode", 1.0), ("topic", 0.5), ("fact", 1.0), ("eturn", 1.0), ("turn", 1.0), ("prior", 1.0)]}
        self.page_expand = _env_int("HYPERMEM_PAGE_EXPAND", 0)
        # max turns taken from any one episode (0 = no cap); spreads the budget
        # across episodes, which multi-hop questions need
        self.per_episode_cap = _env_int("HYPERMEM_PER_EPISODE_CAP", 0)

        self._encode1 = encode
        self._encode_batch = encode_batch

        self.topics = {t["id"]: TopicNode(id=t["id"], title=t["title"], summary=t["summary"],
                                          episode_ids=t["episode_ids"]) for t in data["topics"]}
        self.episodes = {e["id"]: EpisodeNode(id=e["id"], page_ids=e["page_ids"], subject=e["subject"],
                                              summary=e["summary"]) for e in data["episodes"]}
        self.facts = {f["id"]: FactNode(id=f["id"], episode_id=f["episode_id"], content=f["content"],
                                        confidence=f["confidence"], temporal=f.get("temporal"),
                                        spatial=f.get("spatial"), keywords=f.get("keywords") or [])
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
        self.ep_index = {e.id: i for i, e in enumerate(self.episode_list)}
        self.topic_index = {t.id: i for i, t in enumerate(self.topic_list)}
        self.fact_episode_idx = np.array([self.ep_index.get(f.episode_id, -1) for f in self.fact_list])
        self.turn_episode_idx = np.array([self.ep_index.get(e, -1) for e in self.turn_episode])
        self.episode_topic_idx = np.array([self.topic_index.get(self.episode_topic.get(e.id), -1)
                                           for e in self.episode_list])

        topic_txt = [t.to_text() for t in self.topic_list]
        ep_txt = [e.to_text() for e in self.episode_list]
        fact_txt = [f.to_text() for f in self.fact_list]
        T, E, F, U = (self._embed(x) for x in (topic_txt, ep_txt, fact_txt, self.turns))
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
        return _ranks(fused)

    # -- retrieval ------------------------------------------------------------
    def retrieve(self, question: str, diag: dict | None = None) -> list[str]:
        q = self._embed([question])[0]
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

    def _retrieve_soft(self, q: np.ndarray, question: str) -> tuple[list[str], dict]:
        k, w = self.rrf_k, self.w
        r_topic = self._hybrid("topic", self.T, q, question)
        r_ep = self._hybrid("episode", self.E, q, question)
        r_fact = self._hybrid("fact", self.F, q, question)
        r_turn = self._hybrid("turn", self.U, q, question)

        n_ep = len(self.episode_list)
        big = 10 ** 6
        best_fact = np.full(n_ep, big)
        np.minimum.at(best_fact, self.fact_episode_idx[self.fact_episode_idx >= 0],
                      r_fact[self.fact_episode_idx >= 0])
        best_turn = np.full(n_ep, big)
        np.minimum.at(best_turn, self.turn_episode_idx[self.turn_episode_idx >= 0],
                      r_turn[self.turn_episode_idx >= 0])
        topic_rank = np.where(self.episode_topic_idx >= 0,
                              r_topic[np.maximum(self.episode_topic_idx, 0)] if len(r_topic) else big, big)

        ep_score = (w["episode"] / (k + r_ep) + w["topic"] / (k + topic_rank)
                    + w["fact"] / (k + best_fact) + w["eturn"] / (k + best_turn))
        ep_rank = _ranks(ep_score)

        turn_score = w["turn"] / (k + r_turn) + w["prior"] / (k + ep_rank[np.maximum(self.turn_episode_idx, 0)])
        order = np.argsort(-turn_score, kind="stable")

        chosen: list[int] = []
        seen: set[int] = set()
        per_ep: dict[int, int] = {}
        for i in order:
            if len(chosen) >= self.max_turns:
                break
            e = int(self.turn_episode_idx[i])
            if self.per_episode_cap and per_ep.get(e, 0) >= self.per_episode_cap:
                continue
            per_ep[e] = per_ep.get(e, 0) + 1
            group = [int(i)]
            if self.page_expand:
                pid = self.turn_page[i]
                group = [j for j in range(max(0, i - 1), min(len(self.turns), i + 2)) if self.turn_page[j] == pid]
            for j in group:
                if j not in seen and len(chosen) < self.max_turns:
                    seen.add(j)
                    chosen.append(j)

        fact_order = np.argsort(r_fact, kind="stable")
        evidence = self._fact_lines(fact_order) + [self.turns[i] for i in chosen]
        info = {
            "topics": [self.topic_list[i].title for i in np.argsort(r_topic)[:3]],
            "episode_ids": [self.episode_list[i].id for i in np.argsort(ep_rank)[:8]],
            "facts": [self.fact_list[i].content for i in fact_order[:10]],
            "evidence_ids": ["fact"] * min(self.n_facts, len(self.fact_list)) + [self.turn_page[i] for i in chosen],
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


def load_or_build(
    cache_path: str,
    conv_block: dict,
    llm_base_url: str,
    llm_model: str,
    *,
    seg_model_name: str = os.environ.get("HYPERMEM_SEG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    seg_device: str = os.environ.get("HYPERMEM_SEG_DEVICE", "cuda:0"),
    gamma: float = float(os.environ.get("HYPERMEM_GAMMA", "0.5")),
    log: Callable[[str], None] = print,
) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        if data.get("version") == BUILD_VERSION:
            log(f"[hypermem] loaded cached hypergraph {cache_path}")
            return data

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm_llm_client import VLLMClient

    log(f"[hypermem] building hypergraph -> {cache_path} (segmenter {seg_model_name})")
    tok = AutoTokenizer.from_pretrained(seg_model_name)
    dtype = torch.float32 if seg_device == "cpu" else torch.bfloat16
    seg = AutoModelForCausalLM.from_pretrained(seg_model_name, torch_dtype=dtype).to(seg_device).eval()
    llm = VLLMClient(base_url=llm_base_url, model=llm_model)
    try:
        data = build_conversation_hypergraph(_conv_or_self(conv_block), llm, seg, tok, gamma=gamma, log=log)
    finally:
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
