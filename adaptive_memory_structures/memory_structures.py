"""
FluxMem Memory Structures
Implements linear, hierarchical, and hypergraph memory organization for MTEM episodic units.
Based on: "Choosing How to Remember: Adaptive Memory Structures for LLM Agents"
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Core data primitives
# ---------------------------------------------------------------------------

@dataclass
class Page:
    """A single user-agent exchange (one turn)."""
    page_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_text: str = ""
    agent_text: str = ""
    timestamp: float = field(default_factory=time.time)
    embedding: np.ndarray | None = None          # dense vector, set externally
    last_access: float = field(default_factory=time.time)

    def touch(self):
        self.last_access = time.time()

    def to_text(self) -> str:
        return f"User: {self.user_text}\nAgent: {self.agent_text}"


@dataclass
class EpisodicSession:
    """
    One episodic memory unit inside MTEM.
    Holds a group of semantically / temporally related Pages plus
    whichever indexing structure was selected for this unit.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pages: list[Page] = field(default_factory=list)
    summary: str = ""
    summary_embedding: np.ndarray | None = None
    structure_type: str = "linear"              # "linear" | "hierarchical" | "hypergraph"
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    # structure-specific indices (populated by MemoryStructure classes)
    topic_tree: dict[str, Any] = field(default_factory=dict)    # for hierarchical memory
    hypergraph: dict[str, Any] = field(default_factory=dict)    # for hypergraph memory

    # utility tracking
    access_count: int = 0
    interaction_intensity: float = 0.0

    def touch(self):
        self.last_access = time.time()
        self.access_count += 1

    def utility_score(
        self,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.3,
        now: float | None = None,
    ) -> float:
        """
        U(s) = w1*c(s) + w2*l(s) + w3*d(s)
        c = access frequency (normalised to [0,1] with log), l = intensity, d = recency
        """
        now = now or time.time()
        c = min(1.0, np.log1p(self.access_count) / 10.0)
        l_ = min(1.0, self.interaction_intensity)
        age = now - self.last_access
        d = np.exp(-age / (3600 * 24))             # decays over ~1 day
        return w1 * c + w2 * l_ + w3 * d

    def to_text(self) -> str:
        parts = [self.summary] if self.summary else []
        for p in self.pages:
            parts.append(p.to_text())
        return "\n---\n".join(parts)


@dataclass
class LTSMEntry:
    """One consolidated entry in Long-Term Semantic Memory."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    entry_type: str = "fact"                # "fact" | "profile" | "procedural"
    embedding: np.ndarray | None = None
    usage_count: int = 0
    last_access: float = field(default_factory=time.time)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)

    def touch(self):
        self.last_access = time.time()
        self.usage_count += 1

    def is_eligible(
        self,
        tau_u: float = 1,
        tau_r: float = 0.01,
        tau_c: float = 0.0,
        now: float | None = None,
    ) -> bool:
        """Eq. 6 – keep entry iff all thresholds are met."""
        now = now or time.time()
        age = now - self.last_access
        recency = np.exp(-age / (3600 * 24 * 7))   # weekly decay
        return (
            self.usage_count >= tau_u
            and recency >= tau_r
            and (tau_c == 0.0 or self.confidence >= tau_c)
        )


# ---------------------------------------------------------------------------
# Memory structure implementations
# ---------------------------------------------------------------------------

class LinearMemory:
    """
    Chronological sequence of pages.
    Retrieval: cosine similarity + implicit recency weight.
    """

    def retrieve(
        self,
        session: EpisodicSession,
        query_emb: np.ndarray,
        top_k: int = 3,
    ) -> list[Page]:
        if not session.pages:
            return []

        scored: list[tuple[float, int, Page]] = []
        n = len(session.pages)
        for i, page in enumerate(session.pages):
            if page.embedding is None:
                sim = 0.0
            else:
                sim = float(_cosine(query_emb, page.embedding))
            recency_weight = (i + 1) / n           # later pages score higher
            score = 0.7 * sim + 0.3 * recency_weight
            scored.append((score, i, page))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, _, p in scored[:top_k]]


class HypergraphMemory:
    """
    3-layer hypergraph (topic -> episode -> fact) over pages, modeled on the
    HyperMem paper. Episodes come from surprise-based segmentation
    (surprise_episode_segmenter.PageEpisodeSegmenter); topics are LLM-clustered
    over episodes via streaming match, one session at a time
    (hypergraph_extraction.build_topics_for_session); facts are LLM-extracted
    per episode. Retrieval is coarse-to-fine -- topic -> episode -> fact --
    with hyperedges acting as a hard connectivity filter between layers, over
    embeddings already updated once via HyperMem's attention-weighted
    hyperedge propagation formula (hypergraph_embedding.py).

    Heavy imports (torch, the LLM/embedding clients) are deferred into the
    methods below so importing this module never requires a loaded model.
    """

    def __init__(
        self,
        topic_top_k: int = 2,
        episode_top_k: int = 3,
        fact_top_k: int = 5,
        alpha: float = 0.5,
        topic_match_batch_size: int = 10,
        gamma: float = 1.5,
        n_local: int = 4096,
        n_init: int = 128,
        min_block_size: int = 8,
        similarity_refinement: bool = True,
        embedder: Any = None,
        llm_client: Any = None,
    ):
        self.topic_top_k = topic_top_k
        self.episode_top_k = episode_top_k
        self.fact_top_k = fact_top_k
        self.alpha = alpha
        self.topic_match_batch_size = topic_match_batch_size
        self._segmenter_kwargs = dict(
            gamma=gamma,
            n_local=n_local,
            n_init=n_init,
            min_block_size=min_block_size,
            similarity_refinement=similarity_refinement,
        )
        self._embedder = embedder
        self._llm_client = llm_client

    def _resolve_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        from qwen_client import get_client
        self._embedder = get_client()
        return self._embedder

    def _resolve_llm_client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        from vllm_llm_client import VLLMClient, vllm_available
        if vllm_available():
            self._llm_client = VLLMClient()
        else:
            from qwen_client import get_client
            self._llm_client = get_client()
        return self._llm_client

    def build_index(self, session: EpisodicSession) -> None:
        """(Re)build the session's hypergraph, stored in session.hypergraph."""
        from surprise_episode_segmenter import PageEpisodeSegmenter
        from hypergraph_extraction import (
            summarize_episode,
            extract_facts_for_episode,
            assign_fact_roles,
            build_topics_for_session,
        )
        from hypergraph_embedding import propagate_fact_embeddings, propagate_episode_embeddings
        from hypergraph_types import EpisodeNode

        pages = session.pages
        if not pages:
            session.hypergraph = {}
            return

        embedder = self._resolve_embedder()
        llm_client = self._resolve_llm_client()

        segmenter = PageEpisodeSegmenter(**self._segmenter_kwargs)
        model = getattr(embedder, "model", None)
        tokenizer = getattr(embedder, "tokenizer", None)
        page_groups = segmenter.segment(pages, model, tokenizer)

        episodes: dict[str, EpisodeNode] = {}
        facts: dict[str, Any] = {}
        fact_hyperedges: dict[str, Any] = {}
        page_to_episode: dict[str, str] = {}
        episode_order: list[EpisodeNode] = []

        for group in page_groups:
            subject, summary = summarize_episode(group, llm_client)
            episode = EpisodeNode(page_ids=[p.page_id for p in group], subject=subject, summary=summary)
            episode.embedding = embedder.embed(episode.to_text())
            episodes[episode.id] = episode
            episode_order.append(episode)
            for p in group:
                page_to_episode[p.page_id] = episode.id

            episode_facts = extract_facts_for_episode(group, episode.id, llm_client)
            for f in episode_facts:
                f.embedding = embedder.embed(f.to_text())
                facts[f.id] = f
            fact_hyperedge = assign_fact_roles(episode_facts, episode.id, summary, llm_client)
            fact_hyperedges[fact_hyperedge.id] = fact_hyperedge

        topics, episode_hyperedges = build_topics_for_session(
            episode_order, llm_client, batch_size=self.topic_match_batch_size,
        )
        for topic in topics.values():
            topic.embedding = embedder.embed(topic.to_text())

        propagate_fact_embeddings(facts, fact_hyperedges, alpha=self.alpha)
        propagate_episode_embeddings(episodes, episode_hyperedges, alpha=self.alpha)

        session.hypergraph = {
            "topics": topics,
            "episode_hyperedges": episode_hyperedges,
            "episodes": episodes,
            "fact_hyperedges": fact_hyperedges,
            "facts": facts,
            "page_to_episode": page_to_episode,
        }

    def retrieve(
        self,
        session: EpisodicSession,
        query_emb: np.ndarray,
        top_k: int = 3,
    ) -> list[Page]:
        if not session.pages:
            return []

        if not session.hypergraph:
            self.build_index(session)

        hg = session.hypergraph
        topics = hg.get("topics", {})
        episodes = hg.get("episodes", {})
        facts = hg.get("facts", {})
        episode_hyperedges = hg.get("episode_hyperedges", {})
        fact_hyperedges = hg.get("fact_hyperedges", {})
        id_to_page = {p.page_id: p for p in session.pages}

        if not topics:
            return session.pages[:top_k]

        # layer 1: topics
        topic_scored = sorted(
            topics.values(),
            key=lambda t: float(_cosine(query_emb, t.embedding)) if t.embedding is not None else 0.0,
            reverse=True,
        )
        selected_topics = {t.id for t in topic_scored[: self.topic_top_k]}

        # layer 2: episodes, hard-filtered to selected topics via episode hyperedges
        connected_episode_ids: set[str] = set()
        for he in episode_hyperedges.values():
            if he.topic_id in selected_topics:
                connected_episode_ids.update(he.relation.keys())
        candidate_episodes = [episodes[eid] for eid in connected_episode_ids if eid in episodes]
        episode_scored = sorted(
            candidate_episodes,
            key=lambda e: float(_cosine(query_emb, e.embedding)) if e.embedding is not None else 0.0,
            reverse=True,
        )
        selected_episodes = {e.id for e in episode_scored[: self.episode_top_k]}

        # layer 3: facts, hard-filtered to selected episodes via fact hyperedges
        connected_fact_ids: set[str] = set()
        for he in fact_hyperedges.values():
            if he.episode_id in selected_episodes:
                connected_fact_ids.update(he.relation.keys())
        candidate_facts = [facts[fid] for fid in connected_fact_ids if fid in facts]
        fact_scored = sorted(
            candidate_facts,
            key=lambda f: float(_cosine(query_emb, f.embedding)) if f.embedding is not None else 0.0,
            reverse=True,
        )

        # map facts -> pages, preserving score order, deduplicating
        result: list[Page] = []
        seen: set[str] = set()
        for f in fact_scored:
            episode = episodes.get(f.episode_id)
            if episode is None:
                continue
            for pid in episode.page_ids:
                if pid in seen or pid not in id_to_page:
                    continue
                seen.add(pid)
                result.append(id_to_page[pid])
                if len(result) >= top_k:
                    return result

        # backfill from next-best episodes if still short of top_k
        if len(result) < top_k:
            for e in episode_scored:
                for pid in e.page_ids:
                    if pid in seen or pid not in id_to_page:
                        continue
                    seen.add(pid)
                    result.append(id_to_page[pid])
                    if len(result) >= top_k:
                        return result

        return result[:top_k]


class HierarchicalMemory:
    """
    Topic tree over pages.
    Retrieval: coarse-to-fine DFS – match topic cluster, then pages within.
    """

    def build_index(self, session: EpisodicSession) -> None:
        """
        Lightweight clustering: group pages by rough embedding similarity
        using a greedy single-pass approach (no external dependencies).
        Stores cluster centres + member page ids in session.topic_tree.
        """
        pages = session.pages
        if not pages:
            session.topic_tree = {"clusters": []}
            return

        clusters: list[dict[str, Any]] = []
        threshold = 0.5

        for page in pages:
            if page.embedding is None:
                continue
            placed = False
            for cluster in clusters:
                centre = cluster["centre"]
                if float(_cosine(page.embedding, centre)) >= threshold:
                    cluster["members"].append(page.page_id)
                    # update running mean
                    n = len(cluster["members"])
                    cluster["centre"] = (centre * (n - 1) + page.embedding) / n
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "centre": page.embedding.copy(),
                    "members": [page.page_id],
                })

        session.topic_tree = {
            "clusters": clusters,
            "id_to_idx": {p.page_id: idx for idx, p in enumerate(pages)},
        }

    def retrieve(
        self,
        session: EpisodicSession,
        query_emb: np.ndarray,
        top_k: int = 3,
    ) -> list[Page]:
        if not session.pages:
            return []

        if not session.topic_tree:
            self.build_index(session)

        pages = session.pages
        id_to_idx = session.topic_tree.get("id_to_idx", {})
        clusters = session.topic_tree.get("clusters", [])

        if not clusters:
            return pages[:top_k]

        # coarse step: rank clusters by centre similarity
        cluster_scores = [
            float(_cosine(query_emb, c["centre"])) if c["centre"] is not None else 0.0
            for c in clusters
        ]
        best_cluster_idx = int(np.argmax(cluster_scores))
        best_cluster = clusters[best_cluster_idx]

        # fine step: rank pages within best cluster
        member_pages = [
            pages[id_to_idx[pid]]
            for pid in best_cluster["members"]
            if pid in id_to_idx
        ]
        scored = []
        for p in member_pages:
            s = float(_cosine(query_emb, p.embedding)) if p.embedding is not None else 0.0
            scored.append((s, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        result = [p for _, p in scored[:top_k]]

        # if we need more, pull from other clusters (DFS)
        if len(result) < top_k:
            remaining_order = sorted(
                range(len(clusters)),
                key=lambda i: cluster_scores[i],
                reverse=True,
            )
            for ci in remaining_order:
                if ci == best_cluster_idx:
                    continue
                for pid in clusters[ci]["members"]:
                    if len(result) >= top_k:
                        break
                    if pid in id_to_idx:
                        result.append(pages[id_to_idx[pid]])

        return result[:top_k]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity."""
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
