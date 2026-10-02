"""
Hypergraph embedding propagation.

Numpy port of HyperMem's attention-weighted hyperedge embedding + node update
formula (`stage3_hypergraph_index.py`):

    edge_embedding  = sum_i softmax(hyperedge.weights)[i] * member_embedding[i]
    node.embedding' = node.embedding + alpha * edge_embedding      # alpha = 0.5

Applied once per fact hyperedge (updates that episode's fact embeddings) and
once per episode hyperedge (updates that topic's episode embeddings). Topic
embeddings are not propagated -- topics are the top layer, embedded directly
from LLM-written title+summary text, same as HyperMem (no hyperedge sits
above them).
"""

from __future__ import annotations

import numpy as np

from hypergraph_types import EpisodeHyperedge, EpisodeNode, FactHyperedge, FactNode

DEFAULT_ALPHA = 0.5


def _softmax(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()


def _hyperedge_embedding(weights: dict[str, float], member_embeddings: dict[str, np.ndarray]) -> np.ndarray | None:
    ids = [mid for mid in weights.keys() if mid in member_embeddings and member_embeddings[mid] is not None]
    if not ids:
        return None
    softmax_weights = _softmax([weights[mid] for mid in ids])
    edge_emb = np.zeros_like(member_embeddings[ids[0]], dtype=np.float64)
    for mid, w in zip(ids, softmax_weights):
        edge_emb += w * member_embeddings[mid].astype(np.float64)
    return edge_emb


def propagate_fact_embeddings(
    facts: dict[str, FactNode],
    fact_hyperedges: dict[str, FactHyperedge],
    alpha: float = DEFAULT_ALPHA,
) -> None:
    """Update fact.embedding in place via its episode's FactHyperedge."""
    member_embeddings = {fid: f.embedding for fid, f in facts.items()}
    for hyperedge in fact_hyperedges.values():
        edge_emb = _hyperedge_embedding(hyperedge.weights, member_embeddings)
        if edge_emb is None:
            continue
        hyperedge.embedding = edge_emb.astype(np.float32)
        for fact_id in hyperedge.relation:
            fact = facts.get(fact_id)
            if fact is None or fact.embedding is None:
                continue
            fact.embedding = (fact.embedding.astype(np.float64) + alpha * edge_emb).astype(np.float32)


def propagate_episode_embeddings(
    episodes: dict[str, EpisodeNode],
    episode_hyperedges: dict[str, EpisodeHyperedge],
    alpha: float = DEFAULT_ALPHA,
) -> None:
    """Update episode.embedding in place via its topic's EpisodeHyperedge."""
    member_embeddings = {eid: e.embedding for eid, e in episodes.items()}
    for hyperedge in episode_hyperedges.values():
        edge_emb = _hyperedge_embedding(hyperedge.weights, member_embeddings)
        if edge_emb is None:
            continue
        hyperedge.embedding = edge_emb.astype(np.float32)
        for episode_id in hyperedge.relation:
            episode = episodes.get(episode_id)
            if episode is None or episode.embedding is None:
                continue
            episode.embedding = (episode.embedding.astype(np.float64) + alpha * edge_emb).astype(np.float32)
