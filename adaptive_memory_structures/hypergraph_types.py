"""
HypergraphMemory node/hyperedge types.

Plain @dataclass, matching this directory's existing style (Page,
EpisodicSession, LTSMEntry in memory_structures.py are all @dataclass, not
pydantic, even though pydantic is pinned elsewhere in the repo).

Mirrors HyperMem's real 3-layer shape (topic -> episode -> fact), scoped to
del-mem's per-session unit:
  - a session's pages are segmented into episodes (surprise mechanism)
  - episodes are LLM-clustered into topics (one EpisodeHyperedge per topic)
  - facts are LLM-extracted per episode (one FactHyperedge per episode)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FactRole(str, Enum):
    CORE = "core"
    CONTEXT = "context"
    DETAIL = "detail"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


class EpisodeRole(str, Enum):
    INITIATING = "initiating"
    DEVELOPING = "developing"
    CLIMAX = "climax"
    CONCLUDING = "concluding"
    RECURRING = "recurring"
    BACKGROUND = "background"
    KEY_MOMENT = "key_moment"
    TRANSITION = "transition"


@dataclass
class FactNode:
    """L1 layer: an atomic, queryable fact extracted from one episode."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str = ""
    content: str = ""
    confidence: float = 0.8
    temporal: str | None = None
    spatial: str | None = None
    keywords: list[str] = field(default_factory=list)
    query_patterns: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    hyperedge: dict[str, str] = field(default_factory=dict)  # hyperedge_id -> role

    def to_text(self) -> str:
        parts = [self.content]
        if self.temporal:
            parts.append(f"Time: {self.temporal}")
        if self.spatial:
            parts.append(f"Location: {self.spatial}")
        if len(parts) > 1:
            return f"{parts[0]} ({'; '.join(parts[1:])})"
        return self.content


@dataclass
class EpisodeNode:
    """L2 layer: a surprise-segmented contiguous span of Pages."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    page_ids: list[str] = field(default_factory=list)
    subject: str = ""
    summary: str = ""
    embedding: np.ndarray | None = None
    hyperedge: dict[str, str] = field(default_factory=dict)  # hyperedge_id -> role

    def to_text(self) -> str:
        return f"{self.subject}: {self.summary}" if self.subject else self.summary


@dataclass
class TopicNode:
    """L3 layer: an LLM-formed cluster of episodes (streaming-matched)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    summary: str = ""
    episode_ids: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    episode_hyperedge_id: str = ""

    def to_text(self) -> str:
        return f"{self.title}: {self.summary}" if self.title else self.summary


@dataclass
class FactHyperedge:
    """Connects the facts extracted from one episode."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str = ""
    relation: dict[str, str] = field(default_factory=dict)    # fact_id -> role
    weights: dict[str, float] = field(default_factory=dict)   # fact_id -> weight
    embedding: np.ndarray | None = None
    extraction_confidence: float = 0.8
    created_at: float = field(default_factory=time.time)


@dataclass
class EpisodeHyperedge:
    """Connects the episodes LLM-assigned to one topic."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic_id: str = ""
    relation: dict[str, str] = field(default_factory=dict)    # episode_id -> role
    weights: dict[str, float] = field(default_factory=dict)   # episode_id -> weight
    embedding: np.ndarray | None = None
    coherence_score: float = 0.8
    created_at: float = field(default_factory=time.time)
