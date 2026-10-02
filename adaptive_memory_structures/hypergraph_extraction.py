"""
LLM-driven extraction for HypergraphMemory.

Adapted from HyperMem's `extractors/fact_extractor.py` and
`extractors/topic_extractor.py`, scoped to del-mem's per-session unit and
duck-typed against any client exposing `.chat(user, system=None) -> str`
(both QwenClient and VLLMClient satisfy this).

Two groups of functions:
  - fact layer (per episode): summarise_episode, extract_facts_for_episode,
    assign_fact_roles.
  - topic layer (per session, streaming): episodes are processed in temporal
    order and LLM-matched against topics formed so far from earlier episodes
    in the same session -- faithfully mirroring HyperMem's live-arrival
    topic_extractor.py algorithm even though here the "arrivals" are
    already-known episodes from one batch segmentation pass.

No native JSON mode is assumed (vLLM guided-decoding isn't confirmed
enabled for the server this repo launches) -- every call prompts for JSON
and parses with one retry, same pattern as
`best_structure_evaluator.py:_generate_queries`. If the LLM is unreachable
or keeps returning garbage, each function degrades to a deterministic
fallback rather than raising, so a flaky/offline LLM never breaks indexing
outright (it only degrades hypergraph quality) -- consistent with the rest
of this directory's graceful-degradation style (see
`surprise_episode_segmenter.PageEpisodeSegmenter.segment`).
"""

from __future__ import annotations

import json
import re

from hypergraph_types import EpisodeHyperedge, EpisodeNode, EpisodeRole, FactHyperedge, FactNode, FactRole, TopicNode
from memory_structures import Page

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _call_llm_json(llm_client, system: str, user: str, retries: int = 1) -> dict | None:
    """Call the LLM and parse its reply as JSON, retrying once with a
    stricter reminder if the first reply doesn't parse. Returns None
    (never raises) if the LLM is unreachable or never produces valid JSON."""
    prompt = user
    for attempt in range(retries + 1):
        try:
            reply = llm_client.chat(user=prompt, system=system)
        except Exception:
            return None
        parsed = _parse_json(reply)
        if parsed is not None:
            return parsed
        prompt = (
            user
            + "\n\nYour previous response was not valid JSON. Respond with ONLY "
            "valid JSON, no markdown fences, no explanation."
        )
    return None


def _episode_transcript(pages: list[Page]) -> str:
    return "\n".join(p.to_text() for p in pages)


# ---------------------------------------------------------------------------
# Fact layer (per episode)
# ---------------------------------------------------------------------------

def summarize_episode(pages: list[Page], llm_client) -> tuple[str, str]:
    """Return (subject, summary) for an episode's pages."""
    transcript = _episode_transcript(pages)
    system = "You write short, factual summaries of conversation segments. Respond with ONLY valid JSON."
    user = (
        f"Conversation segment:\n{transcript}\n\n"
        'Respond in JSON: {"subject": "a 3-6 word title", "summary": "a 2-3 sentence summary"}'
    )
    parsed = _call_llm_json(llm_client, system, user)
    if parsed and parsed.get("summary"):
        subject = str(parsed.get("subject") or "").strip() or transcript[:40].strip()
        summary = str(parsed["summary"]).strip()
        return subject, summary

    # Fallback: deterministic, no LLM required.
    subject = transcript[:40].strip() or "Episode"
    summary = transcript[:300].strip()
    return subject, summary


def extract_facts_for_episode(pages: list[Page], episode_id: str, llm_client) -> list[FactNode]:
    """LLM-extract atomic, queryable facts from one episode's pages."""
    transcript = _episode_transcript(pages)
    system = (
        "You extract atomic, self-contained facts from a conversation segment. "
        "Each fact must be answerable on its own, without needing the rest of the "
        "conversation for context. Respond with ONLY valid JSON."
    )
    user = (
        f"Conversation segment:\n{transcript}\n\n"
        "Respond in JSON: {\"facts\": [{\"content\": \"...\", \"confidence\": 0.0-1.0, "
        "\"temporal\": \"...\" or null, \"spatial\": \"...\" or null, "
        "\"keywords\": [\"...\"], \"query_patterns\": [\"a question this fact answers\"]}]}"
    )
    parsed = _call_llm_json(llm_client, system, user)

    facts: list[FactNode] = []
    if parsed and isinstance(parsed.get("facts"), list):
        for item in parsed["facts"]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            facts.append(FactNode(
                episode_id=episode_id,
                content=content,
                confidence=float(item.get("confidence", 0.8) or 0.8),
                temporal=item.get("temporal") or None,
                spatial=item.get("spatial") or None,
                keywords=[str(k) for k in (item.get("keywords") or [])],
                query_patterns=[str(q) for q in (item.get("query_patterns") or [])],
            ))

    if not facts:
        # Fallback: one coarse fact so the episode is never unsearchable.
        facts.append(FactNode(episode_id=episode_id, content=transcript[:300].strip(), confidence=0.3))
    return facts


def assign_fact_roles(facts: list[FactNode], episode_id: str, episode_summary: str, llm_client) -> FactHyperedge:
    """LLM-assign a role + importance weight to each fact; build the episode's FactHyperedge."""
    hyperedge = FactHyperedge(episode_id=episode_id)
    if not facts:
        return hyperedge

    fact_lines = "\n".join(f"- {f.id}: {f.content}" for f in facts)
    valid_roles = ", ".join(r.value for r in FactRole)
    system = "You rate the importance and role of facts within a conversation episode. Respond with ONLY valid JSON."
    user = (
        f"Episode summary: {episode_summary}\n\nFacts:\n{fact_lines}\n\n"
        f'Respond in JSON: {{"facts": [{{"fact_id": "...", "role": one of [{valid_roles}], '
        '"weight": 0.0-1.0}], "extraction_confidence": 0.0-1.0}'
    )
    parsed = _call_llm_json(llm_client, system, user)

    role_by_id: dict[str, str] = {}
    weight_by_id: dict[str, float] = {}
    if parsed and isinstance(parsed.get("facts"), list):
        for item in parsed["facts"]:
            if not isinstance(item, dict):
                continue
            fid = item.get("fact_id")
            if fid not in {f.id for f in facts}:
                continue
            role = item.get("role") if item.get("role") in {r.value for r in FactRole} else FactRole.DETAIL.value
            role_by_id[fid] = role
            try:
                weight_by_id[fid] = max(0.0, min(1.0, float(item.get("weight", 0.5))))
            except (TypeError, ValueError):
                weight_by_id[fid] = 0.5
        hyperedge.extraction_confidence = float(parsed.get("extraction_confidence", 0.8) or 0.8)

    for f in facts:
        role = role_by_id.get(f.id, FactRole.DETAIL.value)
        weight = weight_by_id.get(f.id, 0.5)
        hyperedge.relation[f.id] = role
        hyperedge.weights[f.id] = weight
        f.hyperedge[hyperedge.id] = role

    return hyperedge


# ---------------------------------------------------------------------------
# Topic layer (per session, streaming)
# ---------------------------------------------------------------------------

def llm_match_topics(episode: EpisodeNode, existing_topics: list[TopicNode], llm_client, batch_size: int = 10) -> str | None:
    """Batch-match `episode` against `existing_topics`; return the id of a
    matching topic, or None if none match (mirrors topic_extractor.py's
    batched TOPIC_MATCH_PROMPT)."""
    if not existing_topics:
        return None

    system = (
        "You decide whether a new conversation episode belongs to any of the given "
        "topics. Respond with ONLY valid JSON."
    )
    for start in range(0, len(existing_topics), batch_size):
        batch = existing_topics[start:start + batch_size]
        topic_lines = "\n".join(f"- {t.id}: {t.title} -- {t.summary}" for t in batch)
        user = (
            f"Existing topics:\n{topic_lines}\n\n"
            f"New episode: {episode.subject} -- {episode.summary}\n\n"
            'Respond in JSON: {"matched_topic_id": "<id of the best-matching topic>" or null}'
        )
        parsed = _call_llm_json(llm_client, system, user)
        if parsed:
            matched = parsed.get("matched_topic_id")
            if matched in {t.id for t in batch}:
                return matched
    return None


def create_new_topic(episode: EpisodeNode, llm_client) -> TopicNode:
    """Seed a new topic from a single episode (mirrors TOPIC_EXTRACTION_PROMPT)."""
    system = "You name and summarise a new conversational topic from its first episode. Respond with ONLY valid JSON."
    user = (
        f"Episode: {episode.subject} -- {episode.summary}\n\n"
        'Respond in JSON: {"title": "a short topic title", "summary": "a 1-2 sentence topic summary"}'
    )
    parsed = _call_llm_json(llm_client, system, user)
    if parsed and parsed.get("title"):
        title = str(parsed["title"]).strip()
        summary = str(parsed.get("summary") or episode.summary).strip()
    else:
        title = episode.subject or "Topic"
        summary = episode.summary

    return TopicNode(title=title, summary=summary, episode_ids=[episode.id])


def update_existing_topic(topic: TopicNode, new_episode: EpisodeNode, llm_client) -> TopicNode:
    """Fold a new episode into an existing topic, rewriting title/summary
    (mirrors TOPIC_UPDATE_PROMPT). Mutates and returns `topic`."""
    system = "You update a conversational topic's title and summary to incorporate a new episode. Respond with ONLY valid JSON."
    user = (
        f"Current topic: {topic.title} -- {topic.summary}\n\n"
        f"New episode joining this topic: {new_episode.subject} -- {new_episode.summary}\n\n"
        'Respond in JSON: {"title": "updated short title", "summary": "updated 1-3 sentence summary"}'
    )
    parsed = _call_llm_json(llm_client, system, user)
    if parsed and parsed.get("summary"):
        topic.title = str(parsed.get("title") or topic.title).strip()
        topic.summary = str(parsed["summary"]).strip()
    topic.episode_ids.append(new_episode.id)
    return topic


def assign_episode_roles_and_weights(topic: TopicNode, member_episodes: list[EpisodeNode], llm_client) -> EpisodeHyperedge:
    """LLM-assign a role + importance weight to each of a topic's member
    episodes; build/rebuild the topic's EpisodeHyperedge over its full
    current membership (mirrors EPISODE_ROLE_WEIGHT_ASSIGNMENT_PROMPT)."""
    hyperedge = EpisodeHyperedge(topic_id=topic.id)
    if not member_episodes:
        return hyperedge

    episode_lines = "\n".join(f"- {e.id}: {e.subject} -- {e.summary}" for e in member_episodes)
    valid_roles = ", ".join(r.value for r in EpisodeRole)
    system = "You rate the role and importance of episodes within a conversational topic. Respond with ONLY valid JSON."
    user = (
        f"Topic: {topic.title} -- {topic.summary}\n\nEpisodes:\n{episode_lines}\n\n"
        f'Respond in JSON: {{"episodes": [{{"episode_id": "...", "role": one of [{valid_roles}], '
        '"weight": 0.0-1.0}], "coherence_score": 0.0-1.0}'
    )
    parsed = _call_llm_json(llm_client, system, user)

    role_by_id: dict[str, str] = {}
    weight_by_id: dict[str, float] = {}
    if parsed and isinstance(parsed.get("episodes"), list):
        for item in parsed["episodes"]:
            if not isinstance(item, dict):
                continue
            eid = item.get("episode_id")
            if eid not in {e.id for e in member_episodes}:
                continue
            role = item.get("role") if item.get("role") in {r.value for r in EpisodeRole} else EpisodeRole.DEVELOPING.value
            role_by_id[eid] = role
            try:
                weight_by_id[eid] = max(0.0, min(1.0, float(item.get("weight", 0.5))))
            except (TypeError, ValueError):
                weight_by_id[eid] = 0.5
        hyperedge.coherence_score = float(parsed.get("coherence_score", 0.8) or 0.8)

    for e in member_episodes:
        role = role_by_id.get(e.id, EpisodeRole.DEVELOPING.value)
        weight = weight_by_id.get(e.id, 0.5)
        hyperedge.relation[e.id] = role
        hyperedge.weights[e.id] = weight
        e.hyperedge = {hid: r for hid, r in e.hyperedge.items() if hid != topic.episode_hyperedge_id}
        e.hyperedge[hyperedge.id] = role

    topic.episode_hyperedge_id = hyperedge.id
    return hyperedge


def build_topics_for_session(
    episodes: list[EpisodeNode],
    llm_client,
    batch_size: int = 10,
) -> tuple[dict[str, TopicNode], dict[str, EpisodeHyperedge]]:
    """
    Stream episodes (in temporal order, i.e. the order they were segmented
    in) through LLM topic-matching, faithfully mirroring HyperMem's
    live-arrival topic_extractor.py algorithm: each new episode either joins
    a matched existing topic (topic updated, hyperedge rebuilt over the full
    new membership) or seeds a new topic.
    """
    topics: dict[str, TopicNode] = {}
    hyperedges: dict[str, EpisodeHyperedge] = {}
    episodes_by_id = {e.id: e for e in episodes}

    for episode in episodes:
        if not topics:
            topic = create_new_topic(episode, llm_client)
            topics[topic.id] = topic
            hyperedge = assign_episode_roles_and_weights(topic, [episode], llm_client)
            hyperedges[hyperedge.id] = hyperedge
            continue

        matched_id = llm_match_topics(episode, list(topics.values()), llm_client, batch_size=batch_size)

        if matched_id is None:
            topic = create_new_topic(episode, llm_client)
            topics[topic.id] = topic
            hyperedge = assign_episode_roles_and_weights(topic, [episode], llm_client)
            hyperedges[hyperedge.id] = hyperedge
        else:
            topic = topics[matched_id]
            old_hyperedge_id = topic.episode_hyperedge_id
            update_existing_topic(topic, episode, llm_client)
            member_episodes = [episodes_by_id[eid] for eid in topic.episode_ids if eid in episodes_by_id]
            hyperedge = assign_episode_roles_and_weights(topic, member_episodes, llm_client)
            hyperedges.pop(old_hyperedge_id, None)
            hyperedges[hyperedge.id] = hyperedge

    return topics, hyperedges
