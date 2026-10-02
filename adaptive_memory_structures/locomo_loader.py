"""
Loads a conversation sample from the LoCoMo dataset (data/locomo10.json) and
converts it into `Page`s HypergraphMemory can index.

LoCoMo format: a JSON list of 10 conversations, each with a `conversation`
dict holding `speaker_a`/`speaker_b` and `session_N` (a list of
{"speaker", "dia_id", "text"} turns) / `session_N_date_time` keys for
N = 1..19-32 depending on the sample.

`Page` models one user+agent exchange (`user_text` + `agent_text`), but
LoCoMo turns are single-speaker and don't strictly alternate in pairs -- so
consecutive turns are paired up two at a time into one Page each, with the
real speaker name prefixed onto the text (since `user_text`/`agent_text`
aren't literally "user"/"agent" here, just two open slots).
"""

from __future__ import annotations

import json
from typing import Any

from memory_structures import Page


def _session_keys(conversation: dict[str, Any]) -> list[str]:
    keys = [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def load_sample(
    data_path: str,
    sample_index: int = 0,
    sample_id: str | None = None,
    max_sessions: int | None = None,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """
    Returns {"sample_id", "speakers", "sessions_included", "pages"}.
    """
    with open(data_path) as f:
        data = json.load(f)

    if sample_id is not None:
        item = next(x for x in data if x["sample_id"] == sample_id)
    else:
        item = data[sample_index]

    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    session_keys = _session_keys(conversation)
    if max_sessions is not None:
        session_keys = session_keys[:max_sessions]

    turns: list[dict[str, Any]] = []
    for key in session_keys:
        turns.extend(conversation[key])
    if max_turns is not None:
        turns = turns[:max_turns]

    pages: list[Page] = []
    for i in range(0, len(turns), 2):
        t1 = turns[i]
        t2 = turns[i + 1] if i + 1 < len(turns) else None
        user_text = f"{t1['speaker']}: {t1['text']}"
        agent_text = f"{t2['speaker']}: {t2['text']}" if t2 is not None else ""
        pages.append(Page(user_text=user_text, agent_text=agent_text))

    return {
        "sample_id": item["sample_id"],
        "speakers": [speaker_a, speaker_b],
        "sessions_included": session_keys,
        "pages": pages,
    }


def list_samples(data_path: str) -> list[dict[str, Any]]:
    """Summary of every sample in the file, for picking one via --list."""
    with open(data_path) as f:
        data = json.load(f)
    summaries = []
    for i, item in enumerate(data):
        conversation = item["conversation"]
        session_keys = _session_keys(conversation)
        total_turns = sum(len(conversation[k]) for k in session_keys)
        summaries.append({
            "index": i,
            "sample_id": item["sample_id"],
            "speakers": [conversation["speaker_a"], conversation["speaker_b"]],
            "num_sessions": len(session_keys),
            "total_turns": total_turns,
        })
    return summaries
