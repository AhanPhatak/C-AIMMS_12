"""Build a Cue-Tag-Content graph from a long DOCUMENT (LongBench / Qasper).

The document analogue of ``memory_builder.build_ctc_graph_from_dialogue``: the
text is split into passages (``chunk_document``), each passage becomes one
episodic content node with LLM-extracted cues + tag, and a prose-tuned semantic
layer adds distilled fact nodes. Ported from the IterRet-only LongBench harness
(``longbench/doc_memory_builder.py`` in the outer repo), with three changes:

  * no topic layer -- it was never linked into cue->tag->content, so retrieval
    could not reach it (same reason it was removed from the dialogue builder);
  * the semantic call gets ``memory_builder.SEMANTIC_MAX_TOKENS`` instead of the
    client's 256-token default, which silently truncated the JSON reply;
  * passage extraction runs in a small thread pool (calls are independent and
    vLLM batches concurrent requests), since a document has 25-60 passages.

Passages carry no timestamp, so date resolution is a no-op for these graphs.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

from .ctc_graph import CueTagContentGraph
from .llm_client import LLMClient
from .memory_builder import (
    DEFAULT_MAX_CHARS_PER_CALL,
    SEMANTIC_MAX_TOKENS,
    _iter_chunks,
    _safe_chat,
)

BUILD_WORKERS = int(os.environ.get("ITERRET_BUILD_WORKERS", "8"))

_DOC_EVENT_EXTRACTION_SYSTEM_PROMPT = """episode_extraction
You build a Cue-Tag-Content memory graph from a passage of a long document.
Read the passage and produce:
- "tag": a short phrase (<=4 words) naming what the passage is about
  (e.g. "Trial Verdict", "Dataset Statistics", "Method Setup").
- "cues": 5-12 fine-grained cues -- named entities (people, places,
  organizations), dates, numbers, technical terms, or salient key nouns
  explicitly mentioned in the passage. Extract generously: these are the
  surface-form anchors a later query will match on, so more coverage means
  more of the passage is retrievable.
Reply as JSON: {"tag": str, "cues": [str, ...]}.
"""

_DOC_SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """semantic_extraction
You extract stand-alone factual statements from passages of a long document
(the Cue-Tag-Content semantic layer) -- definitions, attributes, results,
relationships, or claims that a reader might later ask about. For each fact,
give the entity/topic cue it is anchored to (a name, term, or key noun), an
aspect tag (e.g. "Definition", "Result", "Property", "Cause"), and the fact as
one short self-contained sentence.
Reply as JSON: {"semantics": [{"cue": str, "tag": str, "content": str}, ...]}.
Extract several facts per passage where possible; reply {"semantics": []} only
if the passages contain no factual content.
"""


def chunk_document(text: str, *, target_words: int = 180, max_words: int = 300) -> List[str]:
    """Split a document into passages of ~target_words, respecting paragraphs.

    Paragraphs (blank-line separated; single newlines if the text has none) are
    merged until a passage reaches target_words; a paragraph longer than
    max_words is cut at word boundaries. Every word lands in exactly one passage.
    """
    if not text or not text.strip():
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if p.strip()]

    pieces: List[str] = []
    for p in paras:
        words = p.split()
        for start in range(0, len(words), max_words):
            pieces.append(" ".join(words[start:start + max_words]))

    passages: List[str] = []
    buf: List[str] = []
    buf_words = 0
    for piece in pieces:
        n = len(piece.split())
        if buf and buf_words + n > max_words:
            passages.append("\n".join(buf))
            buf, buf_words = [], 0
        buf.append(piece)
        buf_words += n
        if buf_words >= target_words:
            passages.append("\n".join(buf))
            buf, buf_words = [], 0
    if buf:
        passages.append("\n".join(buf))
    return passages


def _extract_doc_event(span_text: str, llm: LLMClient) -> dict:
    parsed = _safe_chat(
        _DOC_EVENT_EXTRACTION_SYSTEM_PROMPT, json.dumps({"text": span_text}), llm,
        on_error="doc event extraction failed, using fallback tag/cues",
    )
    tag = str(parsed.get("tag") or "Passage")
    cues = [str(c) for c in parsed.get("cues") or [] if str(c).strip()]
    if not cues:
        cues = ["Unknown"]
    return {"tag": tag, "cues": cues}


def _extract_doc_semantics(episode_summaries: List[dict], llm: LLMClient, *, max_chars: int) -> List[dict]:
    chunks = list(_iter_chunks(episode_summaries, text_key="text", max_chars=max_chars))

    def _one(chunk: List[dict]) -> List[dict]:
        full_text = "\n".join(s["text"] for s in chunk)
        parsed = _safe_chat(_DOC_SEMANTIC_EXTRACTION_SYSTEM_PROMPT, full_text, llm,
                            on_error=f"doc semantic extraction skipped a {len(chunk)}-passage chunk",
                            max_tokens=SEMANTIC_MAX_TOKENS)
        out = []
        for item in parsed.get("semantics") or []:
            if not isinstance(item, dict):
                continue
            cue, tag, content = item.get("cue"), item.get("tag"), item.get("content")
            if cue and tag and content:
                out.append({"cue": str(cue), "tag": str(tag), "content": str(content)})
        return out

    with ThreadPoolExecutor(max_workers=max(1, BUILD_WORKERS)) as pool:
        results = list(pool.map(_one, chunks))
    return [fact for facts in results for fact in facts]


def build_ctc_graph_from_document(
    spans: Sequence[str], llm: LLMClient, *,
    max_chars_per_call: int = DEFAULT_MAX_CHARS_PER_CALL,
) -> CueTagContentGraph:
    """One episodic node per passage (in document order) + a semantic layer."""
    graph = CueTagContentGraph()

    with ThreadPoolExecutor(max_workers=max(1, BUILD_WORKERS)) as pool:
        extracted = list(pool.map(lambda s: _extract_doc_event(s, llm), spans))

    episode_summaries: List[dict] = []
    for i, (span, ext) in enumerate(zip(spans, extracted)):
        content_id = f"e{i + 1}"
        graph.add_content(content_id, span, layer="episodic", time=None)
        for cue in ext["cues"]:
            graph.link(cue, ext["tag"], content_id)
        episode_summaries.append({"content_id": content_id, "tag": ext["tag"], "text": span})

    semantics = _extract_doc_semantics(episode_summaries, llm, max_chars=max_chars_per_call)
    for j, semantic in enumerate(semantics):
        content_id = f"s{j + 1}"
        graph.add_content(content_id, semantic["content"], layer="semantic")
        graph.link(semantic["cue"], semantic["tag"], content_id)

    graph.meta["source"] = "document"
    graph.meta["n_passages"] = len(spans)
    graph.meta["n_semantic"] = len(semantics)
    return graph
