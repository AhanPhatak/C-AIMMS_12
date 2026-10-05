"""Surprise segmentation for continuous documents (EM-LLM native mode).

On dialogue, ``SurpriseEpisodeSegmenter`` snaps surprise boundaries onto whole
turns. A document has no turns, so ``DocumentSurpriseSegmenter`` uses the
emitter's raw token spans and decodes each back to text. It reuses the exact
machinery of the dialogue segmenter (same ``CAIMMSBoundaryEmitter``: Stage-1
Bayesian surprisal + Stage-2 KV-modularity refinement) via ``emit_episodes``.
Ported from the IterRet-only LongBench harness (``longbench/doc_segmenter.py``).

One addition: events longer than ``max_words`` are split at word boundaries.
A stretch of text with no surprise boundary would otherwise become a single
multi-thousand-word node -- too large for the cue/tag extraction call on an
8k-context server, and too coarse to retrieve. This is the document analogue of
EM-LLM's maximum block size; event starts chosen by surprise are never moved.
"""

from __future__ import annotations

from typing import List, Sequence

from .episode_segmenter import SurpriseEpisodeSegmenter


def split_long_spans(spans: Sequence[str], max_words: int) -> List[str]:
    if max_words <= 0:
        return list(spans)
    out: List[str] = []
    for span in spans:
        words = span.split()
        if len(words) <= max_words:
            out.append(span)
            continue
        for start in range(0, len(words), max_words):
            out.append(" ".join(words[start:start + max_words]))
    return out


class DocumentSurpriseSegmenter(SurpriseEpisodeSegmenter):
    def segment_document(self, text: str, *, max_words: int = 400) -> List[str]:
        """Split ``text`` into surprise-bounded event strings. Spans are
        contiguous and together cover the whole document."""
        if not text or not text.strip():
            return []
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            return []

        episodes = self.emit_episodes(ids)

        # Contiguous, gap-free boundary set from every span endpoint; 0 and
        # len(ids) added so the head and tail are covered even if the emitter
        # did not emit them as spans of their own.
        marks = {0, len(ids)}
        for start, end in episodes:
            if 0 < start < len(ids):
                marks.add(start)
            if 0 < end < len(ids):
                marks.add(end)
        bounds = sorted(marks)

        spans: List[str] = []
        for i in range(len(bounds) - 1):
            span = self.tokenizer.decode(ids[bounds[i]:bounds[i + 1]]).strip()
            if span:
                spans.append(span)
        return split_long_spans(spans, max_words)
