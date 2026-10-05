"""Surprise-based episodic segmentation (EM-LLM style, arXiv:2407.09450).

A lightweight "index emitter": it segments a continuous token stream into
events at points of high Bayesian surprise (Stage 1), optionally refining the
cut points to graph-modularity boundaries over the KV-cache Key states
(Stage 2). It does NOT store episodes itself -- it emits ``(start, end)`` token
spans for an orchestrator (here, ``iterret.episode_segmenter``) to slice.

Importing this subpackage requires ``torch`` + ``transformers``; the rest of
``iterret`` does not, so it is only imported lazily where surprise
segmentation is actually requested.
"""

from .boundary_creator import StatefulSurpriseBoundary, SurpriseBoundaryPipeline
from .caimms_boundary_creator import CAIMMSBoundaryEmitter

__all__ = [
    "CAIMMSBoundaryEmitter",
    "StatefulSurpriseBoundary",
    "SurpriseBoundaryPipeline",
]
