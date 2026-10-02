"""
Minimal OpenAI-compatible vLLM client for HypergraphMemory's LLM-JSON calls
(fact extraction, role/weight assignment, topic matching/summarisation).

Mirrors IterRet/iterret/llm_client.py's OpenAICompatibleLLMClient (same
env-var convention, same "never raise on construction" behaviour so importing
this module is side-effect free), but exposes a `chat(user, system=None)`
method shaped like QwenClient.chat -- so HypergraphMemory's extraction helpers
can accept either client interchangeably.

Defaults match the vLLM server the rest of this repo already launches
(scripts/run_pipeline.sh): Qwen/Qwen3-4B-Instruct-2507 on $VLLM_PORT.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ENV_BASE_URL = "ITERRET_LLM_BASE_URL"
ENV_MODEL = "ITERRET_LLM_MODEL"


def _openai_importable() -> bool:
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_base_url(cli_value: str | None = None) -> str:
    return cli_value or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL


def resolve_model(cli_value: str | None = None) -> str:
    return cli_value or os.environ.get(ENV_MODEL) or DEFAULT_MODEL


def vllm_available() -> bool:
    """True if the `openai` package is importable and a base URL is
    configured (env var set, or the shared default). Does not probe the
    network -- a real connection failure surfaces on the first `chat()` call,
    same as `IterRet/iterret/llm_client.py`'s client."""
    return _openai_importable()


class VLLMClient:
    """Wraps an OpenAI-compatible chat-completions endpoint served by vLLM."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str = "not-needed",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = resolve_base_url(base_url)
        self.model = resolve_model(model)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        except Exception:
            self._client = None

    def chat(
        self,
        user: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **_ignored: Any,
    ) -> str:
        """Return the model's reply text. Signature mirrors QwenClient.chat
        so callers can treat either client uniformly."""
        if self._client is None:
            raise RuntimeError(
                "VLLMClient has no usable client (the `openai` package is missing "
                "or could not be imported). Install `openai` and point "
                f"{ENV_BASE_URL} at a running vLLM server (tried {self.base_url})."
            )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_new_tokens if max_new_tokens is not None else self.max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content or ""
