"""
shared/llm_client.py
====================
Unified LLM wrapper that works with OpenAI, Anthropic, and Ollama (local).

Backend auto-detection order:
  1. OPENAI_API_KEY  → OpenAI  (default model: gpt-4o-mini)
  2. ANTHROPIC_API_KEY → Anthropic (default model: claude-3-haiku-20240307)
  3. Neither set     → Ollama local (default model: llama3.2)

Set NOTEBOOK_TEST_MODE=1 to return deterministic mock responses (CI-safe).

Usage
-----
    from shared.llm_client import LLMClient

    client = LLMClient()
    response = client.chat([{"role": "user", "content": "Hello!"}])
    print(response)
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards — imported lazily so the module can be imported
# even when only one backend is installed.
# ---------------------------------------------------------------------------

def _import_openai():
    try:
        import openai
        return openai
    except ImportError as exc:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        ) from exc


def _import_anthropic():
    try:
        import anthropic
        return anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc


def _import_requests():
    try:
        import requests
        return requests
    except ImportError as exc:
        raise ImportError(
            "requests package not installed. Run: pip install requests"
        ) from exc


# ---------------------------------------------------------------------------
# Retry helper (tenacity optional — falls back to simple loop)
# ---------------------------------------------------------------------------

def _with_retry(fn, max_attempts: int = 3, base_delay: float = 1.0):
    """
    Call *fn* with exponential-backoff retry.

    Uses tenacity when available; otherwise falls back to a plain loop so the
    module works in minimal environments.
    """
    try:
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _wrapped():
            return fn()

        return _wrapped()

    except ImportError:
        logger.debug("tenacity not installed; using simple retry loop")
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "LLMClient attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Mock response bank (NOTEBOOK_TEST_MODE)
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    "This is a mock LLM response for testing purposes.",
    "Mock response: The patient's glucose level appears to be within normal range.",
    "Mock response: Based on the provided data, I recommend monitoring closely.",
    "Mock response: No immediate action required at this time.",
    "Mock response: Please consult a healthcare professional for medical advice.",
]

_mock_counter = 0


def _get_mock_response(messages: list[dict]) -> str:
    """Return a deterministic mock response cycling through _MOCK_RESPONSES."""
    global _mock_counter
    # Use last user message content as a hint when possible
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if "glucose" in content.lower() or "patient" in content.lower():
                return _MOCK_RESPONSES[1]
            if "recommend" in content.lower():
                return _MOCK_RESPONSES[2]
    response = _MOCK_RESPONSES[_mock_counter % len(_MOCK_RESPONSES)]
    _mock_counter += 1
    return response


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Estimate token count for *text*.

    Uses tiktoken when available (accurate for OpenAI models).
    Falls back to a simple word-based heuristic (~1.3 tokens/word).

    Parameters
    ----------
    text:
        The string to count tokens for.
    model:
        Model name used to select the correct tiktoken encoding.

    Returns
    -------
    int
        Estimated token count.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Heuristic: ~1.3 tokens per word (conservative estimate)
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client supporting OpenAI, Anthropic, and Ollama backends.

    Backend is selected automatically based on environment variables:
      - ``OPENAI_API_KEY``    → OpenAI
      - ``ANTHROPIC_API_KEY`` → Anthropic
      - (neither)             → Ollama (local)

    Set ``NOTEBOOK_TEST_MODE=1`` to bypass all API calls and return mock
    responses — useful for CI pipelines and notebook testing.

    Parameters
    ----------
    backend : str | None
        Force a specific backend: ``"openai"``, ``"anthropic"``, or
        ``"ollama"``.  When *None* (default) the backend is auto-detected.
    model : str | None
        Override the default model for the selected backend.
    ollama_base_url : str
        Base URL for the Ollama REST API (default: ``http://localhost:11434``).
    max_retries : int
        Number of retry attempts on transient errors (default: 3).
    """

    #: Default models per backend
    DEFAULT_MODELS: dict[str, str] = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "ollama": "llama3.2",
    }

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
        max_retries: int = 3,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.max_retries = max_retries
        self.test_mode = bool(os.environ.get("NOTEBOOK_TEST_MODE"))

        # Resolve backend
        if backend is not None:
            backend = backend.lower()
            if backend not in self.DEFAULT_MODELS:
                raise ValueError(
                    f"Unknown backend '{backend}'. "
                    f"Choose from: {list(self.DEFAULT_MODELS)}"
                )
            self.backend = backend
        else:
            self.backend = self._detect_backend()

        # Resolve model
        self.model = model or self.DEFAULT_MODELS[self.backend]

        logger.info(
            "LLMClient initialised — backend=%s, model=%s, test_mode=%s",
            self.backend,
            self.model,
            self.test_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a chat completion request and return the assistant's reply.

        Parameters
        ----------
        messages : list[dict]
            List of message dicts with ``"role"`` and ``"content"`` keys.
            Example::

                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": "What is 2+2?"},
                ]

        model : str | None
            Override the instance-level model for this call only.
        temperature : float
            Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens : int
            Maximum tokens in the response.

        Returns
        -------
        str
            The assistant's text response.
        """
        if self.test_mode:
            logger.debug("NOTEBOOK_TEST_MODE active — returning mock response")
            return _get_mock_response(messages)

        effective_model = model or self.model

        dispatch = {
            "openai": self._chat_openai,
            "anthropic": self._chat_anthropic,
            "ollama": self._chat_ollama,
        }
        handler = dispatch[self.backend]

        return _with_retry(
            lambda: handler(messages, effective_model, temperature, max_tokens),
            max_attempts=self.max_retries,
        )

    def count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Estimate the total token count for a list of messages.

        Parameters
        ----------
        messages : list[dict]
            Chat messages (same format as :meth:`chat`).

        Returns
        -------
        int
            Estimated token count across all messages.
        """
        total = 0
        for msg in messages:
            total += count_tokens(msg.get("content", ""), model=self.model)
            total += 4  # per-message overhead (role + separators)
        total += 2  # reply priming
        return total

    def __repr__(self) -> str:
        return (
            f"LLMClient(backend={self.backend!r}, model={self.model!r}, "
            f"test_mode={self.test_mode})"
        )

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend() -> str:
        if os.environ.get("OPENAI_API_KEY"):
            logger.debug("Auto-detected backend: openai")
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            logger.debug("Auto-detected backend: anthropic")
            return "anthropic"
        logger.debug("No API keys found — defaulting to ollama")
        return "ollama"

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _chat_openai(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        openai = _import_openai()
        client = openai.OpenAI()  # reads OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _chat_anthropic(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        anthropic = _import_anthropic()
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

        # Anthropic separates system prompt from the messages list
        system_prompt = ""
        chat_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_messages,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _chat_ollama(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        requests = _import_requests()
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


# ---------------------------------------------------------------------------
# CLI / __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Force test mode when running standalone without API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["NOTEBOOK_TEST_MODE"] = "1"
        print("No API keys found — running in NOTEBOOK_TEST_MODE\n")

    client = LLMClient()
    print(f"Client: {client}\n")

    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {
            "role": "user",
            "content": (
                "A patient with Type 1 diabetes has a glucose reading of 55 mg/dL. "
                "What immediate action should be taken?"
            ),
        },
    ]

    print("Sending chat request...")
    reply = client.chat(messages, temperature=0.3, max_tokens=256)
    print(f"\nAssistant: {reply}")

    token_estimate = client.count_message_tokens(messages)
    print(f"\nEstimated prompt tokens: {token_estimate}")
