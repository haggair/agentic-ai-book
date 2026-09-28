"""
llm_utils.py — Shared LLM utility for all GAS Book notebooks
=============================================================

Auto-selects the best available LLM backend:
  1. Ollama (local, free) — preferred, picks best installed model automatically
  2. OpenAI API          — if OPENAI_API_KEY is set and Ollama not available
  3. TEST_MODE           — mock responses, no LLM needed (fallback / CI)

Usage in notebooks:
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), '..', '..'))  # adjust depth
    from llm_utils import llm, TEST_MODE, BACKEND

    # Simple chat
    text = llm.chat([{"role": "user", "content": "Hello"}])

    # JSON output
    data = llm.chat_json([{"role": "user", "content": "Return JSON: {answer: ...}"}])

    # Tool calling (same interface as OpenAI)
    response = llm.chat_tools(messages, tools=MY_TOOLS, tool_choice="auto")
    if response.tool_calls:
        for tc in response.tool_calls:
            print(tc.function.name, tc.function.arguments)
"""

import os
import json
import re
import sys
from typing import Any, Dict, List, Optional

# ── Model preference table ─────────────────────────────────────────────────────
# (name_substring, supports_tools, json_reliable, quality_score)
# Higher quality_score = preferred. Matched by substring against installed model names.
_MODEL_PREFERENCE = [
    ("qwen2.5:72b",     True,  True,  100),
    ("qwen2.5:32b",     True,  True,   90),
    ("qwen2.5:14b",     True,  True,   80),
    ("llama3.3",        True,  True,   75),
    ("llama3.1:70b",    True,  True,   74),
    ("qwen2.5:7b",      True,  True,   70),  # ← sweet spot: fast, great JSON + tools
    ("llama3.1:8b",     True,  True,   68),
    ("mistral:7b",      True,  True,   65),
    ("mistral-nemo",    True,  True,   65),
    ("phi4",            True,  True,   64),
    ("phi4-mini",       True,  True,   62),
    ("gemma3:12b",      True,  True,   60),
    ("gemma3:4b",       True,  True,   55),
    ("qwen2.5:3b",      True,  True,   50),
    ("llama3.2:3b",     True,  False,  40),
    ("llama3.2",        True,  False,  38),  # weak JSON but has tools
    ("llama3.1",        True,  True,   45),
    ("mistral:latest",  True,  True,   45),
    ("phi3",            False, False,  30),
    ("llama2",          False, False,  20),
    ("llama3.2:1b",     False, False,  10),
]

# ── Backend detection ──────────────────────────────────────────────────────────

def _detect_ollama(base_url: str, preferred_model: str) -> Optional[str]:
    """Return best available Ollama model name, or None if Ollama unreachable."""
    try:
        import requests
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        if r.status_code != 200:
            return None
        installed = [m["name"] for m in r.json().get("models", [])]
        if not installed:
            return None

        # If user explicitly set OLLAMA_MODEL and it's installed, use it
        if preferred_model:
            for inst in installed:
                if preferred_model in inst or inst in preferred_model:
                    return inst

        # Otherwise pick by preference table
        best_model = None
        best_score = -1
        for inst in installed:
            for (pattern, _, _, score) in _MODEL_PREFERENCE:
                if pattern in inst and score > best_score:
                    best_score = score
                    best_model = inst
                    break

        # If nothing matched the table, just use the first installed model
        return best_model or installed[0]

    except Exception:
        return None


def _get_model_caps(model_name: str) -> Dict[str, bool]:
    """Return capability flags for a model name."""
    for (pattern, tools, json_rel, _) in _MODEL_PREFERENCE:
        if pattern in model_name:
            return {"tools": tools, "json_reliable": json_rel}
    # Unknown model — assume basic capabilities
    return {"tools": True, "json_reliable": False}


# ── Normalized response objects ────────────────────────────────────────────────

class _FunctionCall:
    """Mimics openai.types.chat.ChatCompletionMessageToolCall.function"""
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments  # JSON string, same as OpenAI


class _ToolCall:
    """Mimics openai.types.chat.ChatCompletionMessageToolCall"""
    def __init__(self, id: str, function: _FunctionCall):
        self.id = id
        self.type = "function"
        self.function = function


class _Message:
    """Mimics openai.types.chat.ChatCompletionMessage"""
    def __init__(self, content: Optional[str], tool_calls: Optional[List[_ToolCall]] = None):
        self.content = content
        self.role = "assistant"
        self.tool_calls = tool_calls or []

    def model_dump(self, backend: str = "openai") -> dict:
        """For appending back to messages list.
        backend="ollama" → arguments as dict (Ollama requires this in message history)
        backend="openai" → arguments as JSON string (OpenAI standard)
        """
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": (
                            json.loads(tc.function.arguments)
                            if backend == "ollama"
                            else tc.function.arguments
                        ),
                    },
                }
                for tc in self.tool_calls
            ]
        return d


class _Choice:
    def __init__(self, message: _Message):
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class _Response:
    """Mimics openai.types.chat.ChatCompletion — works for both Ollama and OpenAI."""
    def __init__(self, message: _Message, backend: str = "openai"):
        self._backend = backend
        # Patch message.model_dump to use the correct backend automatically
        _backend = backend
        _orig_dump = message.model_dump
        message.model_dump = lambda: _orig_dump(backend=_backend)
        self.choices = [_Choice(message)]

    @property
    def tool_calls(self):
        return self.choices[0].message.tool_calls


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified LLM client. Works with Ollama, OpenAI, or TEST_MODE.
    All methods return plain strings or normalized response objects.
    """

    def __init__(self, backend: str, model: str, base_url: str = "",
                 api_key: str = "", caps: Optional[Dict] = None):
        self.backend = backend      # "ollama" | "openai" | "test"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.caps = caps or {"tools": True, "json_reliable": True}

    # ── chat ──────────────────────────────────────────────────────────────────

    def chat(self, messages: List[Dict], temperature: float = 0.1,
             max_tokens: int = 1024, system: Optional[str] = None) -> str:
        """Simple chat completion. Returns response text."""
        msgs = self._prepend_system(messages, system)

        if self.backend == "test":
            return self._mock_chat(msgs)

        if self.backend == "ollama":
            return self._ollama_chat(msgs, temperature, max_tokens)

        # openai
        return self._openai_chat(msgs, temperature, max_tokens)

    # ── chat_json ─────────────────────────────────────────────────────────────

    def chat_json(self, messages: List[Dict], temperature: float = 0.1,
                  system: Optional[str] = None) -> dict:
        """
        Chat completion that returns a parsed dict.
        Adds JSON instruction to system prompt automatically.
        Falls back to regex extraction if model returns markdown-wrapped JSON.
        """
        json_hint = "Reply ONLY with valid JSON. No markdown, no explanation."
        if system:
            system = system + "\n\n" + json_hint
        else:
            system = json_hint

        msgs = self._prepend_system(messages, system)

        if self.backend == "test":
            raw = self._mock_chat(msgs)
        elif self.backend == "ollama":
            raw = self._ollama_chat(msgs, temperature, max_tokens=1024)
        else:
            raw = self._openai_chat_json(msgs, temperature)

        return self._parse_json(raw)

    # ── chat_tools ────────────────────────────────────────────────────────────

    def chat_tools(self, messages: List[Dict], tools: List[Dict],
                   tool_choice: str = "auto",
                   temperature: float = 0.1) -> _Response:
        """
        Tool-calling completion. Returns a _Response object with the same
        interface as openai ChatCompletion — .choices[0].message.tool_calls etc.

        The returned message object has a .model_dump() method so you can
        append it back to messages:
            response = llm.chat_tools(messages, tools)
            messages.append(response.choices[0].message.model_dump())
        """
        if self.backend == "test":
            return self._mock_tools_response(messages, tools)

        if self.backend == "ollama":
            return self._ollama_tools(messages, tools, tool_choice, temperature)

        return self._openai_tools(messages, tools, tool_choice, temperature)

    # ── internal: ollama ──────────────────────────────────────────────────────

    def _ollama_chat(self, messages, temperature, max_tokens) -> str:
        import requests
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _ollama_chat_json(self, messages, temperature) -> str:
        import requests
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",  # Ollama native JSON mode
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _ollama_tools(self, messages, tools, tool_choice, temperature) -> _Response:
        import requests
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        msg_data = data.get("message", {})
        content = msg_data.get("content", "")
        raw_tool_calls = msg_data.get("tool_calls", [])

        tool_calls = []
        for i, tc in enumerate(raw_tool_calls):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            # Ollama may return args as dict; OpenAI expects JSON string
            if isinstance(args, dict):
                args = json.dumps(args)
            tool_calls.append(_ToolCall(
                id=f"call_{i}",
                function=_FunctionCall(name=fn.get("name", ""), arguments=args)
            ))

        return _Response(_Message(content=content or None, tool_calls=tool_calls), backend="ollama")

    # ── internal: openai ──────────────────────────────────────────────────────

    def _openai_chat(self, messages, temperature, max_tokens) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        r = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content

    def _openai_chat_json(self, messages, temperature) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        r = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return r.choices[0].message.content

    def _openai_tools(self, messages, tools, tool_choice, temperature) -> _Response:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        r = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        oai_msg = r.choices[0].message
        content = oai_msg.content

        tool_calls = []
        for tc in (oai_msg.tool_calls or []):
            tool_calls.append(_ToolCall(
                id=tc.id,
                function=_FunctionCall(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
            ))
        return _Response(_Message(content=content, tool_calls=tool_calls))

    # ── internal: mock ────────────────────────────────────────────────────────

    def _mock_chat(self, messages) -> str:
        """Deterministic mock based on last user message content."""
        last = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")
        p = last.lower()

        if "json" in p or "score" in p or "rate" in p or "evaluate" in p:
            return '{"score": 4, "reasoning": "Clinically appropriate response.", "is_safe": true, "is_excellent": false}'
        if "hypoglycemia" in p or "low glucose" in p or "54" in p or "42" in p or "45" in p:
            return ("Step 1: Glucose is critically low — this is severe hypoglycemia.\n"
                    "Step 2: Give 15g fast-acting carbohydrates immediately.\n"
                    "Step 3: Recheck in 15 minutes. Do NOT give insulin.\n"
                    "Answer: Administer fast-acting carbs now. Alert care team.")
        if "insulin" in p and ("dose" in p or "correction" in p):
            return "Correction dose: 2 units rapid-acting insulin. Recheck in 2 hours."
        if "meal" in p or "breakfast" in p or "food" in p:
            return "Recommend: Low-glycemic meal — eggs, vegetables, whole grain toast. Avoid simple sugars."
        if "summary" in p or "summarize" in p:
            return "Patient maintained good glucose control today with 78% time-in-range. One mild hypoglycemic event resolved with carbohydrate treatment."
        if "recommend" in p or "advise" in p:
            return "Recommendation: Continue current insulin regimen. Monitor glucose every 2 hours. Stay hydrated."
        if "step by step" in p or "chain of thought" in p or "cot" in p:
            return ("Let me think step by step.\n"
                    "Step 1: Assess the current glucose level and trend.\n"
                    "Step 2: Compare against target range (70-180 mg/dL).\n"
                    "Step 3: Apply ADA guidelines for the identified condition.\n"
                    "Answer: Follow standard clinical protocol for this glucose level.")
        return "Glucose is within acceptable range. Continue current management plan and recheck in 2 hours."

    def _mock_tools_response(self, messages, tools) -> _Response:
        """Return a mock tool call for the first tool in the list."""
        if not tools:
            return _Response(_Message(content="No tools available.", tool_calls=[]))

        first_tool = tools[0]["function"]
        tool_name = first_tool["name"]

        # Build minimal valid args from the tool's required parameters
        props = first_tool.get("parameters", {}).get("properties", {})
        required = first_tool.get("parameters", {}).get("required", [])
        args = {}
        for param in required:
            ptype = props.get(param, {}).get("type", "string")
            if ptype == "string":
                args[param] = "patient-001" if "patient" in param else "mock_value"
            elif ptype == "number":
                args[param] = 120.0
            else:
                args[param] = "mock"

        return _Response(_Message(
            content=None,
            tool_calls=[_ToolCall(
                id="call_mock_0",
                function=_FunctionCall(name=tool_name, arguments=json.dumps(args))
            )]
        ))

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _prepend_system(messages: List[Dict], system: Optional[str]) -> List[Dict]:
        if not system:
            return messages
        if messages and messages[0].get("role") == "system":
            # Merge with existing system message
            merged = messages[0]["content"] + "\n\n" + system
            return [{"role": "system", "content": merged}] + messages[1:]
        return [{"role": "system", "content": system}] + messages

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        # Strip markdown code fences
        if raw.startswith("```"):
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except Exception:
                    continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract first {...} block
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        # Last resort: return wrapped string
        return {"raw": raw, "error": "Could not parse JSON"}


# ── Module-level initialization ────────────────────────────────────────────────

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_PREFERRED_MODEL = os.getenv("OLLAMA_MODEL", "")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
_FORCE_TEST = os.getenv("NOTEBOOK_TEST_MODE", "0") == "1"

BACKEND: str   # "ollama" | "openai" | "test"
MODEL: str
TEST_MODE: bool

if _FORCE_TEST:
    BACKEND = "test"
    MODEL = "mock"
    TEST_MODE = True
else:
    _ollama_model = _detect_ollama(_OLLAMA_BASE_URL, _PREFERRED_MODEL)
    if _ollama_model:
        BACKEND = "ollama"
        MODEL = _ollama_model
        TEST_MODE = False
    elif _OPENAI_KEY:
        BACKEND = "openai"
        MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        TEST_MODE = False
    else:
        BACKEND = "test"
        MODEL = "mock"
        TEST_MODE = True

# Build the shared client
_caps = _get_model_caps(MODEL) if BACKEND == "ollama" else {"tools": True, "json_reliable": True}

llm = LLMClient(
    backend=BACKEND,
    model=MODEL,
    base_url=_OLLAMA_BASE_URL,
    api_key=_OPENAI_KEY,
    caps=_caps,
)

# ── Startup banner ─────────────────────────────────────────────────────────────

_ICONS = {"ollama": "🦙", "openai": "🤖", "test": "🧪"}
_LABELS = {"ollama": f"Ollama local ({MODEL})", "openai": f"OpenAI ({MODEL})", "test": "TEST MODE (mock data)"}

print(f"{_ICONS[BACKEND]} LLM backend: {_LABELS[BACKEND]}")

if BACKEND == "ollama":
    if not _caps["json_reliable"]:
        print(f"   ⚠️  {MODEL} has weak JSON output. Consider: ollama pull qwen2.5:7b")
    if not _caps["tools"]:
        print(f"   ⚠️  {MODEL} does not support tool calling. Consider: ollama pull qwen2.5:7b")
    else:
        print(f"   ✅ Tool calling: supported | JSON mode: {'reliable' if _caps['json_reliable'] else 'basic'}")

if BACKEND == "test":
    print("   ℹ️  To use a real model: install Ollama + run: ollama pull qwen2.5:7b")
    print("   ℹ️  Or set OPENAI_API_KEY environment variable")
