# `llm_utils.py` — Design & API Reference

> Shared LLM utility for all GAS Book notebooks. Auto-selects the best available backend with zero configuration required.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Architecture](#architecture)
3. [Backend Detection Flow](#backend-detection-flow)
4. [Model Preference Table](#model-preference-table)
5. [API Reference](#api-reference)
   - [`llm.chat()`](#llmchat)
   - [`llm.chat_json()`](#llmchat_json)
   - [`llm.chat_tools()`](#llmchat_tools)
   - [Response Objects](#response-objects)
6. [Module-Level Exports](#module-level-exports)
7. [Environment Variables](#environment-variables)
8. [Notebook Integration Pattern](#notebook-integration-pattern)
9. [Startup Banner](#startup-banner)
10. [How to Add a New Model](#how-to-add-a-new-model)
11. [How to Add a New Backend](#how-to-add-a-new-backend)
12. [TEST_MODE / CI](#test_mode--ci)

---

## Philosophy

**Open-source first, zero config.** Every notebook in this repo should run without an API key. The design priorities are:

1. **Local by default** — Ollama runs on your laptop, costs nothing, and keeps data private.
2. **Automatic** — no `.env` editing, no model name to remember; the library picks the best model you have installed.
3. **OpenAI-compatible interface** — notebooks written for the OpenAI SDK work unchanged; `llm_utils` normalises all backends to the same response shape.
4. **Graceful degradation** — if nothing is available, fall back to deterministic mock responses so notebooks still execute (useful for CI and quick demos).

---

## Architecture

```
llm_utils.py
│
├── _MODEL_PREFERENCE          # Ranked list of known models + capability flags
│
├── _detect_ollama()           # Probes Ollama HTTP API; returns best installed model
├── _get_model_caps()          # Looks up tool/JSON capability flags for a model name
│
├── Response objects           # _FunctionCall, _ToolCall, _Message, _Choice, _Response
│   └── Mirror openai SDK types so notebook dispatch code works unchanged
│
├── LLMClient                  # The main class — one instance shared across a notebook
│   ├── .chat()                # Plain text completion
│   ├── .chat_json()           # JSON-mode completion with auto-parse + fence stripping
│   └── .chat_tools()          # Tool-calling completion; returns _Response
│
└── Module init (bottom)       # Detects backend, builds `llm`, prints startup banner
    ├── llm       (LLMClient)
    ├── BACKEND   (str)
    ├── MODEL     (str)
    └── TEST_MODE (bool)
```

---

## Backend Detection Flow

When `llm_utils` is imported, it runs this decision tree **once**:

```
NOTEBOOK_TEST_MODE=1 set?
    └─ YES → BACKEND="test", MODEL="mock"
    └─ NO  →
        Ollama reachable at OLLAMA_BASE_URL?
            └─ YES →
                OLLAMA_MODEL env var set AND installed?
                    └─ YES → use that model
                    └─ NO  → rank installed models by _MODEL_PREFERENCE → pick highest score
                BACKEND="ollama", MODEL=<best installed>
            └─ NO  →
                OPENAI_API_KEY set?
                    └─ YES → BACKEND="openai", MODEL=OPENAI_MODEL (default: gpt-4o-mini)
                    └─ NO  → BACKEND="test", MODEL="mock"
```

The Ollama probe is a single `GET /api/tags` with a **2-second timeout** — fast enough to not slow notebook startup.

---

## Model Preference Table

The `_MODEL_PREFERENCE` list in `llm_utils.py` ranks known Ollama models. Each entry is:

```python
(name_substring, supports_tools, json_reliable, quality_score)
```

| Model | Tools | JSON | Score | Notes |
|---|:---:|:---:|:---:|---|
| `qwen2.5:72b` | ✅ | ✅ | 100 | Best quality, needs ~45 GB RAM |
| `qwen2.5:32b` | ✅ | ✅ | 90 | Excellent, needs ~20 GB RAM |
| `qwen2.5:14b` | ✅ | ✅ | 80 | Great balance |
| `llama3.3` | ✅ | ✅ | 75 | Meta's latest flagship |
| `llama3.1:70b` | ✅ | ✅ | 74 | Large but capable |
| **`qwen2.5:7b`** | ✅ | ✅ | **70** | **← recommended sweet spot** |
| `llama3.1:8b` | ✅ | ✅ | 68 | Good alternative |
| `mistral:7b` | ✅ | ✅ | 65 | Solid all-rounder |
| `mistral-nemo` | ✅ | ✅ | 65 | — |
| `phi4` | ✅ | ✅ | 64 | Microsoft, compact |
| `phi4-mini` | ✅ | ✅ | 62 | — |
| `gemma3:12b` | ✅ | ✅ | 60 | Google |
| `gemma3:4b` | ✅ | ✅ | 55 | — |
| `qwen2.5:3b` | ✅ | ✅ | 50 | Tiny but functional |
| `llama3.2:3b` | ✅ | ⚠️ | 40 | Weak JSON |
| `llama3.2` | ✅ | ⚠️ | 38 | Weak JSON (currently installed on dev machine) |
| `llama3.1` | ✅ | ✅ | 45 | — |
| `mistral:latest` | ✅ | ✅ | 45 | — |
| `phi3` | ❌ | ❌ | 30 | No tools |
| `llama2` | ❌ | ❌ | 20 | Legacy |
| `llama3.2:1b` | ❌ | ❌ | 10 | Too small |

**Matching is by substring**: `"llama3.2"` matches `"llama3.2:latest"`, `"llama3.2:3b"`, etc. More specific entries (e.g. `"llama3.2:3b"`) should appear **before** less specific ones (e.g. `"llama3.2"`) in the list so they get priority.

---

## API Reference

### `llm.chat()`

```python
def chat(
    messages: List[Dict],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    system: Optional[str] = None,
) -> str
```

Plain text completion. Returns the assistant's response as a string.

- `messages` — OpenAI-format list: `[{"role": "user", "content": "..."}]`
- `temperature` — sampling temperature (0.0 = deterministic)
- `max_tokens` — maximum tokens in the response
- `system` — optional system prompt; prepended automatically (merged if a system message already exists in `messages`)

**Example:**
```python
reply = llm.chat(
    [{"role": "user", "content": "What is the target glucose range for a diabetic patient?"}],
    system="You are a clinical AI assistant."
)
print(reply)
```

---

### `llm.chat_json()`

```python
def chat_json(
    messages: List[Dict],
    temperature: float = 0.1,
    system: Optional[str] = None,
) -> dict
```

JSON-mode completion. Returns a parsed `dict`.

Internally:
- Appends `"Reply ONLY with valid JSON. No markdown, no explanation."` to the system prompt
- For OpenAI: uses `response_format={"type": "json_object"}`
- For Ollama: uses `"format": "json"` (native JSON mode)
- Strips markdown code fences (` ```json ... ``` `) if the model wraps its output
- Falls back to regex `{...}` extraction if `json.loads` fails
- Last resort: returns `{"raw": "<text>", "error": "Could not parse JSON"}`

**Example:**
```python
result = llm.chat_json([{
    "role": "user",
    "content": 'Evaluate this advice: "Take 2 units insulin." Return: {"score": 1-5, "is_safe": bool}'
}])
print(result["score"], result["is_safe"])
```

---

### `llm.chat_tools()`

```python
def chat_tools(
    messages: List[Dict],
    tools: List[Dict],
    tool_choice: str = "auto",
    temperature: float = 0.1,
) -> _Response
```

Tool-calling completion. Returns a `_Response` object with the same interface as the OpenAI SDK's `ChatCompletion`.

- `tools` — OpenAI-format tool definitions (list of `{"type": "function", "function": {...}}` dicts)
- `tool_choice` — `"auto"`, `"none"`, or `{"type": "function", "function": {"name": "..."}}`

**Example:**
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_glucose_reading",
        "description": "Fetch the latest CGM reading for a patient",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient identifier"}
            },
            "required": ["patient_id"]
        }
    }
}]

response = llm.chat_tools(messages, tools=tools)

# Dispatch tool calls
for tc in response.tool_calls:
    print(tc.function.name)                    # "get_glucose_reading"
    args = json.loads(tc.function.arguments)   # {"patient_id": "p001"}
    result = dispatch(tc.function.name, args)

    # Append assistant message + tool result back to messages
    messages.append(response.choices[0].message.model_dump())
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
```

---

### Response Objects

These classes mirror the OpenAI SDK types so existing notebook dispatch code works without modification.

```
_Response
└── .choices          List[_Choice]
└── .tool_calls       shortcut → choices[0].message.tool_calls

_Choice
└── .message          _Message
└── .finish_reason    "tool_calls" | "stop"

_Message
└── .content          Optional[str]
└── .role             "assistant"
└── .tool_calls       List[_ToolCall]
└── .model_dump()     → OpenAI-compatible dict (for appending to messages)

_ToolCall
└── .id               str  (e.g. "call_0" for Ollama, real ID for OpenAI)
└── .type             "function"
└── .function         _FunctionCall

_FunctionCall
└── .name             str
└── .arguments        str  (JSON-encoded, same as OpenAI)
```

**Important:** Ollama returns tool call arguments as a Python `dict`; `llm_utils` automatically serialises them to a JSON string so `.function.arguments` is always a string regardless of backend.

---

## Module-Level Exports

After import, these names are available at module level:

| Name | Type | Description |
|---|---|---|
| `llm` | `LLMClient` | The shared client instance — use this in notebooks |
| `BACKEND` | `str` | Active backend: `"ollama"` \| `"openai"` \| `"test"` |
| `MODEL` | `str` | Active model name (e.g. `"qwen2.5:7b"`, `"gpt-4o-mini"`, `"mock"`) |
| `TEST_MODE` | `bool` | `True` when running in mock mode (no real LLM calls) |

Notebooks can branch on these:

```python
from llm_utils import llm, TEST_MODE, BACKEND, MODEL

if TEST_MODE:
    print("Running in test mode — responses are mocked")
else:
    print(f"Using {BACKEND} / {MODEL}")
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (change for remote Ollama) |
| `OLLAMA_MODEL` | *(empty)* | Force a specific Ollama model by name; skips auto-selection |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key; enables OpenAI backend if Ollama unavailable |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use when OpenAI backend is active |
| `NOTEBOOK_TEST_MODE` | `0` | Set to `1` to force TEST_MODE (mock responses, no LLM needed) |

---

## Notebook Integration Pattern

All notebooks that use `llm_utils` follow this import pattern:

```python
import sys, os

# Adjust the number of '..' segments to reach the repo root from the notebook's directory.
# Notebooks at depth 2 (e.g. part3_reasoning/ch12_rag/notebook.ipynb) use '../..':
sys.path.insert(0, os.path.join(os.getcwd(), '..', '..'))

from llm_utils import llm, TEST_MODE, BACKEND, MODEL

print(f"Backend: {BACKEND} | Model: {MODEL} | Test mode: {TEST_MODE}")
```

**Depth reference:**

| Notebook location | `sys.path` insert |
|---|---|
| `part*/chXX_*/notebook.ipynb` | `'..', '..'` (2 levels up) |
| `capstone_gas/notebook.ipynb` | `'..'` (1 level up) |
| Repo root | `'.'` (already in path) |

---

## Startup Banner

When `llm_utils` is imported it prints a one-line banner to stdout:

```
🦙 LLM backend: Ollama local (qwen2.5:7b)
   ✅ Tool calling: supported | JSON mode: reliable
```

Or with a weak model:
```
🦙 LLM backend: Ollama local (llama3.2:latest)
   ⚠️  llama3.2:latest has weak JSON output. Consider: ollama pull qwen2.5:7b
```

Or in test mode:
```
🧪 LLM backend: TEST MODE (mock data)
   ℹ️  To use a real model: install Ollama + run: ollama pull qwen2.5:7b
   ℹ️  Or set OPENAI_API_KEY environment variable
```

---

## How to Add a New Model

1. Open `llm_utils.py` and find `_MODEL_PREFERENCE`.
2. Add a new tuple in the right position (higher score = higher priority):

```python
_MODEL_PREFERENCE = [
    ("qwen2.5:72b",     True,  True,  100),
    # ... existing entries ...
    ("my-new-model:7b", True,  True,   67),  # ← add here, between score 68 and 65
    # ...
]
```

3. Fields:
   - **`name_substring`** — a string that will be matched against the installed model name using `in`. Use the most specific prefix that uniquely identifies the model variant.
   - **`supports_tools`** — `True` if the model reliably handles the Ollama tools API.
   - **`json_reliable`** — `True` if the model consistently returns valid JSON when asked (without extra prose).
   - **`quality_score`** — integer; higher = preferred. Use the table above as a reference.

4. If you have a more specific variant (e.g. `"my-model:3b"`) that should rank differently from the base name (`"my-model"`), add the specific variant **before** the base name in the list.

---

## How to Add a New Backend

To add a backend (e.g. Anthropic, Groq, a local vLLM server):

1. **Add detection logic** — write a `_detect_<backend>()` function similar to `_detect_ollama()`. It should return the model name string if available, or `None`.

2. **Add a new `BACKEND` value** — update the detection block at the bottom of `llm_utils.py`:

```python
# After the Ollama check, before the OpenAI check:
_groq_model = _detect_groq()
if _groq_model:
    BACKEND = "groq"
    MODEL = _groq_model
    TEST_MODE = False
```

3. **Add backend methods to `LLMClient`** — implement `_groq_chat()`, `_groq_chat_json()`, `_groq_tools()` following the same pattern as the Ollama methods.

4. **Wire into the three public methods** — add `elif self.backend == "groq":` branches in `chat()`, `chat_json()`, and `chat_tools()`.

5. **Update the startup banner** — add entries to `_ICONS` and `_LABELS` dicts.

The key invariant to preserve: **all three public methods must return the same types regardless of backend** (`str`, `dict`, `_Response`).

---

## TEST_MODE / CI

All notebooks are tested in CI with:

```bash
NOTEBOOK_TEST_MODE=1 jupyter execute <notebook.ipynb>
```

In TEST_MODE:
- No network calls are made.
- `llm.chat()` returns deterministic strings based on keywords in the last user message (glucose levels, insulin, meals, etc.).
- `llm.chat_json()` returns a fixed JSON dict: `{"score": 4, "reasoning": "...", "is_safe": true, ...}`.
- `llm.chat_tools()` returns a mock `_Response` with one tool call using the first tool's required parameters filled with placeholder values.

The mock responses are designed to be clinically plausible for the GAS domain so that downstream notebook logic (parsing, branching, display) exercises real code paths.
