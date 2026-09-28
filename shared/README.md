# Shared Utilities — `agentic-ai-book/shared/`

Reusable Python modules used across all chapter notebooks in the book.
Import from the package root:

```python
from shared import LLMClient, GASState, PatientProfile, CGMSimulator
```

---

## Modules

### `llm_client.py` — Unified LLM Wrapper

A single `LLMClient` class that works with **OpenAI**, **Anthropic**, and
**Ollama** (local) without changing your code.

**Backend auto-detection** (in priority order):

| Environment variable | Backend selected | Default model |
|---|---|---|
| `OPENAI_API_KEY` set | OpenAI | `gpt-4o-mini` |
| `ANTHROPIC_API_KEY` set | Anthropic | `claude-3-haiku-20240307` |
| Neither set | Ollama (local) | `llama3.2` |

**Key features:**
- `client.chat(messages, temperature=0.7, max_tokens=1024) -> str`
- `client.count_message_tokens(messages) -> int`
- Retry logic with exponential backoff (3 attempts, uses `tenacity` when installed)
- `NOTEBOOK_TEST_MODE=1` → returns deterministic mock responses (CI-safe)

**Quick start:**

```python
import os
os.environ["NOTEBOOK_TEST_MODE"] = "1"   # skip real API calls

from shared import LLMClient

client = LLMClient()
reply = client.chat([
    {"role": "system", "content": "You are a medical assistant."},
    {"role": "user",   "content": "What is hypoglycemia?"},
])
print(reply)
```

**Optional dependencies:** `openai`, `anthropic`, `requests` (Ollama),
`tenacity` (retry), `tiktoken` (accurate token counting).

---

### `gas_state.py` — Guardian Angel System State Schema

Defines the data structures that flow through the GAS multi-agent pipeline.

#### Dataclasses

| Class | Purpose |
|---|---|
| `PatientProfile` | Static demographics, target ranges, medications, allergies |
| `GlucoseReading` | Single CGM/manual measurement with trend and status |
| `GASMemory` | Rolling 24-hour history: readings, episodes, medication log, contacts |
| `MedicationLogEntry` | Single medication administration record |
| `Episode` | Recorded clinical event (hypo/hyperglycemia crisis) |
| `CaregiverContact` | Person to notify on alert |

#### `GASState` TypedDict (LangGraph state)

```python
state: GASState = {
    "patient":            PatientProfile(...),
    "current_reading":    GlucoseReading(...),
    "memory":             GASMemory(...),
    "alert_level":        "high",          # none/low/medium/high/critical
    "recommendation":     "Drink 15g fast carbs immediately.",
    "safety_approved":    False,
    "caregiver_notified": False,
    "agent_messages":     [],
    "next_agent":         "safety_check",
    "metadata":           {},
}
```

#### Helper functions

```python
from shared import create_default_patient, glucose_status, alert_level

patient = create_default_patient()          # realistic demo patient

reading = GlucoseReading(value_mgdl=52.0, trend="falling")
print(glucose_status(reading))   # → "hypoglycemia"
print(alert_level(reading))      # → "critical"
```

**Alert level thresholds (mg/dL):**

| Level | Low threshold | High threshold |
|---|---|---|
| `critical` | < 54 | > 400 |
| `high` | < 70 | > 300 |
| `medium` | < 80 | > 250 |
| `low` | < 90 | > 200 |
| `none` | 90–200 | — |

---

### `patient_data.py` — Synthetic CGM Data Generator

Generates realistic glucose time series for demos and testing — no real
patient data required.

#### `CGMSimulator`

```python
from shared import CGMSimulator
from datetime import date

sim = CGMSimulator(patient_id="demo-001", diabetes_type=1, seed=42)

# Generate a full day (288 readings × 5-minute intervals)
readings = sim.generate_day(
    date(2024, 1, 15),
    meal_times=[7, 12, 18],
    exercise_events=[15],
)

# Or use a pre-built scenario
readings = sim.generate_scenario("hypoglycemia_episode")

# Summary statistics
stats = sim.summary_stats(readings)
print(f"Mean glucose: {stats['mean_glucose']} mg/dL")
print(f"Time in range: {stats['time_in_range']:.1%}")

# Plot (requires matplotlib)
sim.plot_glucose(readings, title="Hypoglycemia Episode")

# Export to pandas DataFrame
df = sim.to_dataframe(readings)
```

#### Pre-built scenarios

| Scenario name | Description |
|---|---|
| `"normal_day"` | Well-controlled day, three meals, glucose mostly 70–180 |
| `"hypoglycemia_episode"` | Dangerous nocturnal low at ~3 am (nadir ≈ 48 mg/dL) + Somogyi rebound |
| `"post_meal_spike"` | Severe hyperglycemia after large lunch (missed bolus, peak > 300) |
| `"dawn_phenomenon"` | Early-morning cortisol-driven glucose rise (4–8 am) |
| `"exercise_induced_low"` | Glucose drop during afternoon exercise + delayed low risk |

#### Type 1 vs Type 2 differences

| Parameter | Type 1 | Type 2 |
|---|---|---|
| Fasting baseline | ~105 mg/dL | ~115 mg/dL |
| Post-meal spike | 70–130 mg/dL | 50–90 mg/dL |
| Rise duration | 50 min | 75 min |
| Decay duration | 100 min | 150 min |
| Sensor noise | ±7 mg/dL | ±4 mg/dL |

**Optional dependencies:** `matplotlib` (plotting), `pandas` (DataFrame export).

---

## Installation

All shared utilities are pure Python with optional dependencies.
Install the full set for all features:

```bash
pip install openai anthropic requests tenacity tiktoken matplotlib pandas
```

For local Ollama support, install and run [Ollama](https://ollama.ai):

```bash
ollama pull llama3.2
ollama serve
```

---

## Environment Variables

| Variable | Effect |
|---|---|
| `OPENAI_API_KEY` | Enables OpenAI backend |
| `ANTHROPIC_API_KEY` | Enables Anthropic backend |
| `NOTEBOOK_TEST_MODE` | Any non-empty value → mock LLM responses (no API calls) |

---

## Chapter Usage Map

| Chapter | Modules used |
|---|---|
| Part 1 — Foundations | `llm_client` |
| Part 2 — RL Methods | `llm_client`, `patient_data` |
| Part 3 — Reasoning | `llm_client`, `gas_state`, `patient_data` |
| Part 4 — Evaluation | `llm_client`, `gas_state`, `patient_data` |
| Part 5 — Agentic | All modules |
| Capstone GAS | All modules |
