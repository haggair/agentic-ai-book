# GAS Capstone: Guardian Angel System for Diabetic Patient Supervision

> **The final deliverable of *Building Agentic AI: From First Principles to a Multi-Agent Guardian Angel System***

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

---

## ⚠️ Clinical Disclaimer

**This software is for educational and research purposes only.**
It is **NOT a medical device** and must **NOT** be used for clinical decision-making,
patient care, or any medical purpose. Always consult a qualified healthcare professional
for medical advice. The authors accept no liability for any clinical outcomes.

---

## Overview

The **Guardian Angel System (GAS)** is a complete, production-quality multi-agent AI
system for continuous supervision of diabetic patients. It demonstrates every major
concept from the book — from Markov Decision Processes and RLHF to LangGraph
orchestration and RAG — assembled into a single, coherent application.

### The Clinical Problem

Diabetes management requires continuous vigilance: blood glucose can drop dangerously
low during sleep (nocturnal hypoglycemia), spike after meals, or behave unpredictably
during exercise. Patients and caregivers cannot monitor 24/7. The GAS acts as an
always-on "guardian angel" that:

1. **Monitors** continuous glucose monitor (CGM) data every 5 minutes
2. **Detects** anomalies, dangerous trends, and clinical patterns
3. **Advises** with evidence-based, ADA-guideline-grounded recommendations
4. **Validates** every recommendation through a safety critic before delivery
5. **Escalates** to human caregivers when the situation requires it

---

## Architecture

```
GASState (TypedDict)
  patient | readings_history | current_reading | alert_level
  alert_reason | recommendation | safety_score | safety_approved
  caregiver_notified | agent_messages | next_agent | iteration

LangGraph StateGraph flow:

  START
    |
    v
  [Monitor Agent]          reads CGM data, classifies alert level & trends
    |
    v
  [Supervisor]             LangGraph router, decides next agent
    |
    v
  [Advisor Agent]          RAG-grounded, evidence-based recommendation
    |
    v
  [Safety Critic]          validates recommendation (score 0.0 - 1.0)
    |
    +-- score >= 0.7 --> [Caregiver Notifier] --> END
    |
    +-- score <  0.7 --> retry (up to 2x) --> [Advisor Agent]

Supporting services:
  CGM Simulator       5 clinical scenarios (24-hour traces)
  RAG Knowledge Base  FAISS + sentence-transformers, 15 ADA guidelines
  GASLLMClient        auto-detects OpenAI / Anthropic / Ollama / Mock
```

### Data Flow

```
CGM Sensor (5-min interval)
        │
        ▼
  GlucoseReading
  {timestamp, value_mgdl, trend, source, notes}
        │
        ▼
  Monitor Agent ──── classifies alert level (none/low/medium/high/critical)
        │            detects trends (rising_fast, falling_fast)
        │            identifies patterns (dawn, post-meal, exercise)
        ▼
  Advisor Agent ──── RAG query → retrieve 3 ADA guidelines
        │            LLM prompt with patient context + guidelines
        │            generates specific, actionable recommendation
        ▼
  Safety Critic ──── checks 5 hard safety rules
        │            scores 0.0–1.0 (threshold: 0.7)
        │            blocks dangerous advice (insulin during hypo, etc.)
        ▼
  Caregiver Notifier ── sends structured alert for high/critical events
        │                includes: patient info, glucose, trend, recommendation
        ▼
  Result stored in simulation trace
```

---

## The Five Agents

| Agent | Role | Key Logic |
|---|---|---|
| **Monitor** | Reads CGM data, classifies alerts | Threshold-based + trend detection + pattern recognition |
| **Advisor** | Generates recommendations | RAG over 15 ADA guidelines + LLM with patient context |
| **Safety Critic** | Validates recommendations | 5 hard safety rules + 0.0–1.0 scoring |
| **Caregiver Notifier** | Escalates to humans | Structured alert for high/critical events |
| **Supervisor** | Orchestrates routing | Conditional edges: monitor→advisor→safety→caregiver |

---

## Five Clinical Scenarios

| Scenario | Description | Key Event |
|---|---|---|
| `normal_day` | Typical 24-hour profile | Two post-meal bumps, stable overnight |
| `hypoglycemia_episode` | Nocturnal low | Glucose drops to ~48 mg/dL at 02:00, recovers by 03:30 |
| `post_meal_spike` | Post-prandial hyperglycemia | Glucose spikes to ~280 mg/dL after lunch |
| `dawn_phenomenon` | Early-morning rise | Counter-regulatory hormones push glucose to ~220 mg/dL |
| `exercise_induced_low` | Exercise hypoglycemia | Aerobic exercise drops glucose to ~58 mg/dL at 17:30 |

---

## How It Maps to the Book Chapters

Every component of GAS was built incrementally across the 28 chapters:

| Chapter | Concept | GAS Component Built |
|---|---|---|
| 1 | Agentic AI Landscape | Project scaffold, shared utilities |
| 2 | LLM Foundations | `PatientProfile`, `GlucoseReading` data models |
| 3 | MDPs for LLMs | `GASState` TypedDict — the MDP state representation |
| 4 | Reward Modelling | Reward model for clinical advice quality |
| 5 | PPO | PPO fine-tune loop for the Advisor agent |
| 6 | DPO | DPO alternative for Advisor; A/B comparison |
| 7 | GRPO | GRPO fine-tune for Safety Critic scoring |
| 8 | Constitutional AI | Safety Critic's 5 constitutional rules |
| 9 | Chain-of-Thought | Monitor reasoning trace for anomaly detection |
| 10 | Tree-of-Thought | Supervisor planning tree for multi-step decisions |
| 11 | Tool Use | Tool registry for Monitor (CGM API, meal DB) |
| 12 | RAG | `GASKnowledgeBase` — FAISS + sentence-transformers |
| 13 | Memory Systems | `readings_history` rolling window in GASState |
| 14 | LangChain | LangChain wiring for Monitor + Advisor |
| 15 | **LangGraph** | **`build_gas_graph()` — the full StateGraph** |
| 16 | CrewAI | CrewAI crew wrapping all 5 GAS agents |
| 17 | AutoGen | AutoGen fallback orchestration layer |
| 18 | MCP | MCP server exposing GAS tools to any LLM |
| 19 | Evaluation | Evaluation harness for Advisor responses |
| 20 | Safety & Red-Teaming | Red-team suite for Safety Critic |
| 21 | Observability | Tracing middleware for all GAS agents |
| 22 | Streaming | Real-time glucose alert streaming |
| 23 | Vector Databases | Production Qdrant setup for Advisor RAG |
| 24 | PEFT & LoRA | LoRA adapter for domain-adapted Advisor |
| 25 | Deployment | Dockerised GAS microservice |
| 26 | Human-in-the-Loop | HITL approval gate in Supervisor graph |
| 27 | Scaling | Redis Streams task queue for GAS |
| 28 | Future Directions | Research roadmap |
| ★ | **Capstone** | **Complete Guardian Angel System** |

---

## How to Run

### Prerequisites

```bash
# From the repository root
pip install -r requirements.txt

# Or install just the capstone dependencies
pip install -r capstone_gas/requirements.txt

# Copy and configure API keys (optional — mock mode works without keys)
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### Run the simulation script

```bash
# Default: hypoglycemia_episode scenario, alice patient
python capstone_gas/gas_system.py

# Specific scenario
python capstone_gas/gas_system.py --scenario post_meal_spike

# Different patient
python capstone_gas/gas_system.py --scenario dawn_phenomenon --patient bob

# Quick demo (first 48 readings = 4 hours)
python capstone_gas/gas_system.py --max-readings 48

# Run all 5 scenarios
python capstone_gas/gas_system.py --all-scenarios

# CI-safe mock mode (no API keys needed)
NOTEBOOK_TEST_MODE=1 python capstone_gas/gas_system.py
```

### Run the capstone notebook

```bash
jupyter lab
# Navigate to: capstone_gas/notebook_capstone.ipynb
```

---

## Configuration Options

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (auto-detected) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (auto-detected) |
| `NOTEBOOK_TEST_MODE` | `0` | Set to `1` for mock mode (no API calls) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### LLM Backend Auto-Detection

The system automatically selects the best available LLM backend:

```
1. OPENAI_API_KEY set    → OpenAI gpt-4o-mini
2. ANTHROPIC_API_KEY set → Anthropic claude-3-haiku-20240307
3. Ollama reachable      → Ollama llama3.2 (local, free)
4. None available        → Mock responses (deterministic, CI-safe)
```

---

## Sample Output

```
======================================================================
🏥 Guardian Angel System — Simulation
   Scenario : hypoglycemia_episode
   Patient  : Alice Chen (Age 34, T1D)
   Readings : 288 (24h at 5-min intervals)
======================================================================

  [00:00]  105.2 mg/dL  stable        Alert: NONE
  [02:00]   62.1 mg/dL  falling       Alert: HIGH     → Glucose is below target range. Apply the Rule of 15...
  [02:05]   54.8 mg/dL  falling_fast  Alert: CRITICAL → CRITICAL ALERT: Glucose is dangerously low...

🚨 GUARDIAN ANGEL SYSTEM — CRITICAL ALERT
============================================================
Patient:     Alice Chen (Age 34, T1D)
Time:        2025-08-30T02:05:00+00:00
Glucose:     54.8 mg/dL
Trend:       falling_fast
Alert Level: CRITICAL

Recommended Action:
CRITICAL ALERT: Glucose is dangerously low. Administer 15–20g
fast-acting carbohydrates immediately. If patient is unconscious,
use glucagon and call emergency services. Do NOT give insulin.

To: David Chen (spouse) <alice-caregiver@example.com>
============================================================

  [02:30]   71.4 mg/dL  rising        Alert: LOW
  [03:00]   95.2 mg/dL  stable        Alert: NONE
  [08:00]  138.7 mg/dL  rising        Alert: NONE
  ...

======================================================================
📊 Simulation Summary — hypoglycemia_episode
======================================================================
  Total readings     : 288
  Average glucose    : 112.4 mg/dL
  Time in range (TIR): 78.5% (target: >70%)
  Time below range   : 8.3% (target: <4%)
  Time above range   : 13.2%

  Alert distribution:
    none    : 198 ( 68.8%)  ██████████████████████████████████
    low     :  42 ( 14.6%)  ███████
    medium  :  28 (  9.7%)  ████
    high    :  14 (  4.9%)  ██
    critical:   6 (  2.1%)  █

  Caregiver notifications: 6
  Safety rejections      : 0
======================================================================
```

---

## Project Structure

```
capstone_gas/
├── README.md                    # This file
├── requirements.txt             # Capstone-specific dependencies
├── gas_system.py                # Complete GAS implementation (crown jewel)
│   ├── PatientProfile           # Patient data model
│   ├── GlucoseReading           # CGM reading data model
│   ├── GASState                 # LangGraph TypedDict state
│   ├── CGMSimulator             # 5-scenario CGM data generator
│   ├── GASKnowledgeBase         # RAG system (FAISS + sentence-transformers)
│   ├── GASLLMClient             # Unified LLM client (OpenAI/Anthropic/Ollama/Mock)
│   ├── monitor_node()           # Monitor Agent
│   ├── advisor_node()           # Advisor Agent
│   ├── safety_node()            # Safety Critic
│   ├── caregiver_node()         # Caregiver Notifier
│   ├── supervisor_node()        # Supervisor / Router
│   ├── build_gas_graph()        # LangGraph assembly
│   └── run_simulation()         # 24-hour simulation runner
├── notebook_capstone.ipynb      # End-to-end capstone notebook
└── figures/                     # Saved plots from notebook
```

---

## Extending GAS

### Add a new agent (e.g., Nutritionist Agent)

```python
def nutritionist_node(state: GASState) -> dict:
    """Nutritionist Agent — suggests meal adjustments."""
    glucose = state["current_reading"]["value_mgdl"]
    # ... your logic here
    return {
        "recommendation": "Consider a low-glycemic snack...",
        "agent_messages": ["[Nutritionist] Meal recommendation generated"],
    }

# Add to graph
builder.add_node("nutritionist", nutritionist_node)
builder.add_edge("advisor", "nutritionist")
builder.add_edge("nutritionist", "supervisor")
```

### Add a new tool (e.g., insulin pump control)

```python
from langchain_core.tools import tool

@tool
def adjust_basal_rate(rate_units_per_hour: float) -> str:
    """Adjust insulin pump basal rate. Use only when glucose > 200 mg/dL."""
    # ... pump API call
    return f"Basal rate adjusted to {rate_units_per_hour} U/hr"
```

### Deploy to production

```bash
# Build Docker image
docker build -t gas-system:latest .

# Run with Docker Compose (includes Qdrant + Redis)
docker compose up -d

# Health check
curl http://localhost:8000/health
```

---

## Clinical Considerations

### What GAS does well
- Continuous 24/7 monitoring without human fatigue
- Consistent application of ADA guidelines
- Rapid escalation for critical events (< 1 second)
- Audit trail of all agent decisions

### What GAS cannot do
- Replace clinical judgment
- Account for individual patient variation not in the profile
- Handle novel clinical situations outside its training data
- Guarantee recommendation accuracy (LLM hallucination risk)

### Path to clinical validation
1. IRB approval for clinical study
2. Prospective trial comparing GAS vs. standard care
3. FDA 510(k) clearance as a Software as a Medical Device (SaMD)
4. Post-market surveillance and continuous monitoring

---

## License

MIT License — see [../LICENSE](../LICENSE) for details.

The book text is © 2026 H. Roitman. The code is freely reusable under MIT.
