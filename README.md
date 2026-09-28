# Building Agentic AI: From First Principles to a Multi-Agent Guardian Angel System

> **Companion code repository for the book *[The Hitchhiker's Guide to Agentic AI: From Foundations to Systems](https://arxiv.org/abs/2606.24937)* — H. Roitman (2026)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange.svg)](https://jupyter.org/)
[![nbmake](https://img.shields.io/badge/tested%20with-nbmake-green.svg)](https://github.com/treebeardtech/nbmake)

---

## Overview

This repository is the hands-on companion to the book. It provides a working notebook for each of the 28 code chapters, plus shared utilities and a full capstone project. The code follows an **incremental build philosophy**: each chapter introduces one new concept and adds one concrete piece to a running capstone project — the **Guardian Angel System (GAS)** — so that by the final chapter you have a fully operational, production-quality multi-agent AI application.

You do not need to read the chapters in order to run the notebooks, but doing so will give you the deepest understanding of *why* each design decision was made. Every notebook is self-contained, annotated, and tested in CI.

---

## The Guardian Angel System (GAS) — Capstone Project

The GAS is a **multi-agent AI system for continuous supervision of diabetic patients**. It monitors real-time glucose readings, activity data, and meal logs; reasons about risk; provides personalised advice; enforces safety constraints; and escalates to caregivers when needed — all without requiring constant human oversight.

### Origin Story

The "Guardian Angel" concept was first articulated in a landmark 1994 MIT technical report by Szolovits, Doyle, Long, Kohane, and Pauker — *"Guardian Angel: Patient-Centered Health Information Systems"* (MIT LCS TR-604) — which envisioned lifelong, patient-centred AI agents that would monitor health, reason about risk, and coordinate care autonomously. It was a visionary blueprint, decades ahead of the technology needed to realise it.

More than two decades ago, as an undergraduate student in Information Systems Engineering, the author of this book was inspired by this vision and explored it as a student project — implemented in **JADE** (Java Agent DEvelopment Framework), a FIPA-compliant multi-agent platform. That experience with structured agent communication protocols, ontology-based knowledge representation, and rule-based reasoning planted the seed for everything in this book.

The version you will build here is the 2026 realisation of that same vision: inspired by Szolovits et al.'s original concept, now powered by large language models, reinforcement learning, retrieval-augmented generation, and modern orchestration frameworks. The original idea was right — only the tools have changed.

### Five Specialised Agents

| Agent | Role |
|---|---|
| **Supervisor** | Orchestrates the other agents; decides which agent acts next based on patient state |
| **Monitor** | Ingests sensor streams (CGM, activity, meals); detects anomalies and trend shifts |
| **Advisor** | Generates personalised, evidence-based recommendations using RAG over medical literature |
| **Safety Critic** | Evaluates every proposed action against hard safety rules; blocks unsafe outputs |
| **Caregiver Notifier** | Composes and dispatches alerts to family members, nurses, or emergency services |

### GAS Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Guardian Angel System                        │
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │  Supervisor │────>│   Monitor   │────>│   Advisor   │       │
│   │   (LangGraph│<────│  (Streaming │     │  (RAG + LLM)│       │
│   │    Router)  │     │   + Tools)  │     │             │       │
│   └──────┬──────┘     └─────────────┘     └──────┬──────┘       │
│          │                                        │             │
│          ▼                                        ▼             │
│   ┌─────────────┐                        ┌─────────────┐        │
│   │   Safety    │<───────────────────────│  Caregiver  │        │
│   │   Critic    │  (blocks unsafe msgs)  │  Notifier   │        │
│   │  (RLHF/PPO) │                        │  (Email/SMS)│        │
│   └─────────────┘                        └─────────────┘        │
│                                                                 │
│   ─────────────────── Shared Services ──────────────────────    │
│   │  Qdrant Vector DB  │  Redis State  │  MCP Tool Server  │    │
│   └────────────────────┴───────────────┴───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chapter Table

| # | Title | Key Concept | GAS Piece Built |
|---|-------|-------------|-----------------|
| 1 | The Agentic AI Landscape | Agents, environments, reward signals | Project scaffold & shared utilities |
| 2 | LLM Foundations for RL | Tokenisation, next-token prediction, RLHF overview | Patient data schema & mock sensor stream |
| 3 | Markov Decision Processes for LLMs | MDPs, policies, value functions | State representation for glucose readings |
| 4 | Reward Modelling | Reward hacking, Bradley-Terry, preference datasets | Reward model for clinical advice quality |
| 5 | Proximal Policy Optimisation (PPO) | PPO algorithm, clipping, KL penalty | PPO fine-tune loop for the Advisor agent |
| 6 | Direct Preference Optimisation (DPO) | DPO loss, reference model, β parameter | DPO alternative for Advisor; A/B comparison |
| 7 | GRPO and Group Relative Rewards | GRPO, group baselines, variance reduction | GRPO fine-tune for Safety Critic scoring |
| 8 | Constitutional AI & RLAIF | Critique-revision, AI feedback loops | Safety Critic constitutional rules |
| 9 | Reasoning Traces and Chain-of-Thought | CoT, scratchpads, process reward models | Monitor reasoning trace for anomaly detection |
| 10 | Tree-of-Thought and Search | ToT, beam search, MCTS for LLMs | Supervisor planning tree for multi-step decisions |
| 11 | Tool Use and Function Calling | OpenAI tools API, JSON schema, error handling | Tool registry for Monitor (CGM API, meal DB) |
| 12 | Retrieval-Augmented Generation | Dense retrieval, re-ranking, Qdrant | Advisor RAG pipeline over medical literature |
| 13 | Memory Systems for Agents | Episodic, semantic, working memory | Shared Redis memory layer for all agents |
| 14 | LangChain Fundamentals | Chains, runnables, LCEL | LangChain wiring for Monitor + Advisor |
| 15 | LangGraph: Stateful Agent Graphs | Nodes, edges, conditional routing, checkpoints | Supervisor graph with LangGraph |
| 16 | CrewAI: Role-Based Multi-Agent Teams | Crews, roles, tasks, delegation | CrewAI crew wrapping all 5 GAS agents |
| 17 | AutoGen: Conversational Multi-Agent | GroupChat, AssistantAgent, UserProxy | AutoGen fallback orchestration layer |
| 18 | Model Context Protocol (MCP) | MCP spec, tool servers, resources | MCP server exposing GAS tools to any LLM |
| 19 | Evaluation Frameworks | LLM-as-judge, G-Eval, RAGAS, FactScore | Evaluation harness for Advisor responses |
| 20 | Safety, Alignment, and Red-Teaming | Jailbreaks, prompt injection, adversarial inputs | Red-team suite for Safety Critic |
| 21 | Observability and Tracing | LangSmith, OpenTelemetry, structured logging | Tracing middleware for all GAS agents |
| 22 | Streaming and Real-Time Agents | SSE, WebSockets, async generators | Real-time glucose alert streaming |
| 23 | Vector Databases Deep Dive | HNSW, quantisation, filtering, Qdrant collections | Production Qdrant setup for Advisor RAG |
| 24 | Fine-Tuning with PEFT and LoRA | LoRA, QLoRA, adapter merging | LoRA adapter for domain-adapted Advisor |
| 25 | Deploying Agents to Production | FastAPI, Docker, health checks, rolling updates | Dockerised GAS microservice deployment |
| 26 | Human-in-the-Loop Patterns | Interrupt nodes, approval workflows, audit logs | HITL approval gate in Supervisor graph |
| 27 | Scaling Multi-Agent Systems | Horizontal scaling, message queues, idempotency | Redis Streams task queue for GAS |
| 28 | Future Directions | World models, embodied agents, AGI safety | Research roadmap notebook |
| ★ | **Capstone: Full GAS Integration** | End-to-end multi-agent system | **Complete Guardian Angel System** |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/haggair/agentic-ai-book.git
cd agentic-ai-book

# 2. Create and activate a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your LLM backend (choose one)

# Option A — Ollama (recommended, free, local, no API key needed):
#   Install Ollama from https://ollama.com, then:
ollama pull qwen2.5:7b          # ~4.7 GB — best quality/speed balance
#   llm_utils.py will detect it automatically on notebook import.

# Option B — OpenAI API:
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
# llm_utils.py will use OpenAI if Ollama is not running.

# Option C — No setup (CI / quick demo):
# Set NOTEBOOK_TEST_MODE=1 to run all notebooks with mock responses.

# 5. Start supporting services (Qdrant + Redis)
docker compose up -d qdrant redis

# 6. Launch Jupyter
jupyter lab

# 7. Open the first notebook
# Navigate to: part1_foundations/ch01_agentic_landscape/notebook.ipynb
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | 3.11 recommended |
| pip | ≥ 24.0 | `pip install --upgrade pip` |
| Docker Desktop | ≥ 4.25 | For Qdrant & Redis |
| Git | ≥ 2.40 | — |
| **Ollama** (recommended) | ≥ 0.3 | Free, local LLM — install from [ollama.com](https://ollama.com); pull `qwen2.5:7b` |
| OpenAI API key *(optional)* | — | Fallback if Ollama is not running; set `OPENAI_API_KEY` |
| 16 GB RAM | — | 32 GB recommended for local models |
| GPU (optional) | CUDA 12+ | Required for Chapter 24 QLoRA fine-tuning |

---

## LLM Backend

All notebooks share a single utility — **`llm_utils.py`** — that auto-selects the best available LLM with zero configuration:

| Priority | Backend | When used |
|---|---|---|
| 1 | **Ollama (local)** | Ollama is running and has at least one model installed |
| 2 | **OpenAI API** | `OPENAI_API_KEY` is set and Ollama is not available |
| 3 | **TEST_MODE (mock)** | Neither is available; or `NOTEBOOK_TEST_MODE=1` |

**Recommended setup** — install [Ollama](https://ollama.com) and pull the sweet-spot model:

```bash
ollama pull qwen2.5:7b   # 4.7 GB — fast, reliable JSON + tool calling
```

That's it. Every notebook will detect it automatically on import.

See **[LLM_UTILS.md](LLM_UTILS.md)** for the full design doc: model preference table, API reference, environment variables, and how to add new models or backends.

---

## Repository Structure

```
agentic-ai-book/
├── README.md
├── LLM_UTILS.md                     # llm_utils.py design & API reference
├── SETUP.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── .gitignore
├── LICENSE
├── llm_utils.py                     # ← Shared LLM client (auto-selects Ollama / OpenAI / mock)
│
├── shared/                          # Utilities shared across all chapters
│   ├── gas_types.py                 # Pydantic models for GAS data structures
│   ├── sensor_stream.py             # Mock CGM / activity sensor simulator
│   ├── vector_store.py              # Qdrant helper
│   └── tracing.py                   # OpenTelemetry setup
│
├── part1_foundations/               # Chapters 1–4
│   ├── ch01_agentic_landscape/
│   ├── ch02_llm_foundations/
│   ├── ch03_mdp_for_llms/
│   └── ch04_reward_modelling/
│
├── part2_rl_methods/                # Chapters 5–8
│   ├── ch05_ppo/
│   ├── ch06_dpo/
│   ├── ch07_grpo/
│   └── ch08_constitutional_ai/
│
├── part3_reasoning/                 # Chapters 9–13
│   ├── ch09_chain_of_thought/
│   ├── ch10_tree_of_thought/
│   ├── ch11_tool_use/
│   ├── ch12_rag/
│   └── ch13_memory/
│
├── part4_evaluation/                # Chapters 19–21
│   ├── ch19_evaluation/
│   ├── ch20_safety_redteam/
│   └── ch21_observability/
│
├── part5_agentic/                   # Chapters 14–18, 22–28
│   ├── ch14_langchain/
│   ├── ch15_langgraph/
│   ├── ch16_crewai/
│   ├── ch17_autogen/
│   ├── ch18_mcp/
│   ├── ch22_streaming/
│   ├── ch23_vector_db/
│   ├── ch24_peft_lora/
│   ├── ch25_deployment/
│   ├── ch26_hitl/
│   ├── ch27_scaling/
│   └── ch28_future/
│
└── capstone_gas/                    # Full Guardian Angel System
    ├── agents/
    │   ├── supervisor.py
    │   ├── monitor.py
    │   ├── advisor.py
    │   ├── safety_critic.py
    │   └── caregiver_notifier.py
    ├── services/
    │   ├── vector_store.py
    │   ├── memory.py
    │   └── mcp_server.py
    ├── gas_app.py                   # Streamlit demo UI
    ├── gas_graph.py                 # LangGraph orchestration
    └── notebook.ipynb               # End-to-end walkthrough
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository and create a feature branch (`git checkout -b feat/your-feature`)
2. Ensure all notebooks pass `pytest --nbmake` before opening a PR
3. Follow the existing notebook style: one concept per cell, prose before code
4. Open an issue first for large changes

See [SETUP.md](SETUP.md) for the full development environment guide.

---

## Citing This Work

If you use this code or the book in your research, please cite:

The capstone project is inspired by the original Guardian Angel concept:

```bibtex
@techreport{szolovits1994guardian,
  title     = {Guardian Angel: Patient-Centered Health Information Systems},
  author    = {Szolovits, Peter and Doyle, Jon and Long, William J. and Kohane, Isaac and Pauker, Stephen G.},
  year      = {1994},
  number    = {TR-604},
  institution = {MIT Laboratory for Computer Science},
  url       = {https://groups.csail.mit.edu/medg/projects/ga/manifesto/GAtr.html}
}
```

```bibtex
@book{roitman2026hitchhiker,
  title     = {The Hitchhiker's Guide to Agentic AI: From Foundations to Systems},
  author    = {Roitman, Haggai},
  year      = {2026},
  publisher = {arXiv},
  url       = {https://arxiv.org/abs/2606.24937},
  note      = {arXiv:2606.24937}
}
```

For the companion code repository specifically:

```bibtex
@misc{roitman2026hitchhiker-code,
  title     = {Building Agentic AI: Companion Code for The Hitchhiker's Guide to Agentic AI},
  author    = {Roitman, Haggai},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/haggair/agentic-ai-book}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The book text itself is © 2026 H. Roitman. The code in this repository is freely reusable under MIT.
