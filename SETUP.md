# Setup Guide

This guide walks you through every step needed to run the notebooks in this repository, from a fresh machine to a working Jupyter environment.

---

## Table of Contents

1. [Python Environment](#1-python-environment)
2. [API Keys (.env)](#2-api-keys-env)
3. [Ollama — Free Local Alternative](#3-ollama--free-local-alternative)
4. [Docker Services (Qdrant + Redis)](#4-docker-services-qdrant--redis)
5. [Per-Chapter Requirements](#5-per-chapter-requirements)
6. [GPU Setup for Fine-Tuning (Chapter 24)](#6-gpu-setup-for-fine-tuning-chapter-24)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Python Environment

### Option A — conda (recommended)

```bash
# Create a dedicated environment
conda create -n agentic-ai python=3.11 -y
conda activate agentic-ai

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — venv

```bash
# Requires Python 3.10, 3.11, or 3.12 on your PATH
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

### Verify the installation

```bash
python - <<'EOF'
import openai, langchain, langgraph, crewai, qdrant_client
print("All core packages imported successfully ✓")
EOF
```

---

## 2. API Keys (.env)

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

Then open `.env` in your editor:

```
OPENAI_API_KEY=sk-...          # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com/
```

The notebooks load these automatically via `python-dotenv`. You never need to hard-code a key.

### Which keys do I actually need?

| Chapters | Minimum required |
|---|---|
| 1–13 (foundations) | `OPENAI_API_KEY` **or** Ollama (see §3) |
| 14–18 (frameworks) | `OPENAI_API_KEY` **or** Ollama |
| 19–21 (evaluation) | `OPENAI_API_KEY` (LLM-as-judge calls) |
| 24 (fine-tuning) | No API key needed — runs locally |
| Capstone GAS | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` recommended |

---

## 3. Ollama — Free Local Alternative

[Ollama](https://ollama.com) lets you run open-weight models (Llama 3, Mistral, Phi-3, Gemma 2, etc.) entirely on your laptop — no API key, no cost, no data leaving your machine.

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download the installer from https://ollama.com/download
```

### Pull a model

```bash
# Llama 3.1 8B — good balance of quality and speed (4.7 GB)
ollama pull llama3.1

# Mistral 7B — fast, great for tool use
ollama pull mistral

# Phi-3 Mini — runs on 8 GB RAM
ollama pull phi3:mini
```

### Start the Ollama server

```bash
ollama serve
# Runs on http://localhost:11434 by default
```

### Configure the notebooks to use Ollama

In your `.env` file:

```
OLLAMA_BASE_URL=http://localhost:11434
```

All notebooks check for `OLLAMA_BASE_URL` and fall back to Ollama automatically when `OPENAI_API_KEY` is not set. You can also force Ollama by setting:

```
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
```

### Ollama with LangChain

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")
response = llm.invoke("Explain hypoglycaemia in one sentence.")
print(response.content)
```

---

## 4. Docker Services (Qdrant + Redis)

Chapters 12, 13, 23, 27, and the capstone require a running Qdrant vector database and Redis instance. The easiest way is Docker Compose:

```bash
# Start both services in the background
docker compose up -d qdrant redis

# Verify they are healthy
docker compose ps

# Check Qdrant dashboard
open http://localhost:6333/dashboard

# Stop services when done
docker compose down
```

### Without Docker

**Qdrant** — install the binary directly:

```bash
# macOS (Homebrew)
brew install qdrant/tap/qdrant
qdrant  # starts on port 6333

# Or run via pip (in-memory, for testing only)
pip install qdrant-client[fastembed]
```

**Redis** — install via Homebrew or apt:

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu / Debian
sudo apt install redis-server
sudo systemctl start redis
```

---

## 5. Per-Chapter Requirements

Most chapters use the shared `requirements.txt`. A few have heavier dependencies that are optional unless you are working on that chapter:

| Chapter | Extra package | Install command |
|---|---|---|
| 5 — PPO | `trl`, `accelerate` | Already in requirements.txt |
| 6 — DPO | `trl` | Already in requirements.txt |
| 7 — GRPO | `trl>=0.9.0` | Already in requirements.txt |
| 16 — CrewAI | `crewai[tools]` | `pip install "crewai[tools]"` |
| 17 — AutoGen | `pyautogen` | `pip install pyautogen` |
| 24 — QLoRA | `bitsandbytes` | `pip install bitsandbytes` (Linux/CUDA only) |
| 25 — Deployment | `fastapi`, `uvicorn` | Already in requirements.txt |
| Capstone | All of the above | `pip install -r requirements.txt` |

### Chapter-level `requirements.txt` files

Each chapter directory may contain its own `requirements.txt` with pinned versions for reproducibility. Install them with:

```bash
pip install -r part2_rl_methods/ch05_ppo/requirements.txt
```

---

## 6. GPU Setup for Fine-Tuning (Chapter 24)

Chapter 24 (PEFT / QLoRA) benefits greatly from a CUDA GPU. On CPU it will run but slowly.

### Check your GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### Install CUDA-enabled PyTorch (if needed)

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (M1/M2/M3)

PyTorch MPS backend works for most operations:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

`bitsandbytes` (4-bit quantisation) does **not** support MPS. Use `load_in_8bit=False` and reduce batch size instead.

---

## 7. Troubleshooting

### `ModuleNotFoundError` after `pip install -r requirements.txt`

Make sure you activated your virtual environment first:

```bash
conda activate agentic-ai   # or: source .venv/bin/activate
which python                 # should point inside your env
```

### Jupyter kernel does not see installed packages

Register the environment as a Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name agentic-ai --display-name "Agentic AI"
```

Then select **Agentic AI** from the kernel picker in JupyterLab.

### `openai.AuthenticationError`

- Check that `.env` exists and contains a valid `OPENAI_API_KEY`
- Ensure `python-dotenv` is installed and the notebook calls `load_dotenv()` at the top
- Verify the key has not expired or been revoked at https://platform.openai.com/api-keys

### Qdrant connection refused

```bash
# Check if the container is running
docker compose ps

# Restart if needed
docker compose restart qdrant

# Check logs
docker compose logs qdrant
```

### Redis connection refused

```bash
docker compose restart redis
redis-cli ping   # should return PONG
```

### `torch` import hangs on macOS

This is a known issue with some macOS + Python 3.12 combinations. Downgrade to Python 3.11:

```bash
conda create -n agentic-ai python=3.11 -y
conda activate agentic-ai
pip install -r requirements.txt
```

### Ollama model not found

```bash
# List downloaded models
ollama list

# Pull the model the notebook expects
ollama pull llama3.1
```

### nbmake test failures in CI

Run locally first to see the full traceback:

```bash
pytest --nbmake part1_foundations/ch01_agentic_landscape/notebook.ipynb -v
```

Common causes: missing `.env` variables, Docker services not running, or a cell that takes longer than the 120 s timeout.

---

## Getting Help

- Open a [GitHub Issue](https://github.com/yourusername/agentic-ai-book/issues) for bugs or setup problems
- Check the book's companion website for errata and updates
- Join the community Discord (link in the book's preface)
