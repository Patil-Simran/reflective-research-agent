# Reflective Research Agent

A LangGraph-based research assistant that plans retrieval, gathers evidence from web search and optional Chroma RAG, reflects on coverage, and synthesizes a cited report. Ships with a FastAPI backend (REST + SSE) and a React (Vite) web UI.

## Features

- **Orchestrated pipeline:** plan → gather → reflect → evidence brief → synthesize → optional verification and revision.
- **Retrieval:** DuckDuckGo and related providers, optional URL fetching, configurable Chroma vector store for local documents.
- **Local inference:** Ollama for chat; embeddings via Hugging Face sentence-transformers (CPU) or Ollama, configurable in `.env`.
- **Interfaces:** `research` CLI, HTTP API, browser UI with streaming progress and Markdown output.

## Architecture

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph, LangChain |
| LLM | Ollama (`ChatOllama`) |
| Embeddings | sentence-transformers / HuggingFace or Ollama |
| Vector store | Chroma (persistent) |
| Web search | duckduckgo-search and integrated providers |
| API | FastAPI, Uvicorn |
| Frontend | React 19, Vite, Tailwind |

## Prerequisites

- **Python** 3.11, 3.12, or 3.13 (3.11–3.12 recommended for dependency wheels).
- **Node.js** 18+ (for the `web/` application).
- **Ollama** installed and running for chat models ([ollama.com](https://ollama.com)).

## Installation

Clone the repository and install the Python package in a virtual environment from the repository root (directory containing `pyproject.toml`).

**Windows (PowerShell):**

```powershell
cd reflective-research-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[web,dev]"
```

**macOS / Linux:**

```bash
cd reflective-research-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[web,dev]"
```

**Alternative:** install from lock-style lists, then the package:

```bash
pip install -r requirements-full.txt
pip install -e . --no-deps
```

Dependencies are defined in `pyproject.toml`. The `requirements.txt` and `requirements-full.txt` files list pinned-style requirements for the agent runtime and for API/dev tooling respectively.

## Configuration

1. Copy `.env.example` to `.env`.
2. Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` to match your Ollama deployment (on Windows, `http://127.0.0.1:11434` is often more reliable than `localhost`).
3. Set `EMBEDDING_PROVIDER` to `huggingface` or `ollama` and align model names with what you have pulled.

See `.env.example` for all options. For a concise reference on external services and variables, see [CREDENTIALS.md](CREDENTIALS.md).

## Usage

### Pull Ollama models

Match models to `.env` (example chat model):

```bash
ollama pull llama3.2
```

If `EMBEDDING_PROVIDER=ollama`, pull the configured embedding model as well (e.g. `nomic-embed-text`).

### Run the HTTP API

```bash
research serve
```

Default bind: `http://127.0.0.1:8000`. First startup may take one to two minutes while dependencies and the graph load.

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Liveness |
| `GET /api/health/ready` | Readiness (Ollama and configuration) |
| `POST /api/research` | Synchronous research run |
| `POST /api/research/stream` | SSE stream of progress and result |

### Run the web UI

```bash
cd web
npm install
npm run dev
```

Open the URL shown by Vite (typically `http://127.0.0.1:5173`). The dev server proxies `/api` to `http://127.0.0.1:8000`. Override with `VITE_DEV_PROXY_TARGET` in `web/.env` (see `web/.env.example`).

### CLI

```bash
research run "Your research question"
research ingest path/to/documents_or_files
```

## Development

```bash
pytest tests -q
ruff check src tests
```

Frontend production build:

```bash
cd web
npm run build
```

## Repository layout

```
reflective-research-agent/
├── src/reflective_research/   # Application package
├── web/                         # React frontend
├── tests/                       # Pytest suite
├── pyproject.toml
├── requirements.txt
├── requirements-full.txt
├── .env.example
├── CREDENTIALS.md               # Environment and third-party notes
└── README.md
```

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `WinError 10061` connecting to Ollama | Ollama not running or incorrect `OLLAMA_BASE_URL`. |
| Slow first RAG query | Initial download of Hugging Face embedding weights. |
| `chromadb` install failure | Incompatible Python version; use 3.11 or 3.12. |
| UI shows backend unavailable | API not listening yet; wait for Uvicorn startup, then check `GET /api/health/ready`. |
| Mermaid blocks not rendering inline | Model output may be invalid Mermaid; the UI shows the raw source only (no external diagram images). |


