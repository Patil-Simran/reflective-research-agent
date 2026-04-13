# Free stack — APIs and keys

This project uses **one fixed, $0 stack**. **You do not sign up for any paid inference API** for the default setup.

## Do you need API keys?

| Service | API key? | What you do |
|---------|----------|-------------|
| **Ollama** (LLM) | **No** | Install the app, run it, `ollama pull` the chat model. |
| **Hugging Face** (embeddings) | **No** (public model) | Pulled with `pip`; model `sentence-transformers/all-MiniLM-L6-v2` runs locally on CPU. |
| **Chroma** (vector DB) | **No** | Files on disk under `CHROMA_PERSIST_DIR`. |
| **DuckDuckGo** (web search) | **No** | Used from Python; respect normal rate limits. |
| **FastAPI** (`research serve`) | **No** | Local HTTP; add your own auth in production if you expose it. |

**Total third-party API keys required for this stack: 0.**

If you use **Ollama Cloud** or a password-protected Ollama server, set the environment variable **`OLLAMA_API_KEY`** yourself (standard Ollama client behaviour). The app does not fetch it for you.

---

## Fixed Gen AI stack (implemented)

| Layer | Technology |
|-------|------------|
| Orchestration | **LangGraph** + **LangChain** |
| LLM | **Ollama** (`ChatOllama`) |
| Embeddings | **sentence-transformers** via **LangChain HuggingFaceEmbeddings** (CPU) |
| RAG store | **Chroma** (persistent) |
| Web retrieval | **duckduckgo-search** |
| HTTP API | **FastAPI** + **Uvicorn** |
| UI | **React** + **Vite** + **Tailwind** |

---

## Environment variables (`.env`)

Copy `.env.example` → `.env`. Everything there is **configuration** (URLs, paths, model names), not secret keys for the default stack.

| Variable | Role |
|----------|------|
| `OLLAMA_BASE_URL` | Where the Ollama daemon listens (default `http://127.0.0.1:11434`). |
| `OLLAMA_MODEL` | Chat model name **exactly** as in `ollama list` (e.g. `llama3.2:latest`). A typo or missing pull → **404**. |
| `EMBEDDING_PROVIDER` | Use `huggingface` (default) or `ollama` if you want embeddings via Ollama instead. |
| `HF_EMBEDDING_MODEL` | HuggingFace model id for vectors (default `sentence-transformers/all-MiniLM-L6-v2`). |
| `CHROMA_*` | Local DB location and collection name. |

---

## Backend routes (local)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Process is up. |
| GET | `/api/health/ready` | Ollama reachable + embedding backend OK. **503** if not. |
| POST | `/api/research` | JSON `{"query":"..."}`. |
| POST | `/api/research/stream` | Same, **SSE** stream. |

---

## Frontend (`web/`)

| Variable | Purpose |
|----------|---------|
| `VITE_DEV_PROXY_TARGET` | Dev proxy to FastAPI (default `http://127.0.0.1:8000`). |
| `VITE_API_BASE_URL` | Production: full API origin if UI is on another host. |

---

## Checklist before running

1. `pip install -e ".[web,dev]"` (or `pip install -r requirements-full.txt` + `pip install -e . --no-deps`).
2. **Ollama** running; `ollama pull <OLLAMA_MODEL>` (see `.env`).
3. `.env` from `.env.example`.
4. `GET http://127.0.0.1:8000/api/health/ready` → `"ready": true`.

### Windows `WinError 10061`

Ollama is not running or `OLLAMA_BASE_URL` is wrong. Start Ollama from the Start menu, then retry.

### `model '…' not found` (HTTP 404)

Ollama is up but that name is not installed. Run `ollama list` — if empty, pull the model:

```bash
ollama pull llama3.2
```

Then set `OLLAMA_MODEL` to the **exact** name shown (often `llama3.2:latest`). If you use `EMBEDDING_PROVIDER=ollama`, also `ollama pull nomic-embed-text` (or your `OLLAMA_EMBEDDING_MODEL`). Restart `research serve` after changing `.env`.
