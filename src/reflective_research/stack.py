"""
Canonical free Gen AI stack for this project (implemented across the codebase).

- **Orchestration:** LangGraph + LangChain
- **LLM:** Ollama (``ChatOllama``) — no external inference API key
- **Embeddings:** HuggingFace sentence-transformers (default, local CPU);
  or Ollama if ``EMBEDDING_PROVIDER=ollama``
- **RAG store:** Chroma (persistent directory)
- **Web retrieval:** DuckDuckGo instant + Wikipedia + library merge; optional **URL fetch** (trafilatura)
- **Gather:** parallel plan steps; **evidence rerank** (query overlap) before LLM reads
- **HTTP:** FastAPI + Uvicorn
- **UI:** React + Vite + Tailwind

Authoritative checklist for operators: ``CREDENTIALS.md`` at repo root.
"""

__all__: list[str] = []
