"""Aggregate readiness for /api/health/ready (free stack)."""

from __future__ import annotations

from typing import Any

from reflective_research.config.settings import Settings
from reflective_research.llm.ollama_health import ping_ollama


def readiness_report(settings: Settings) -> tuple[bool, dict[str, Any]]:
    """Returns (all_ok, payload). Real checks only — no paid APIs in default stack."""
    components: dict[str, Any] = {}
    ok_all = True

    o_ok, msg = ping_ollama(settings)
    v_model = (settings.ollama_verifier_model or "").strip() or settings.ollama_model
    components["ollama"] = {
        "ok": o_ok,
        "base_url": settings.ollama_base_url,
        "llm_model": settings.ollama_model,
        "verifier_model": v_model if settings.verification_enabled else None,
        "verification_enabled": settings.verification_enabled,
        "embeddings": settings.embedding_provider,
        "detail": None if o_ok else msg,
    }
    if not o_ok:
        ok_all = False

    emb_note = (
        "sentence-transformers runs locally (CPU); public weights need no Hub API key."
        if settings.embedding_provider == "huggingface"
        else "Embeddings served by Ollama; ensure `ollama pull` for your OLLAMA_EMBEDDING_MODEL."
    )
    components["embeddings_backend"] = {
        "ok": True,
        "provider": settings.embedding_provider,
        "model": (
            settings.hf_embedding_model
            if settings.embedding_provider == "huggingface"
            else settings.ollama_embedding_model
        ),
        "note": emb_note,
    }

    components["web_search"] = {
        "ok": True,
        "provider": "duckduckgo",
        "note": "No API key; respect rate limits.",
    }
    components["url_fetch"] = {
        "ok": True,
        "enabled": settings.url_fetch_enabled,
        "note": "Fetches top search URLs (trafilatura extract); blocked for localhost/private IPs.",
    }

    return ok_all, {"components": components}
