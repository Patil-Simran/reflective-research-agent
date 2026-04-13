"""Fail fast with a clear message when Ollama is required but not reachable."""

from __future__ import annotations

import logging

import httpx

from reflective_research.config.settings import Settings

log = logging.getLogger(__name__)


def ollama_tags_url(settings: Settings) -> str:
    base = settings.ollama_base_url.rstrip("/")
    return f"{base}/api/tags"


def ping_ollama(settings: Settings) -> tuple[bool, str]:
    """
    Returns (True, "") if Ollama answers.
    Returns (False, human_message) if unreachable.
    """
    url = ollama_tags_url(settings)
    base = settings.ollama_base_url.rstrip("/")
    try:
        r = httpx.get(url, timeout=3.0)
        r.raise_for_status()
        return True, ""
    except httpx.ConnectError as e:
        msg = (
            f"Cannot connect to Ollama at {base} (connection refused — WinError 10061 on Windows). "
            "Fix: install Ollama from https://ollama.com, start the application, then run:\n"
            f"  ollama pull {settings.ollama_model}\n"
            "If Ollama runs on another host/port, set OLLAMA_BASE_URL in .env. Details: "
            f"{e}"
        )
        return False, msg
    except httpx.HTTPError as e:
        return False, f"Ollama at {base} error: {e}"


def assert_ollama_reachable(settings: Settings) -> None:
    """LLM always needs Ollama; always verify daemon is up."""
    ok, msg = ping_ollama(settings)
    if not ok:
        log.error(msg)
        raise RuntimeError(msg)
