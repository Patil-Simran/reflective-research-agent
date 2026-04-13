"""Unit tests for multi-provider web search (no live network when mocked)."""

from __future__ import annotations

from reflective_research.config.settings import Settings
from reflective_research.tools import web_search as ws


def test_web_search_merges_instant_answer(monkeypatch) -> None:
    def fake_instant(_q: str, _t: float) -> list[dict]:
        return [
            {
                "title": "Vector clock",
                "body": "A vector clock is a data structure.",
                "href": "https://example.com/vc",
                "provider": "duckduckgo_instant",
            }
        ]

    monkeypatch.setattr(ws, "_brave_web_search", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_instant_answer_ddg", fake_instant)
    monkeypatch.setattr(ws, "_wikipedia_opensearch", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_semantic_scholar_search", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_crossref_search", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_ddg_library_query", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_image_search_evidence", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_arxiv_api_search", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_ddg_news_search", lambda *a, **k: [])

    s = Settings()
    items = ws.web_search_ddg(s, "vector clock")
    assert len(items) == 1
    assert items[0].source_type == "search"
    assert "vector clock" in items[0].content.lower()
    assert items[0].metadata.get("provider") == "duckduckgo_instant"


def test_dedupe_key_normalizes_tracking_params() -> None:
    a = {
        "href": "https://example.com/article?utm_source=twitter&id=1",
        "body": "Same body text here.",
        "title": "t",
    }
    b = {
        "href": "https://example.com/article?utm_campaign=spring&id=1",
        "body": "Same body text here.",
        "title": "t",
    }
    assert ws._dedupe_key(a) == ws._dedupe_key(b)


def test_web_search_empty_when_all_providers_fail(monkeypatch) -> None:
    monkeypatch.setattr(ws, "_brave_web_search", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_instant_answer_ddg", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_wikipedia_opensearch", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_semantic_scholar_search", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_crossref_search", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_image_search_evidence", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_arxiv_api_search", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_ddg_news_search", lambda *a, **k: [])

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(ws, "_ddg_library_query", boom)

    s = Settings()
    items = ws.web_search_ddg(s, "xyzabc123unlikely")
    assert len(items) == 1
    assert items[0].source_type == "system"
    assert "No web results" in items[0].content
