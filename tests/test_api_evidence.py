"""API evidence shaping: writer-aligned order and cited-only list."""

from __future__ import annotations

from reflective_research.config.settings import Settings
from reflective_research.service import build_api_evidence_list


def _row(url: str, body: str, n: int = 5) -> dict:
    text = (body + " ") * n
    return {
        "id": f"id-{hash(url) % 10000}",
        "source_type": "search",
        "source_ref": url,
        "content": text.strip(),
        "metadata": {"title": url},
    }


def test_build_api_evidence_cited_only_filters_to_brackets() -> None:
    ev = [
        _row("https://a.example/a", "alpha topic"),
        _row("https://b.example/b", "beta topic"),
        _row("https://c.example/c", "gamma topic"),
    ]
    st = {
        "user_query": "alpha beta gamma",
        "evidence": ev,
        "report": "Discussion see [3] and [1].",
    }
    s = Settings(
        api_evidence_cited_only=True,
        evidence_rerank_enabled=False,
        search_allow_chinese_qa_mirrors=False,
    )
    out = build_api_evidence_list(st, s)
    cites = [int(x["writer_cite"]) for x in out]
    assert cites == [1, 3]
    refs = [x["source_ref"] for x in out]
    assert "a.example" in refs[0]
    assert "c.example" in refs[1]


def test_build_api_evidence_falls_back_when_no_brackets_in_report() -> None:
    # Short tokens need more repeats: _row_usable_for_client requires len(content) >= 50.
    ev = [_row("https://only.example/x", "solo", n=15)]
    st = {"user_query": "solo", "evidence": ev, "report": "No citations here."}
    s = Settings(api_evidence_cited_only=True, evidence_rerank_enabled=False)
    out = build_api_evidence_list(st, s)
    assert len(out) == 1
    assert out[0]["writer_cite"] == 1
