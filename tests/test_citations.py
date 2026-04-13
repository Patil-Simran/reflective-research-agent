import json
import re

from reflective_research.graph.citations import (
    audit_numbered_citations,
    build_numbered_evidence_for_prompt,
    cited_reference_numbers_from_report,
    sanitize_report_markdown,
)


def test_numbered_citations_valid() -> None:
    ok, issues = audit_numbered_citations("See [1] and [2] for details.", 3)
    assert ok is True
    assert issues == []


def test_numbered_citations_out_of_range() -> None:
    ok, issues = audit_numbered_citations("Bad ref [9].", 2)
    assert ok is False
    assert any("9" in i for i in issues)


def test_year_brackets_ignored() -> None:
    ok, issues = audit_numbered_citations("Published in [2023] and cited as [1].", 2)
    assert ok is True
    assert issues == []


def test_cited_reference_numbers_from_report() -> None:
    s = cited_reference_numbers_from_report("See [1] and [ 2 ] — also [2023] year.")
    assert s == {1, 2}


def test_sanitize_strips_legacy_e_citations_without_evidence() -> None:
    raw = "See [E:a8c23b5872ffeca4] and [1] for x."
    out = sanitize_report_markdown(raw)
    assert "[E:" not in out
    assert "[1]" in out


def test_sanitize_nfkc_fullwidth_and_spaced_e_tags() -> None:
    raw = "Bad ［E:52191611］ and [E : abcdef ] end."
    out = sanitize_report_markdown(raw)
    assert not re.search(r"\[\s*E\s*:", out, re.I)
    assert "end." in out


def test_sanitize_strips_chroma_websearch_stub_lines() -> None:
    raw = "## Refs\n[E:1] x.\nChroma.\nWeb search.\n\nOK."
    out = sanitize_report_markdown(raw)
    assert "Chroma" not in out
    assert "Web search" not in out
    assert "OK." in out


def test_sanitize_strips_rag_empty_and_bad_step_tags() -> None:
    raw = "Text [E:rag-empty-97490384] and [E:bad-step-abc] end."
    out = sanitize_report_markdown(raw)
    assert "[E:" not in out
    assert "rag-empty" not in out


def test_sanitize_maps_e_tags_to_numbered_cites() -> None:
    raw = "Claim [E:4585ff164b686d51] and later [E:4585ff164b686d51]."
    evidence = [
        {"id": "4585ff164b686d51deadbeef", "source_type": "search", "source_ref": "u", "content": "x" * 60},
        {"id": "99999999999999999999", "source_type": "search", "source_ref": "v", "content": "y" * 60},
    ]
    out = sanitize_report_markdown(raw, evidence)
    assert "[E:" not in out
    assert out.count("[1]") == 2
    assert "[2]" not in out


def test_evidence_json_includes_image_urls() -> None:
    items = [
        {
            "source_type": "search",
            "source_ref": "https://example.com/p",
            "content": "Long enough excerpt " * 5,
            "metadata": {"image_urls": ["https://upload.wikimedia.org/wiki/test.png"]},
        }
    ]
    _idx, blob = build_numbered_evidence_for_prompt(items)
    data = json.loads(blob)
    assert data[0].get("image_urls") == ["https://upload.wikimedia.org/wiki/test.png"]
