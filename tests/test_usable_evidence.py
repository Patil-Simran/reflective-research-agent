from reflective_research.tools.evidence_quality import _row_usable_for_client


def test_usable_evidence_keeps_substantive_search() -> None:
    ev = [
        {
            "id": "a",
            "source_type": "system",
            "content": "RAG returned no chunks.",
            "source_ref": "chroma",
        },
        {
            "id": "b",
            "source_type": "search",
            "content": "RAG returned no chunks (empty corpus or no match).",
            "source_ref": "chroma",
        },
        {
            "id": "c",
            "source_type": "search",
            "content": "The CAP theorem states trade-offs among consistency, availability, "
            "and partition tolerance in distributed data stores. Designers must choose "
            "which guarantees to prioritize when networks fail.",
            "source_ref": "https://example.com/cap",
        },
    ]
    u = [e for e in ev if _row_usable_for_client(e)]
    assert len(u) == 1
    assert u[0]["id"] == "c"
