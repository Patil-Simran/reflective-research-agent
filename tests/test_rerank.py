from reflective_research.config.settings import Settings
from reflective_research.tools.evidence_quality import rerank_by_query_overlap


def test_rerank_orders_by_overlap() -> None:
    q = "vector clock distributed"
    items = [
        {"id": "a", "content": "unrelated cooking recipe pasta sauce"},
        {"id": "b", "content": "vector clocks track causal order in distributed systems"},
    ]
    out = rerank_by_query_overlap(q, items, Settings(evidence_authority_bonus_enabled=False))
    assert out[0]["id"] == "b"
    assert out[1]["id"] == "a"


def test_rerank_empty_passthrough() -> None:
    assert rerank_by_query_overlap("x", []) == []


def test_rerank_trusted_domain_tiebreak() -> None:
    """Same lexical overlap: prefer arXiv / Wikipedia-style URLs (OSS deep-research pattern)."""
    q = "distributed consensus raft"
    body = "distributed consensus raft algorithm leader election " * 2
    items = [
        {
            "id": "blog",
            "source_type": "search",
            "source_ref": "https://random.example.net/raft-notes",
            "content": body,
        },
        {
            "id": "wiki",
            "source_type": "search",
            "source_ref": "https://en.wikipedia.org/wiki/Raft_(algorithm)",
            "content": body,
        },
    ]
    out = rerank_by_query_overlap(q, items, Settings())
    assert out[0]["id"] == "wiki"
    assert out[1]["id"] == "blog"
