from reflective_research.config.settings import Settings
from reflective_research.tools.evidence_quality import filter_evidence_relevance


def test_drops_amazon_even_if_long_snippet() -> None:
    s = Settings()
    ev = [
        {
            "id": "1",
            "source_type": "search",
            "source_ref": "https://www.amazon.in/women",
            "content": "Amazon.in: women Check each product page for other buying options. "
            "Price and other details may vary based on product size and colour. " * 2,
        }
    ]
    out, dropped = filter_evidence_relevance(ev, "quantized LLM inference papers", s)
    assert out == []
    assert dropped == 1


def test_drops_unrelated_search_text() -> None:
    s = Settings(evidence_substantive_overlap_fraction=0.12)
    ev = [
        {
            "id": "1",
            "source_type": "search",
            "source_ref": "https://example.com/blog",
            "content": "Best pasta recipes and sauce ideas for home cooks. " * 5,
        }
    ]
    out, dropped = filter_evidence_relevance(
        ev, "quantized large language model inference optimization", s
    )
    assert out == []
    assert dropped == 1


def test_keeps_on_topic_search() -> None:
    s = Settings()
    ev = [
        {
            "id": "1",
            "source_type": "search",
            "source_ref": "https://arxiv.org/abs/1234",
            "content": "We study post-training quantization for large language models and "
            "compare inference latency with FP16 baselines on standard benchmarks.",
        }
    ]
    out, dropped = filter_evidence_relevance(ev, "quantized LLM inference", s)
    assert len(out) == 1
    assert dropped == 0


def test_keeps_rag_without_term_overlap() -> None:
    s = Settings()
    ev = [
        {
            "id": "1",
            "source_type": "rag",
            "source_ref": "internal.pdf",
            "content": "Proprietary internal notes that use different wording entirely.",
        }
    ]
    out, dropped = filter_evidence_relevance(ev, "quantized LLM inference", s)
    assert len(out) == 1
    assert dropped == 0
