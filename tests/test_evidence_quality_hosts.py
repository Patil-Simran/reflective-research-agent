"""Host-based evidence filters (commerce, non-English Q&A mirrors)."""

from __future__ import annotations

from reflective_research.config.settings import Settings
from reflective_research.tools.evidence_quality import (
    filter_evidence_for_client,
    filter_evidence_relevance,
    is_glossary_spam_host,
    is_non_english_qa_mirror_host,
)


def test_non_english_qa_host_detection() -> None:
    assert is_non_english_qa_mirror_host("https://zhidao.baidu.com/question/1")
    assert is_non_english_qa_mirror_host("https://www.zhihu.com/question/123")
    assert not is_non_english_qa_mirror_host("https://en.wikipedia.org/wiki/RAG")


def test_filter_drops_baidu_zhihu_by_default() -> None:
    s = Settings()
    items = [
        {
            "source_type": "search",
            "source_ref": "https://zhidao.baidu.com/question/2018946678738792308.html",
            "content": "retrieval augmented generation domain adaptation " * 4,
        }
    ]
    out, disc = filter_evidence_relevance(items, "retrieval domain augmented", s)
    assert out == []
    assert disc == 1


def test_zhihu_still_dropped_when_legacy_prefer_english_false() -> None:
    """Old SEARCH_PREFER_ENGLISH_SOURCES=false must not re-enable CN mirrors."""
    s = Settings(search_prefer_english_sources=False, search_allow_chinese_qa_mirrors=False)
    items = [
        {
            "source_type": "search",
            "source_ref": "https://www.zhihu.com/question/1",
            "content": "retrieval augmented generation domain " * 4,
        }
    ]
    out, disc = filter_evidence_relevance(items, "retrieval domain augmented", s)
    assert out == []
    assert disc == 1


def test_filter_evidence_for_client_matches_relevance_gate() -> None:
    s = Settings()
    raw = [
        {
            "source_type": "search",
            "source_ref": "https://www.zhihu.com/question/1",
            "content": "latency networking delay meaning " * 5,
        },
        {
            "source_type": "search",
            "source_ref": "https://example.com/paper",
            "content": "network latency delay TCP performance " * 5,
        },
    ]
    out = filter_evidence_for_client(raw, "network latency versus delay", s)
    assert len(out) == 1
    assert "example.com" in (out[0].get("source_ref") or "")


def test_glossary_and_brand_spam_hosts() -> None:
    assert is_glossary_spam_host("https://www.merriam-webster.com/dictionary/foo")
    assert is_glossary_spam_host("https://www.caseih.com/en-gb/")
    assert is_glossary_spam_host("https://zh.hinative.com/questions/1")
    assert is_glossary_spam_host("https://jingyan.baidu.com/article/455a99500a264ea166277826.html")
    assert not is_glossary_spam_host("https://en.wikipedia.org/wiki/CAP_theorem")


def test_filter_drops_hinative_when_glossary_block_on() -> None:
    s = Settings()
    items = [
        {
            "source_type": "search",
            "source_ref": "https://hinative.com/questions/5139453",
            "content": "retrieval augmented generation supervised fine tuning " * 4,
        }
    ]
    out, disc = filter_evidence_relevance(items, "retrieval domain augmented", s)
    assert out == []
    assert disc == 1


def test_filter_drops_search_hit_when_overlap_only_generic_words() -> None:
    """Generic prompt words like 'failure' must not admit ESL pages about English idioms."""
    s = Settings()
    items = [
        {
            "source_type": "search",
            "source_ref": "https://example.com/esl",
            "content": "Failure is not an option idiom English learning phrase meaning " * 3,
        }
    ]
    q = (
        "Compare retrieval-augmented generation versus supervised fine-tuning for "
        "domain-specific question answering: failure modes and cost"
    )
    out, disc = filter_evidence_relevance(items, q, s)
    assert out == []
    assert disc == 1


def test_filter_drops_dictionary_when_glossary_block_on() -> None:
    s = Settings()
    items = [
        {
            "source_type": "search",
            "source_ref": "https://dictionary.cambridge.org/dictionary/english/operational",
            "content": "operational definition distributed systems messaging " * 4,
        }
    ]
    out, disc = filter_evidence_relevance(items, "synchronous asynchronous messaging distributed", s)
    assert out == []
    assert disc == 1


def test_filter_keeps_zhihu_when_allow_chinese_mirrors_on() -> None:
    s = Settings(search_allow_chinese_qa_mirrors=True)
    items = [
        {
            "source_type": "search",
            "source_ref": "https://www.zhihu.com/question/1",
            "content": "retrieval augmented generation domain " * 4,
        }
    ]
    out, disc = filter_evidence_relevance(items, "retrieval domain augmented", s)
    assert len(out) == 1
    assert disc == 0
