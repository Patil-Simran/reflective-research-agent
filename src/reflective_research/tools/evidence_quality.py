"""Single place: commerce URL filter, query–text relevance gate, and overlap rerank."""

from __future__ import annotations

import math
import re
from typing import Any
from urllib.parse import urlparse

from reflective_research.config.settings import Settings

# Same semantics as graph.nodes._usable_evidence (avoid importing nodes).
_DISPLAY_EMPTY_MARKERS: tuple[str, ...] = (
    "no web results returned",
    "rag returned no chunks",
    "search failed:",
    "no documents loaded",
)


def _row_usable_for_client(e: dict[str, Any]) -> bool:
    st = e.get("source_type")
    if st not in ("search", "rag"):
        return False
    content = (e.get("content") or "").strip()
    if len(content) < 50:
        return False
    low = content.lower()
    return not any(m in low for m in _DISPLAY_EMPTY_MARKERS)


def prepare_writer_evidence_pipeline(
    raw_items: list[dict[str, Any]],
    user_query: str,
    settings: Settings,
) -> tuple[list[dict[str, Any]], int]:
    """
    Same pipeline as the graph writer: usable rows → relevance gate → optional rerank.
    Keeps API citation indices aligned with REFERENCE LIST / EVIDENCE JSON.
    """
    usable = [e for e in raw_items if isinstance(e, dict) and _row_usable_for_client(e)]
    discarded = len(raw_items) - len(usable)
    usable, drop_rel = filter_evidence_relevance(usable, user_query, settings)
    discarded += drop_rel
    if settings.evidence_rerank_enabled and usable:
        usable = rerank_by_query_overlap(user_query, usable, settings)
    return usable, discarded


def filter_evidence_for_client(
    items: list[dict[str, Any]],
    user_query: str,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Backward-compatible alias: full writer pipeline (includes rerank when enabled)."""
    out, _ = prepare_writer_evidence_pipeline(
        [e for e in items if isinstance(e, dict)],
        user_query,
        settings,
    )
    return out


_BLOCKED_RETAIL_SUFFIXES: tuple[str, ...] = (
    "amazon.com",
    "amazon.in",
    "amazon.co.uk",
    "amazon.de",
    "amazon.fr",
    "amazon.es",
    "amazon.ca",
    "amazon.com.au",
    "amazon.co.jp",
    "flipkart.com",
    "ebay.com",
    "ebay.in",
    "walmart.com",
    "aliexpress.com",
    "etsy.com",
    "myntra.com",
    "snapdeal.com",
    "bestbuy.com",
    "shopify.com",
    "pinterest.com",
    "instagram.com",
    "tiktok.com",
)

_STOP = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had has have
    he her him his how if in into is it its may me might more most must my no nor not of
    on or our out s shall she should some such than that the their them then there these
    they this those through to too under until up us very was we were what when where
    which who why will with would you your
    """.split()
)


def is_commerce_or_social_host(url: str) -> bool:
    host = (urlparse((url or "").strip()).hostname or "").lower()
    if not host:
        return False
    if "aws.amazon.com" in host:
        return False
    return any(host == suf or host.endswith("." + suf) for suf in _BLOCKED_RETAIL_SUFFIXES)


# Geo-locked / Chinese UGC Q&A that often ranks for English tech terms but yields unusable links.
_BLOCKED_NON_ENGLISH_QA_SUFFIXES: tuple[str, ...] = (
    "zhidao.baidu.com",
    "tieba.baidu.com",
    "wen.baidu.com",
    "zhihu.com",
    "zhihu.cn",
)


def is_non_english_qa_mirror_host(url: str) -> bool:
    host = (urlparse((url or "").strip()).hostname or "").lower()
    if not host:
        return False
    return any(host == suf or host.endswith("." + suf) for suf in _BLOCKED_NON_ENGLISH_QA_SUFFIXES)


# Generic dictionary / thesaurus pages that rank for adjectives in tech prose (“operational”, …).
_GLOSSARY_SPAM_SUFFIXES: tuple[str, ...] = (
    "dictionary.cambridge.org",
    "merriam-webster.com",
    "dictionary.com",
    "oxfordlearnersdictionaries.com",
    "thefreedictionary.com",
    "collinsdictionary.com",
    "yourdictionary.com",
    "vocabulary.com",
)

# Brand / vertical landers that match short tokens from unrelated searches (e.g. “CASE” → tractors).
_BRAND_SPAM_SUFFIXES: tuple[str, ...] = (
    "caseih.com",
    "caseih.eu",
    "newholland.com",
)

# ESL / “what does X mean” UGC — ranks for generic English words in tech prompts (failure, cost, …).
_LANGUAGE_EXCHANGE_SUFFIXES: tuple[str, ...] = (
    "hinative.com",
    "jingyan.baidu.com",
)


def is_glossary_spam_host(url: str) -> bool:
    host = (urlparse((url or "").strip()).hostname or "").lower()
    if not host:
        return False
    if any(host == suf or host.endswith("." + suf) for suf in _GLOSSARY_SPAM_SUFFIXES):
        return True
    if any(host == suf or host.endswith("." + suf) for suf in _BRAND_SPAM_SUFFIXES):
        return True
    return any(host == suf or host.endswith("." + suf) for suf in _LANGUAGE_EXCHANGE_SUFFIXES)


def _substantive_terms(query: str) -> set[str]:
    words = re.findall(r"\w+", (query or "").lower())
    return {w for w in words if len(w) >= 3 and w not in _STOP}


# Strip generic comparison vocabulary so “failure modes / cost / latency” alone cannot match ESL spam.
_OVERLAP_NOISE_TERMS: frozenset[str] = frozenset(
    {
        "failure",
        "fail",
        "error",
        "cost",
        "latency",
        "mode",
        "modes",
        "trade",
        "off",
        "offs",
        "versus",
        "comparison",
        "compare",
        "when",
        "use",
        "each",
        "application",
        "applications",
        "issue",
        "issues",
        "risk",
        "risks",
        "problem",
        "problems",
        "benefit",
        "benefits",
        "drawback",
        "drawbacks",
        "strength",
        "weakness",
        "weaknesses",
        "advantage",
        "advantages",
        "disadvantage",
        "disadvantages",
        "option",
        "options",
        "implication",
        "implications",
    }
)


def _substantive_terms_for_search_overlap(query: str) -> set[str]:
    base = _substantive_terms(query)
    strict = base - _OVERLAP_NOISE_TERMS
    return strict if len(strict) >= 3 else base


def _required_term_hits(qset: set[str], settings: Settings) -> int:
    if not qset:
        return 0
    raw = math.ceil(len(qset) * float(settings.evidence_substantive_overlap_fraction))
    return max(1, min(settings.evidence_substantive_max_required_hits, raw))


def filter_evidence_relevance(
    items: list[dict[str, Any]],
    user_query: str,
    settings: Settings,
) -> tuple[list[dict[str, Any]], int]:
    if not items:
        return items, 0
    qset = _substantive_terms_for_search_overlap(user_query)
    need = _required_term_hits(qset, settings)
    out: list[dict[str, Any]] = []
    discarded = 0
    for e in items:
        st = e.get("source_type")
        ref = str(e.get("source_ref") or "")
        content = str(e.get("content") or "")

        if st == "search" and settings.search_block_commerce_hosts and is_commerce_or_social_host(
            ref
        ):
            discarded += 1
            continue

        if (
            st == "search"
            and not settings.search_allow_chinese_qa_mirrors
            and is_non_english_qa_mirror_host(ref)
        ):
            discarded += 1
            continue

        if (
            st == "search"
            and settings.search_block_glossary_spam_hosts
            and is_glossary_spam_host(ref)
        ):
            discarded += 1
            continue

        if st != "search":
            out.append(e)
            continue

        if not qset:
            out.append(e)
            continue

        text_words = set(re.findall(r"\w+", content.lower()))
        hits = len(qset & text_words)
        min_hits = need
        if len(qset) >= 4:
            min_hits = max(min_hits, 2)
        if hits < min_hits:
            discarded += 1
            continue
        out.append(e)

    return out, discarded


# Tie-break after query overlap — pattern used in many open “deep research” / agent templates.
_AUTHORITY_HOST_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("arxiv.org", 0.42),
    ("semanticscholar.org", 0.28),
    ("wikipedia.org", 0.34),
    ("wikimedia.org", 0.2),
    ("nature.com", 0.14),
    ("ieee.org", 0.12),
    ("acm.org", 0.12),
    ("usenix.org", 0.12),
    ("readthedocs.io", 0.1),
    ("developer.mozilla.org", 0.1),
    ("docs.python.org", 0.1),
    ("kernel.org", 0.1),
    ("ietf.org", 0.14),
    ("rfc-editor.org", 0.12),
    ("stackoverflow.com", 0.06),
    ("stackexchange.com", 0.04),
)


def _edu_or_gov_bonus(host: str) -> float:
    b = 0.0
    if re.search(r"(\.edu$|\.edu\.|\.ac\.uk$)", host):
        b = max(b, 0.16)
    if host.endswith(".gov") or host.endswith(".gov.uk"):
        b = max(b, 0.12)
    return b


def source_authority_bonus(url: str) -> float:
    """Small non-negative bonus for well-known technical / scholarly hosts."""
    try:
        host = (urlparse((url or "").strip()).hostname or "").lower()
    except Exception:
        return 0.0
    if not host:
        return 0.0
    best = 0.0
    for suf, w in _AUTHORITY_HOST_WEIGHTS:
        if host == suf or host.endswith("." + suf):
            best = max(best, w)
    best = max(best, _edu_or_gov_bonus(host))
    return min(best, 0.5)


def rerank_by_query_overlap(
    user_query: str,
    items: list[dict[str, Any]],
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return items
    q_words = set(re.findall(r"\w+", (user_query or "").lower()))
    if not q_words:
        return items

    use_bonus = settings is None or bool(
        getattr(settings, "evidence_authority_bonus_enabled", True)
    )

    def score(e: dict[str, Any]) -> float:
        tw = set(re.findall(r"\w+", (e.get("content") or "").lower()))
        overlap = len(q_words & tw) / max(len(q_words), 1)
        if not use_bonus:
            return overlap
        st = e.get("source_type")
        if st == "rag":
            return overlap + 0.1
        if st == "search":
            return overlap + source_authority_bonus(str(e.get("source_ref") or ""))
        return overlap

    return sorted(items, key=score, reverse=True)
