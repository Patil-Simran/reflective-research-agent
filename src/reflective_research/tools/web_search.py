"""Multi-provider retrieval: Brave (optional key), DDG instant, Wikipedia, Semantic Scholar, Crossref, arXiv, optional DDG package."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from reflective_research.config.settings import Settings
from reflective_research.domain.models import EvidenceItem
from reflective_research.tools.evidence_quality import (
    is_commerce_or_social_host,
    is_glossary_spam_host,
    is_non_english_qa_mirror_host,
)
from reflective_research.tools.url_fetch import _is_safe_public_http_url

log = logging.getLogger(__name__)

_USER_AGENT = (
    "ReflectiveResearchAgent/0.1 (+https://github.com; research bot; contact: local)"
)

# Strong English preference (otherwise many APIs follow client IP / locale → CN mirrors).
_HTTP_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9,en-GB;q=0.8,*;q=0.05",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _norm_result(
    title: str,
    body: str,
    href: str,
    provider: str,
    *,
    image: str | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "title": (title or "").strip(),
        "body": (body or "").strip()[:4000],
        "href": (href or "").strip(),
        "provider": provider,
    }
    if image and isinstance(image, str) and image.strip():
        d["image"] = image.strip()
    return d


def _safe_embed_image_url(url: str) -> bool:
    u = (url or "").strip()
    if not _is_safe_public_http_url(u):
        return False
    low = u.lower()
    if low.startswith("data:") or low.startswith("file:"):
        return False
    return True


def _brave_web_search(
    settings: Settings,
    query: str,
    max_results: int,
    timeout: float,
) -> list[dict[str, Any]]:
    key = (settings.brave_search_api_key or "").strip()
    if not key or max_results <= 0:
        return []
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={
                    "q": query,
                    "count": min(max_results, 20),
                    "search_lang": "en",
                    "country": "us",
                },
                headers={
                    "X-Subscription-Token": key,
                    "Accept": "application/json",
                    "User-Agent": _USER_AGENT,
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning("Brave Search failed (non-fatal): %s", e)
        return []
    out: list[dict[str, Any]] = []
    for row in (data.get("web") or {}).get("results") or []:
        title = str(row.get("title") or "")
        desc = str(row.get("description") or "")
        if not desc and isinstance(row.get("extra_snippets"), list) and row["extra_snippets"]:
            desc = str(row["extra_snippets"][0])[:2000]
        url = str(row.get("url") or "")
        if not (title.strip() or desc.strip()):
            continue
        body = f"{title}\n{desc}".strip()[:4000]
        out.append(_norm_result(title, body, url, "brave_web"))
    if out:
        log.info("Brave web: %s hit(s)", len(out))
    return out


def _semantic_scholar_search(query: str, max_results: int, timeout: float) -> list[dict[str, Any]]:
    if max_results <= 0:
        return []
    q = " ".join((query or "").split()[:28]).strip()
    if len(q) < 2:
        return []
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_HTTP_HEADERS) as client:
            r = client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": q,
                    "limit": max_results,
                    "fields": "title,abstract,year,venue,url,paperId",
                },
            )
            if r.status_code == 429:
                log.warning("Semantic Scholar rate limited (429)")
                return []
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.debug("Semantic Scholar failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for p in data.get("data") or []:
        title = str(p.get("title") or "")
        abst = str(p.get("abstract") or "")
        url = str(p.get("url") or "")
        pid = p.get("paperId")
        if not url and pid:
            url = f"https://www.semanticscholar.org/paper/{pid}"
        year = p.get("year")
        venue = str(p.get("venue") or "")
        body = (f"{abst}\n\nYear: {year}\nVenue: {venue}").strip()[:4000]
        blob = f"{title}\n\n{body}".strip()
        if len(blob) < 35:
            continue
        out.append(_norm_result(title, blob, url or "https://www.semanticscholar.org/", "semantic_scholar"))
    if out:
        log.info("Semantic Scholar: %s hit(s)", len(out))
    return out


def _crossref_search(
    query: str,
    max_results: int,
    timeout: float,
    mailto: str,
) -> list[dict[str, Any]]:
    if max_results <= 0:
        return []
    q = " ".join((query or "").split()[:40]).strip()[:400]
    if len(q) < 2:
        return []
    m = (mailto or "mailto:anonymous@example.org").strip()
    if not m.startswith("mailto:"):
        m = f"mailto:{m}"
    ua = f"ReflectiveResearchAgent/0.1 ({m})"
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": ua, "Accept": "application/json"},
        ) as client:
            r = client.get(
                "https://api.crossref.org/works",
                params={"query": q, "rows": max_results},
            )
            r.raise_for_status()
            payload = r.json()
    except Exception as e:
        log.debug("Crossref failed: %s", e)
        return []
    items = (payload.get("message") or {}).get("items") or []
    out: list[dict[str, Any]] = []
    for it in items:
        titles = it.get("title")
        title = titles[0] if isinstance(titles, list) and titles else ""
        title = str(title or "").strip()
        doi = str(it.get("DOI") or "").strip()
        href = f"https://doi.org/{doi}" if doi else ""
        container = it.get("container-title")
        cont = container[0] if isinstance(container, list) and container else ""
        cont = str(cont or "")
        raw_sub = it.get("subtitle")
        if isinstance(raw_sub, list) and raw_sub:
            subtitle = str(raw_sub[0] or "")
        else:
            subtitle = str(raw_sub or "").strip()
        pub = it.get("published-print") or it.get("published-online") or {}
        year = ""
        if isinstance(pub, dict) and pub.get("date-parts"):
            parts = pub["date-parts"][0]
            if isinstance(parts, list) and parts:
                year = str(parts[0])
        body = "\n".join(
            x for x in (subtitle and f"Subtitle: {subtitle}", f"Venue: {cont}", f"Year: {year}") if x
        )
        blob = f"{title}\n{body}".strip() if body else title
        if len(blob) < 15:
            continue
        out.append(_norm_result(title, blob[:4000], href or title[:48], "crossref"))
    if out:
        log.info("Crossref: %s hit(s)", len(out))
    return out


def _instant_answer_ddg(query: str, timeout: float) -> list[dict[str, Any]]:
    """DuckDuckGo Instant Answer API — stable JSON, works when HTML/Bing backends fail."""
    out: list[dict[str, Any]] = []
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=_HTTP_HEADERS,
        ) as client:
            r = client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.debug("DDG instant answer failed: %s", e)
        return out

    if data.get("AbstractText"):
        img = data.get("Image")
        img_s = str(img).strip() if img else None
        out.append(
            _norm_result(
                str(data.get("Heading") or query),
                str(data.get("AbstractText") or ""),
                str(data.get("AbstractURL") or ""),
                "duckduckgo_instant",
                image=img_s if img_s and _safe_embed_image_url(img_s) else None,
            )
        )
    for rt in data.get("RelatedTopics") or []:
        if len(out) >= 8:
            break
        if isinstance(rt, dict) and rt.get("Text"):
            out.append(
                _norm_result(
                    rt["Text"][:200],
                    str(rt.get("Text") or ""),
                    str(rt.get("FirstURL") or ""),
                    "duckduckgo_instant",
                )
            )
        elif isinstance(rt, dict) and isinstance(rt.get("Topics"), list):
            for t in rt["Topics"]:
                if len(out) >= 8:
                    break
                if isinstance(t, dict) and t.get("Text"):
                    out.append(
                        _norm_result(
                            str(t.get("Text", ""))[:200],
                            str(t.get("Text") or ""),
                            str(t.get("FirstURL") or ""),
                            "duckduckgo_instant",
                        )
                    )
    return out


def _wikipedia_opensearch(query: str, max_results: int, timeout: float) -> list[dict[str, Any]]:
    """MediaWiki opensearch + REST summaries — no key, high signal for definitions."""
    out: list[dict[str, Any]] = []
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=_HTTP_HEADERS,
        ) as client:
            r = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": max_results,
                    "namespace": 0,
                    "format": "json",
                },
            )
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 4:
                return out
            titles: list[str] = payload[1] if isinstance(payload[1], list) else []
            descs: list[str] = payload[2] if isinstance(payload[2], list) else []
            urls: list[str] = payload[3] if isinstance(payload[3], list) else []
            for i, title in enumerate(titles):
                if i >= max_results:
                    break
                snippet = descs[i] if i < len(descs) else ""
                page_url = urls[i] if i < len(urls) else ""
                body = snippet
                thumb_src: str | None = None
                if title:
                    path_seg = quote(title.replace(" ", "_"), safe="")
                    try:
                        sr = client.get(
                            f"https://en.wikipedia.org/api/rest_v1/page/summary/{path_seg}",
                        )
                        if sr.status_code == 200:
                            jd = sr.json()
                            extract = jd.get("extract")
                            if isinstance(extract, str) and extract.strip():
                                body = extract.strip()[:4000]
                            th = jd.get("thumbnail")
                            if isinstance(th, dict):
                                src = th.get("source")
                                if isinstance(src, str) and src.strip():
                                    thumb_src = src.strip()
                    except Exception:
                        pass
                if title or body:
                    safe_thumb = (
                        thumb_src
                        if (thumb_src and _safe_embed_image_url(thumb_src))
                        else None
                    )
                    out.append(
                        _norm_result(
                            title,
                            body,
                            page_url,
                            "wikipedia",
                            image=safe_thumb,
                        )
                    )
    except Exception as e:
        log.debug("Wikipedia search failed: %s", e)
    return out


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    reraise=True,
)
def _ddg_library_query(query: str, max_results: int, timeout: int) -> list[dict[str, Any]]:
    """Legacy duckduckgo_search package (currently Bing-only upstream — often empty)."""
    from duckduckgo_search import DDGS

    with DDGS(timeout=timeout) as ddgs:
        raw = list(
            ddgs.text(
                query,
                max_results=max_results,
                region="us-en",
            )
        )
    return [
        _norm_result(
            str(r.get("title") or ""),
            str(r.get("body") or ""),
            str(r.get("href") or ""),
            "duckduckgo_library",
        )
        for r in raw
    ]


_TRACKING_QUERY_KEYS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "mc_eid",
        "ref",
        "ref_src",
    }
)


def _canonical_href_for_dedupe(href: str) -> str:
    raw = (href or "").strip()
    if not raw:
        return ""
    try:
        p = urlparse(raw)
        if not p.netloc:
            return raw.lower()
        q = parse_qs(p.query, keep_blank_values=True)
        for k in list(q.keys()):
            lk = k.lower()
            if lk in _TRACKING_QUERY_KEYS or lk.startswith("utm_"):
                del q[k]
        new_query = urlencode(sorted(q.items()), doseq=True)
        return urlunparse(
            (
                (p.scheme or "https").lower(),
                p.netloc.lower(),
                p.path or "",
                p.params,
                new_query,
                "",
            )
        )
    except Exception:
        return raw.lower()


def _dedupe_key(r: dict[str, Any]) -> str:
    href = _canonical_href_for_dedupe(str(r.get("href") or ""))
    h = href + "|" + (r.get("body") or "")[:160]
    return hashlib.sha256(h.encode()).hexdigest()[:24]


def _unique_dict_results(merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in merged:
        if not (r.get("body") or r.get("title")):
            continue
        k = _dedupe_key(r)
        if k in seen:
            continue
        seen.add(k)
        unique.append(r)
    return unique


def _likely_research_query(q: str) -> bool:
    low = (q or "").lower()
    if len(low) > 52:
        return True
    needles = (
        "arxiv",
        "paper",
        "papers",
        "survey",
        "research",
        "llm",
        "gpt",
        "model",
        "infer",
        "quantiz",
        "transform",
        "neural",
        "learn",
        "rag",
        "retriev",
        "benchmark",
        "dataset",
        "train",
        "fine-tun",
        "nlp",
        "compare",
        "versus",
        "vs ",
    )
    return any(n in low for n in needles)


def _simplified_query(q: str) -> str:
    words = re.sub(r"[^\w\s]+", " ", q or "").split()
    if len(words) <= 3:
        return ""
    sq = " ".join(words[:8]).strip()
    return sq if sq.lower() != (q or "").strip().lower() else ""


def _clean_arxiv_xml_text(blob: str) -> str:
    t = re.sub(r"<[^>]+>", " ", blob)
    for a, b in (
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&amp;", "&"),
        ("&#x27;", "'"),
        ("&quot;", '"'),
    ):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def _arxiv_api_search(query: str, max_results: int, timeout: float) -> list[dict[str, Any]]:
    if max_results <= 0:
        return []
    q = " ".join((query or "").split()[:18]).strip()
    if len(q) < 2:
        return []
    url = (
        "https://export.arxiv.org/api/query?search_query=all:"
        f"{quote(q)}&max_results={max_results}&sortBy=relevance"
    )
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_HTTP_HEADERS) as client:
            r = client.get(url)
            r.raise_for_status()
            xml = r.text
    except Exception as e:
        log.debug("arXiv API failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"<entry>\s*([\s\S]*?)\s*</entry>", xml):
        block = m.group(1)
        id_m = re.search(r"<id>\s*([^<\s]+)\s*</id>", block)
        title_m = re.search(r"<title>\s*([\s\S]*?)\s*</title>", block)
        summary_m = re.search(r"<summary>\s*([\s\S]*?)\s*</summary>", block)
        if not id_m or not title_m:
            continue
        href = id_m.group(1).strip()
        title = _clean_arxiv_xml_text(title_m.group(1))[:500]
        summary = _clean_arxiv_xml_text(summary_m.group(1)) if summary_m else ""
        blob = f"{title}\n\n{summary}".strip()[:4000]
        if len(blob) < 40:
            continue
        out.append(_norm_result(title, blob, href, "arxiv"))
        if len(out) >= max_results:
            break
    if out:
        log.info("arXiv: %s hit(s) for query=%r", len(out), q[:80])
    return out


def _ddg_news_search(query: str, max_results: int, timeout: int) -> list[dict[str, Any]]:
    if max_results <= 0:
        return []
    try:
        from duckduckgo_search import DDGS

        with DDGS(timeout=timeout) as ddgs:
            raw = list(ddgs.news(query, region="us-en", max_results=max_results))
    except Exception as e:
        log.debug("DDG news failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for r in raw:
        title = str(r.get("title") or "")
        body = str(
            r.get("body") or r.get("description") or r.get("excerpt") or r.get("snippet") or ""
        )
        href = str(r.get("url") or r.get("link") or r.get("href") or "")
        if not (title.strip() or body.strip()):
            continue
        out.append(_norm_result(title, body, href, "duckduckgo_news"))
    return out


def _image_search_evidence(settings: Settings, query: str) -> list[EvidenceItem]:
    """DuckDuckGo image results as separate evidence rows (diagrams, figures, photos)."""
    if settings.image_search_max_per_query <= 0:
        return []
    ddg_timeout = max(10, min(int(min(float(settings.request_timeout_s), 30.0)), 60))
    out: list[EvidenceItem] = []
    try:
        from duckduckgo_search import DDGS

        with DDGS(timeout=ddg_timeout) as ddgs:
            raw = list(
                ddgs.images(
                    query,
                    region="us-en",
                    max_results=settings.image_search_max_per_query,
                )
            )
    except Exception as e:
        log.debug("DDG image search failed: %s", e)
        return out

    for j, r in enumerate(raw):
        img_url = str(r.get("image") or "").strip()
        if not _safe_embed_image_url(img_url):
            continue
        title = str(r.get("title") or "Image result")
        page_url = str(r.get("url") or r.get("source") or "").strip()
        if settings.search_block_commerce_hosts and page_url and is_commerce_or_social_host(
            page_url
        ):
            continue
        content = (
            f"Image search hit (figure, diagram, or photo): {title}\n"
            f"Direct image URL: {img_url}\n"
            f"Source / context page: {page_url or '(unknown)'}\n"
            "Embed this URL in the report as a Markdown image when it clarifies the answer; "
            "use the same citation index as this evidence row."
        )
        key = f"img|{img_url}|{title[:48]}"
        eid = hashlib.sha256(key.encode()).hexdigest()[:20]
        out.append(
            EvidenceItem(
                id=eid,
                content=content,
                source_type="search",
                source_ref=page_url or img_url,
                metadata={
                    "title": title,
                    "provider": "image_search",
                    "rank": j,
                    "image_urls": [img_url],
                },
            )
        )
    log.info("Image search: %s item(s) for query=%r", len(out), query[:80])
    return out


def _gather_search_dicts(
    settings: Settings,
    q: str,
    *,
    timeout: float,
    ddg_timeout: int,
) -> list[dict[str, Any]]:
    """Collect raw provider rows for one query string (used for primary + simplified retry)."""
    merged: list[dict[str, Any]] = []
    merged.extend(_brave_web_search(settings, q, settings.search_max_results, timeout))
    if settings.search_ddg_instant_enabled:
        merged.extend(_instant_answer_ddg(q, timeout))
    merged.extend(_wikipedia_opensearch(q, settings.search_max_results, timeout))
    if settings.search_semantic_scholar_enabled:
        merged.extend(
            _semantic_scholar_search(
                q, settings.search_semantic_scholar_max_results, timeout
            )
        )
    if settings.search_crossref_enabled:
        merged.extend(
            _crossref_search(
                q,
                settings.search_crossref_max_results,
                timeout,
                settings.crossref_mailto,
            )
        )
    if settings.search_duckduckgo_package_enabled:
        try:
            merged.extend(_ddg_library_query(q, settings.search_max_results, ddg_timeout))
        except Exception as e:
            log.warning("DDG library search failed (non-fatal): %s", e)
    return merged


def web_search_ddg(settings: Settings, query: str) -> list[EvidenceItem]:
    """Merge Brave (optional), DDG instant, Wikipedia, Semantic Scholar, Crossref, optional DDG package, arXiv."""
    timeout = min(float(settings.request_timeout_s), 30.0)
    ddg_timeout = max(10, min(int(timeout), 60))

    merged = _gather_search_dicts(settings, query, timeout=timeout, ddg_timeout=ddg_timeout)
    unique = _unique_dict_results(merged)

    if settings.search_arxiv_enabled and (
        _likely_research_query(query) or len(unique) < 2
    ):
        merged.extend(
            _arxiv_api_search(query, settings.search_arxiv_max_results, timeout)
        )
        unique = _unique_dict_results(merged)

    if (
        settings.search_duckduckgo_package_enabled
        and settings.search_news_fallback_enabled
        and len(unique) < 2
    ):
        merged.extend(
            _ddg_news_search(query, settings.search_news_max_results, ddg_timeout)
        )
        unique = _unique_dict_results(merged)

    sq = _simplified_query(query)
    if sq and len(unique) < 2:
        merged.extend(_gather_search_dicts(settings, sq, timeout=timeout, ddg_timeout=ddg_timeout))
        if settings.search_arxiv_enabled:
            merged.extend(
                _arxiv_api_search(sq, settings.search_arxiv_max_results, timeout)
            )
        if (
            settings.search_duckduckgo_package_enabled
            and settings.search_news_fallback_enabled
        ):
            merged.extend(
                _ddg_news_search(sq, settings.search_news_max_results, ddg_timeout)
            )
        unique = _unique_dict_results(merged)

    items: list[EvidenceItem] = []
    for i, r in enumerate(unique[: max(settings.search_max_results * 3, 8)]):
        title = r.get("title") or ""
        body = r.get("body") or ""
        href = (r.get("href") or "").strip()
        if settings.search_block_commerce_hosts and href and is_commerce_or_social_host(href):
            continue
        if (
            not settings.search_allow_chinese_qa_mirrors
            and href
            and is_non_english_qa_mirror_host(href)
        ):
            continue
        if (
            settings.search_block_glossary_spam_hosts
            and href
            and is_glossary_spam_host(href)
        ):
            continue
        prov = r.get("provider") or "web"
        content = f"{title}\n{body}".strip()
        key = f"{href}|{content[:120]}"
        eid = hashlib.sha256(key.encode()).hexdigest()[:20]
        meta: dict[str, Any] = {"title": title, "rank": i, "provider": prov}
        img = r.get("image")
        if isinstance(img, str) and _safe_embed_image_url(img):
            meta["image_urls"] = [img.strip()]
        items.append(
            EvidenceItem(
                id=eid,
                content=content,
                source_type="search",
                source_ref=href or prov,
                metadata=meta,
            )
        )

    image_items: list[EvidenceItem] = []
    if (
        settings.search_duckduckgo_package_enabled
        and settings.image_search_enabled
        and settings.image_search_max_per_query > 0
    ):
        try:
            image_items = _image_search_evidence(settings, query)
        except Exception as e:
            log.warning("Image search failed (non-fatal): %s", e)

    if not items:
        if image_items:
            items = image_items
        else:
            nid = hashlib.sha256(f"search-empty|{query}".encode()).hexdigest()[:16]
            items.append(
                EvidenceItem(
                    id=nid,
                    content="No web results returned for this query (tried Brave when "
                    "BRAVE_SEARCH_API_KEY is set, Wikipedia, Semantic Scholar, Crossref, "
                    "optional DuckDuckGo instant and library, and arXiv when applicable).",
                    source_type="system",
                    source_ref="web_search",
                    metadata={"query": query},
                )
            )
    else:
        items.extend(image_items)
    return items
