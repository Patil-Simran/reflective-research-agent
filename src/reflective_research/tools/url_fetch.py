"""Fetch and extract main text from search result URLs (no API keys)."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import httpx

try:
    import trafilatura  # type: ignore[import-untyped]
except ImportError:
    trafilatura = None

from reflective_research.config.settings import Settings
from reflective_research.domain.models import EvidenceItem

log = logging.getLogger(__name__)

_USER_AGENT = (
    "ReflectiveResearchAgent/0.1 (+https://github.com; research fetch; contact: local)"
)

_FETCH_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_SKIP_HOST_SUFFIXES = (
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
)


def _is_safe_public_http_url(url: str) -> bool:
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "::1"):
        return False
    for suf in _SKIP_HOST_SUFFIXES:
        if host == suf or host.endswith("." + suf):
            return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
            return False
    except ValueError:
        pass
    return True


def _prefer_english_content_url(url: str) -> str:
    """Swap Chinese Wikipedia for English when possible (same article title path)."""
    u = url.strip()
    if "//zh.wikipedia.org/" in u:
        return u.replace("//zh.wikipedia.org/", "//en.wikipedia.org/", 1)
    return u


def _strip_html_fallback(html: str) -> str:
    t = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    t = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def fetch_url_main_text(url: str, settings: Settings) -> tuple[str | None, str | None]:
    """
    Download URL (capped) and return (extracted_text, error_message).
    error_message is None on success.
    """
    if not _is_safe_public_http_url(url):
        return None, "blocked or unsupported URL"
    cap = int(settings.url_fetch_max_bytes)
    timeout = float(settings.url_fetch_timeout_s)
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers=_FETCH_HEADERS,
        ) as client:
            r = client.get(url)
            r.raise_for_status()
            raw = r.content[:cap]
            charset = r.encoding or "utf-8"
    except Exception as e:
        log.debug("fetch failed %s: %s", url, e)
        return None, str(e)
    try:
        html = raw.decode(charset, errors="replace")
    except Exception:
        html = raw.decode("utf-8", errors="replace")

    text: str | None = None
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if isinstance(extracted, str) and extracted.strip():
                text = extracted.strip()
        except Exception as e:
            log.debug("trafilatura failed %s: %s", url, e)
    if not text:
        text = _strip_html_fallback(html)
    if not text or len(text) < 80:
        return None, "extracted text too short"
    return text[:12000], None


def enrich_search_evidence(
    settings: Settings,
    search_items: list[EvidenceItem],
) -> list[EvidenceItem]:
    """
    For unique http(s) URLs from search hits, fetch page text and append new evidence rows.
    """
    if not settings.url_fetch_enabled or settings.url_fetch_per_search_step <= 0:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for it in search_items:
        if it.source_type != "search":
            continue
        ref = _prefer_english_content_url((it.source_ref or "").strip())
        if not ref.startswith("http") or ref in seen:
            continue
        if not _is_safe_public_http_url(ref):
            continue
        seen.add(ref)
        urls.append(ref)
        if len(urls) >= settings.url_fetch_per_search_step:
            break

    if not urls:
        return []

    extra: list[EvidenceItem] = []
    max_workers = min(4, len(urls))

    def one(u: str) -> EvidenceItem | None:
        body, err = fetch_url_main_text(u, settings)
        if err or not body:
            return None
        key = f"fetch|{u}|{body[:80]}"
        eid = hashlib.sha256(key.encode()).hexdigest()[:20]
        title = u
        return EvidenceItem(
            id=eid,
            content=f"{title}\n\n{body}",
            source_type="search",
            source_ref=u,
            metadata={"provider": "url_fetch", "parent": "web_search"},
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(one, u): u for u in urls}
        for fut in as_completed(futs):
            try:
                row = fut.result()
                if row is not None:
                    extra.append(row)
            except Exception as e:
                log.debug("fetch task error: %s", e)

    log.info("URL enrich: added %s fetched page(s)", len(extra))
    return extra
