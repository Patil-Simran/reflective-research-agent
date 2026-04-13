"""Numbered in-text citations [1], [2] for prompts and verification."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

# Optional spaces inside brackets: [1] or [ 2 ]
_NUM_CIT_RE = re.compile(r"\[\s*(\d+)\s*\]")
# Any legacy [E:…] the model invents (hex ids, rag-empty-*, bad-step-*, etc.).
_E_TAG_ANY_RE = re.compile(r"\[E:([^\]]+)\]")
# Catch spacing / variant colons after first pass (model or unicode oddities).
_E_TAG_LOOSE_RE = re.compile(r"\[\s*E\s*:\s*[^\]]+\]", re.IGNORECASE)


def _is_likely_year_bracket(token: str) -> bool:
    """Treat [1999]–[2035] as years, not reference tags."""
    if len(token) == 4 and token.isdigit():
        y = int(token)
        return 1900 <= y <= 2035
    return False


def cited_reference_numbers_from_report(report: str) -> set[int]:
    """
    In-text citation markers [1], [2], … used in the final report (years like [2023] excluded).
    Used to show only sources the model actually cited — common in answer-style research UIs.
    """
    found: set[int] = set()
    for m in _NUM_CIT_RE.finditer(report or ""):
        tok = m.group(1)
        if _is_likely_year_bracket(tok):
            continue
        n = int(tok)
        if n > 0:
            found.add(n)
    return found


def _image_urls_from_metadata(meta: Any) -> list[str]:
    if not isinstance(meta, dict):
        return []
    out: list[str] = []
    raw = meta.get("image_urls")
    if isinstance(raw, list):
        for u in raw:
            if isinstance(u, str) and u.strip():
                out.append(u.strip())
    one = meta.get("image_url")
    if isinstance(one, str) and one.strip():
        out.append(one.strip())
    # de-dupe, keep order
    seen: set[str] = set()
    uniq: list[str] = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq[:8]


def build_numbered_evidence_for_prompt(items: list[dict[str, Any]]) -> tuple[str, str]:
    """
    Human-readable index + JSON for the LLM. Uses cite numbers 1..n only (no raw hashes in prose).
    """
    lines: list[str] = []
    payload: list[dict[str, Any]] = []
    for i, e in enumerate(items, start=1):
        ref = str(e.get("source_ref", ""))[:240]
        st = e.get("source_type", "")
        lines.append(f"{i}. [{st}] {ref}")
        entry: dict[str, Any] = {
            "cite": i,
            "source_type": st,
            "source_ref": str(e.get("source_ref", ""))[:900],
            "excerpt": (e.get("content") or "")[:12000],
        }
        imgs = _image_urls_from_metadata(e.get("metadata"))
        if imgs:
            entry["image_urls"] = [u[:900] for u in imgs]
        payload.append(entry)
    idx = "\n".join(lines) if lines else "(none)"
    blob = json.dumps(payload, indent=2) if payload else "[]"
    return idx, blob


def _e_tag_is_internal_noise(tag: str) -> bool:
    t = (tag or "").strip().lower()
    return bool(
        t.startswith("rag-empty")
        or t.startswith("bad-step")
        or "empty-corpus" in t
        or t.startswith("search-empty")
    )


def _cite_index_for_e_tag(h: str, evidence: list[dict[str, Any]]) -> int | None:
    """Map [E:hash] body to 1-based cite index using evidence order (prefix / exact on id)."""
    h = (h or "").strip().lower()
    if len(h) < 4 or _e_tag_is_internal_noise(h):
        return None
    for i, e in enumerate(evidence, start=1):
        eid = str(e.get("id") or "").strip().lower()
        if not eid:
            continue
        if eid == h or eid.startswith(h) or h.startswith(eid):
            return i
    return None


def sanitize_report_markdown(
    report: str,
    evidence: list[dict[str, Any]] | None = None,
) -> str:
    """
    Turn legacy [E:hash] tags into [1], [2], … when the hash matches an evidence id
    (exact or prefix). Unmatched tags are removed. Then tidy spacing.
    """
    raw = unicodedata.normalize("NFKC", report or "")

    def repl(m: re.Match[str]) -> str:
        tag = (m.group(1) or "").strip()
        if _e_tag_is_internal_noise(tag):
            return ""
        if evidence:
            n = _cite_index_for_e_tag(tag, evidence)
            if n is not None:
                return f"[{n}]"
        return ""

    t = _E_TAG_ANY_RE.sub(repl, raw)
    t = _E_TAG_LOOSE_RE.sub("", t)
    # Stub lines often left after tags: "Chroma." / "Web search." (no usable URL).
    t = re.sub(
        r"(?m)^\s*(Chroma|Web search)\.?\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def audit_numbered_citations(report: str, num_refs: int) -> tuple[bool, list[str]]:
    """Every [n] must satisfy 1 <= n <= num_refs (when num_refs > 0)."""
    issues: list[str] = []
    if num_refs <= 0:
        for m in _NUM_CIT_RE.finditer(report or ""):
            tok = m.group(1)
            if _is_likely_year_bracket(tok):
                continue
            issues.append(f"Citation [{tok}] invalid — no numbered references available.")
        return len(issues) == 0, issues
    for m in _NUM_CIT_RE.finditer(report or ""):
        tok = m.group(1)
        if _is_likely_year_bracket(tok):
            continue
        n = int(tok)
        if n < 1 or n > num_refs:
            issues.append(f"Invalid citation [{n}] — use 1..{num_refs} only.")
    return len(issues) == 0, issues
