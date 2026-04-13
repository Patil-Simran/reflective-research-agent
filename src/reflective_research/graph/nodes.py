"""LangGraph node implementations (pure-ish: side effects only via tools/LLM)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from reflective_research.config.settings import Settings, get_settings
from reflective_research.domain.models import (
    EvidenceItem,
    EvidenceResearchBrief,
    PlanStep,
    ReflectionDecision,
    ResearchPlan,
    VerificationOutcome,
)
from reflective_research.domain.state import ResearchState
from reflective_research.graph.citations import (
    audit_numbered_citations,
    build_numbered_evidence_for_prompt,
    sanitize_report_markdown,
)
from reflective_research.graph.prompts import (
    BRIEF_SYSTEM,
    PLAN_SYSTEM,
    REFLECT_SYSTEM,
    REVISE_SYSTEM,
    SYNTH_SYSTEM,
    VERIFY_SYSTEM,
)
from reflective_research.retrieval.chroma_store import similarity_search
from reflective_research.tools.evidence_quality import prepare_writer_evidence_pipeline
from reflective_research.tools.url_fetch import enrich_search_evidence
from reflective_research.tools.web_search import web_search_ddg

log = logging.getLogger(__name__)

def _prepare_evidence_for_llm(
    ev: list[dict[str, Any]],
    user_query: str,
    settings: Settings,
) -> tuple[list[dict[str, Any]], int]:
    raw = [e for e in ev if isinstance(e, dict)]
    usable, discarded = prepare_writer_evidence_pipeline(raw, user_query, settings)
    discarded += len(ev) - len(raw)
    return usable, discarded


def _execute_gather_step(
    step: dict,
    settings: Settings,
    embeddings: Any,
) -> list[EvidenceItem]:
    out: list[EvidenceItem] = []
    try:
        ps = PlanStep.model_validate(step)
    except Exception as e:
        return [
            EvidenceItem(
                id=f"bad-step-{hashlib.sha256(str(step).encode()).hexdigest()[:10]}",
                content=f"Invalid plan step skipped: {e}",
                source_type="error",
                source_ref="plan",
                metadata={"step": step},
            )
        ]
    if ps.tool == "search":
        found = web_search_ddg(settings, ps.query)
        out.extend(found)
        out.extend(enrich_search_evidence(settings, found))
    else:
        docs = similarity_search(settings, embeddings, ps.query, k=settings.rag_top_k)
        if not docs:
            out.append(
                EvidenceItem(
                    id=f"rag-empty-{hash(ps.query) % 10**8}",
                    content="RAG returned no chunks (empty corpus or no match).",
                    source_type="system",
                    source_ref="chroma",
                    metadata={"query": ps.query},
                )
            )
        for d in docs:
            src = str(d.metadata.get("source", "unknown"))
            key = f"{src}:{d.page_content[:120]}"
            eid = hashlib.sha256(key.encode()).hexdigest()[:20]
            out.append(
                EvidenceItem(
                    id=eid,
                    content=d.page_content[:6000],
                    source_type="rag",
                    source_ref=src,
                    metadata={"query": ps.query},
                )
            )
    return out


def _structured_invoke(
    llm: BaseChatModel,
    schema: type[Any],
    system: str,
    user: str,
) -> Any:
    """Invoke LLM with structured output; fall back to JSON extraction."""
    try:
        bound = llm.with_structured_output(schema)
        out = bound.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if isinstance(out, schema):
            return out
        if isinstance(out, dict):
            return schema.model_validate(out)
    except Exception as e:
        log.warning("Structured output failed (%s); trying JSON parse.", e)

    raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = getattr(raw, "content", str(raw))
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in text
        )
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"Could not parse JSON from model output: {text[:500]}")
    return schema.model_validate_json(match.group(0))


def make_plan_node(settings: Settings, llm: BaseChatModel) -> Callable[[ResearchState], dict[str, Any]]:
    def plan(state: ResearchState) -> dict[str, Any]:
        q = state["user_query"]
        user = f"USER QUESTION:\n{q}\n\nReturn a ResearchPlan JSON."
        try:
            plan_out = _structured_invoke(llm, ResearchPlan, PLAN_SYSTEM, user)
            steps = [s.model_dump() for s in plan_out.steps]
            log.info("Plan: %s steps", len(steps))
            return {"plan": steps, "gather_count": 0, "last_gather_plan_len": 0}
        except Exception as e:
            log.exception("Planning failed")
            fallback = PlanStep(
                tool="search",
                query=q,
                purpose="Fallback: direct web search for user question",
            )
            return {
                "plan": [fallback.model_dump()],
                "gather_count": 0,
                "last_gather_plan_len": 0,
                "errors": [f"plan_node: {e}"],
            }

    return plan


def make_gather_node(
    settings: Settings,
    embeddings: Any,
) -> Callable[[ResearchState], dict[str, Any]]:
    def gather(state: ResearchState) -> dict[str, Any]:
        steps: list[dict] = state["plan"]
        start = int(state.get("last_gather_plan_len", 0))
        pending = steps[start:]
        items: list[EvidenceItem] = []
        if not pending:
            items.append(
                EvidenceItem(
                    id=hashlib.sha256(b"noop-gather").hexdigest()[:16],
                    content="No new plan steps to execute this round (reflection requested more but "
                    "added no steps).",
                    source_type="system",
                    source_ref="gather",
                    metadata={},
                )
            )
        elif len(pending) <= 1 or settings.gather_parallelism <= 1:
            for step in pending:
                items.extend(_execute_gather_step(step, settings, embeddings))
        else:
            workers = min(settings.gather_parallelism, len(pending))

            def _run_step(s: dict) -> list[EvidenceItem]:
                return _execute_gather_step(s, settings, embeddings)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                chunk_lists = list(pool.map(_run_step, pending))
            for chunk in chunk_lists:
                items.extend(chunk)

        next_count = state["gather_count"] + 1
        log.info(
            "Gather round %s: %s new steps, %s evidence items",
            next_count,
            len(pending),
            len(items),
        )
        return {
            "evidence": [i.model_dump() for i in items],
            "gather_count": next_count,
            "last_gather_plan_len": len(steps),
        }

    return gather


def make_reflect_node(
    settings: Settings,
    llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    def reflect(state: ResearchState) -> dict[str, Any]:
        q = state["user_query"]
        plan_json = json.dumps(state["plan"], indent=2)[:12000]
        window = state["evidence"][-55:]
        ranked, _ = _prepare_evidence_for_llm(window, q, settings)
        sys_err = [e for e in window if e.get("source_type") in ("system", "error")]
        top_ids = {e.get("id") for e in ranked[:28]}

        def _noise_system_row(e: dict) -> bool:
            eid = str(e.get("id") or "")
            blob = f'{eid} {(e.get("content") or "")}'.lower()
            return bool(
                eid.startswith("rag-empty")
                or "rag returned no chunks" in blob
                or "empty corpus" in blob
                or "no new plan steps" in blob
            )

        tail_sys = [
            e
            for e in sys_err
            if e.get("id") not in top_ids and not _noise_system_row(e)
        ][:10]
        ev = ranked[:32] + tail_sys
        idx, ev_blob = build_numbered_evidence_for_prompt(ev)
        user = (
            f"USER QUESTION:\n{q}\n\nPLAN:\n{plan_json}\n\n"
            f"REFERENCE LIST (use these numbers when reasoning about gaps):\n{idx}\n\n"
            f"EVIDENCE JSON:\n{ev_blob[:14000]}"
        )
        try:
            out = _structured_invoke(llm, ReflectionDecision, REFLECT_SYSTEM, user)
        except Exception as e:
            log.exception("Reflection failed; defaulting to synthesize.")
            return {
                "need_more": False,
                "reflection_rationale": f"Reflection error: {e}",
                "errors": [f"reflect_node: {e}"],
            }

        new_plan = list(state["plan"])
        if out.need_more and out.new_steps:
            new_plan.extend([s.model_dump() for s in out.new_steps])

        log.info("Reflect: need_more=%s rationale=%s", out.need_more, out.rationale[:200])
        return {
            "need_more": out.need_more,
            "reflection_rationale": out.rationale,
            "plan": new_plan,
        }

    return reflect


def make_evidence_brief_node(
    settings: Settings,
    llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Extract grounded bullets (deep-research pattern) before synthesis."""

    def evidence_brief(state: ResearchState) -> dict[str, Any]:
        q = state["user_query"]
        ev = state["evidence"]
        usable, discarded = _prepare_evidence_for_llm(ev, q, settings)
        if not usable:
            msg = (
                f"(No usable references after filtering; discarded_rows≈{discarded}. "
                "Synthesis should state retrieval limits.)"
            )
            return {"evidence_brief": msg}
        idx, ev_blob = build_numbered_evidence_for_prompt(usable[:55])
        user = (
            f"USER QUESTION:\n{q}\n\n"
            f"REFERENCE LIST:\n{idx}\n\n"
            f"EVIDENCE JSON:\n{ev_blob[:14000]}"
        )
        try:
            out = _structured_invoke(llm, EvidenceResearchBrief, BRIEF_SYSTEM, user)
        except Exception as e:
            log.exception("Evidence brief extraction failed")
            return {
                "evidence_brief": "",
                "errors": [f"evidence_brief_node: {e}"],
            }
        lines = "\n".join(f"- {x}" for x in out.anchored_facts)
        cov = (out.coverage_note or "").strip()
        body = "## Anchored facts\n" + (lines if lines else "- (none)") + "\n\n## Coverage\n"
        body += cov if cov else "(none)"
        return {"evidence_brief": body[:8000]}

    return evidence_brief


def make_synthesize_node(
    settings: Settings,
    llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    def synthesize(state: ResearchState) -> dict[str, Any]:
        q = state["user_query"]
        ev = state["evidence"]
        usable, discarded = _prepare_evidence_for_llm(ev, q, settings)
        idx, ev_blob = build_numbered_evidence_for_prompt(usable[:80])
        rationale = state.get("reflection_rationale", "")
        brief = (state.get("evidence_brief") or "").strip()
        brief_block = (
            f"RESEARCH BRIEF (grounded extraction; EVIDENCE JSON below is authoritative):\n{brief}\n\n"
            if brief
            else ""
        )
        user = (
            f"USER QUESTION:\n{q}\n\n"
            f"REFLECTION NOTE:\n{rationale}\n\n"
            f"{brief_block}"
            f"REFERENCE COUNT: {len(usable)} "
            f"(discarded placeholder/error rows: {discarded})\n\n"
            f"REFERENCE LIST:\n{idx}\n\n"
            f"EVIDENCE JSON (cite = in-text number to use):\n{ev_blob[:18000]}\n\n"
            "Write the full Markdown research report per the system instructions. "
            "If references exist, prioritize depth: multiple sections, table(s), example(s), "
            "and a diagram (Mermaid or ASCII). Do not stop at a short overview."
        )
        try:
            msg = llm.invoke(
                [
                    SystemMessage(content=SYNTH_SYSTEM),
                    HumanMessage(content=user),
                ]
            )
            text = getattr(msg, "content", str(msg))
            if isinstance(text, list):
                text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in text
                )
        except Exception as e:
            log.exception("Synthesis failed")
            text = f"# Error\nSynthesis failed: {e}"
            return {
                "report": sanitize_report_markdown(text, usable[:80]),
                "errors": [f"synthesize_node: {e}"],
            }

        return {"report": sanitize_report_markdown(str(text), usable[:80])}

    return synthesize


def make_verify_node(
    settings: Settings,
    verifier_llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    def verify(state: ResearchState) -> dict[str, Any]:
        report = str(state.get("report") or "")
        usable, _ = _prepare_evidence_for_llm(state["evidence"], state["user_query"], settings)
        n = len(usable)
        cite_ok, cite_issues = audit_numbered_citations(report, n)
        _, usable_blob = build_numbered_evidence_for_prompt(usable[:50])
        user = (
            f"USER QUESTION:\n{state['user_query']}\n\n"
            f"CITATION AUDIT (programmatic, [1]..[{n}]): ok={cite_ok}\n"
            f"Issues:\n{chr(10).join(cite_issues) or '(none)'}\n\n"
            f"DRAFT REPORT:\n{report[:12000]}\n\n"
            f"EVIDENCE JSON:\n{usable_blob or '[]'}"
        )
        try:
            out = _structured_invoke(verifier_llm, VerificationOutcome, VERIFY_SYSTEM, user)
        except Exception as e:
            log.exception("Verification LLM failed; accepting draft with warning.")
            notes = f"Verifier error: {e}. Citation audit ok={cite_ok}."
            passed = cite_ok
            return {
                "verification_passed": passed,
                "verification_notes": notes,
                "errors": [f"verify_node: {e}"],
            }

        grounded_ok = out.grounded_ok
        llm_issues = list(out.unsupported_claims)
        passed = cite_ok and grounded_ok
        notes_parts = [out.summary.strip()] if out.summary.strip() else []
        if not cite_ok:
            notes_parts.append("Citation problems: " + "; ".join(cite_issues[:5]))
        if llm_issues:
            notes_parts.append("Grounding: " + "; ".join(llm_issues[:6]))
        notes = " | ".join(notes_parts) if notes_parts else ("OK" if passed else "Failed checks")

        log.info(
            "Verify: passed=%s cite_ok=%s grounded_ok=%s",
            passed,
            cite_ok,
            grounded_ok,
        )
        return {
            "verification_passed": passed,
            "verification_notes": notes,
        }

    return verify


def make_revise_node(
    settings: Settings,
    llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    def revise(state: ResearchState) -> dict[str, Any]:
        usable, _ = _prepare_evidence_for_llm(
            state["evidence"],
            state["user_query"],
            settings,
        )
        idx, ev_blob = build_numbered_evidence_for_prompt(usable[:60])
        feedback = state.get("verification_notes", "")
        draft = str(state.get("report") or "")
        user = (
            f"USER QUESTION:\n{state['user_query']}\n\n"
            f"VERIFICATION FEEDBACK:\n{feedback}\n\n"
            f"REFERENCE LIST:\n{idx}\n\n"
            f"EVIDENCE JSON:\n{ev_blob[:14000]}\n\n"
            f"DRAFT REPORT TO FIX:\n{draft[:14000]}"
        )
        try:
            msg = llm.invoke(
                [SystemMessage(content=REVISE_SYSTEM), HumanMessage(content=user)]
            )
            text = getattr(msg, "content", str(msg))
            if isinstance(text, list):
                text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in text
                )
        except Exception as e:
            log.exception("Revise failed")
            return {
                "errors": [f"revise_node: {e}"],
                "revision_count": state.get("revision_count", 0) + 1,
            }

        return {
            "report": sanitize_report_markdown(str(text), usable[:60]),
            "revision_count": state.get("revision_count", 0) + 1,
        }

    return revise


def make_finalize_node() -> Callable[[ResearchState], dict[str, Any]]:
    """Prepend banner when verification still failing after max revision rounds."""

    def finalize(state: ResearchState) -> dict[str, Any]:
        notes = (state.get("verification_notes") or "").strip() or (
            "Verification could not clear all issues within the revision budget."
        )
        usable, _ = _prepare_evidence_for_llm(
            state["evidence"],
            state["user_query"],
            get_settings(),
        )
        report = sanitize_report_markdown(str(state.get("report") or ""), usable[:80])
        banner = f"> **Verification (not fully cleared):** {notes}\n\n"
        return {"report": banner + report}

    return finalize


def make_route_after_verify(settings: Settings) -> Callable[[ResearchState], str]:
    def route_after_verify(state: ResearchState) -> str:
        if state.get("verification_passed"):
            return "end"
        rc = int(state.get("revision_count", 0))
        if rc >= settings.max_verification_revisions:
            return "finalize"
        return "revise"

    return route_after_verify


def make_route_after_reflect(settings: Settings) -> Callable[[ResearchState], str]:
    def route_after_reflect(state: ResearchState) -> str:
        if state.get("need_more") and state["gather_count"] < state["max_iterations"]:
            return "gather"
        return "brief" if settings.evidence_brief_enabled else "synthesize"

    return route_after_reflect
