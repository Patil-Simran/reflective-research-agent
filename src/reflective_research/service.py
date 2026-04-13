"""Application service: wiring, invoke, thread isolation."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from reflective_research.config.settings import Settings, get_settings
from reflective_research.graph.builder import build_research_graph
from reflective_research.graph.citations import cited_reference_numbers_from_report
from reflective_research.tools.evidence_quality import prepare_writer_evidence_pipeline
from reflective_research.llm.factory import (
    get_chat_model,
    get_embeddings,
    get_verifier_chat_model,
    get_writer_chat_model,
)
from reflective_research.llm.ollama_health import assert_ollama_reachable
from reflective_research.logging_config import configure_logging

log = logging.getLogger(__name__)


def build_api_evidence_list(state: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    """
    Evidence for JSON/SSE clients: same pipeline as the writer, stable ``writer_cite`` 1..n,
    optionally restricted to in-text [n] markers (answer-style UIs).
    """
    raw_ev = state.get("evidence", [])
    ev_list = raw_ev if isinstance(raw_ev, list) else []
    q = str(state.get("user_query") or "")
    evidence_out, _ = prepare_writer_evidence_pipeline(
        [e for e in ev_list if isinstance(e, dict)],
        q,
        settings,
    )
    annotated: list[dict[str, Any]] = []
    for i, e in enumerate(evidence_out, start=1):
        row = dict(e)
        row["writer_cite"] = i
        annotated.append(row)

    report_text = str(state.get("report") or "")
    if settings.api_evidence_cited_only and report_text.strip():
        cited = cited_reference_numbers_from_report(report_text)
        if cited:
            slim = [r for r in annotated if int(r.get("writer_cite") or 0) in cited]
            if slim:
                annotated = sorted(slim, key=lambda x: int(x.get("writer_cite") or 0))
    return annotated


class ResearchService:
    """Production entry: one instance per process (or pool workers)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        checkpointer: MemorySaver | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings)
        w = (self.settings.ollama_writer_model or self.settings.ollama_model).strip()
        v = (
            (self.settings.ollama_verifier_model or self.settings.ollama_model)
            if self.settings.verification_enabled
            else "off"
        )
        log.info(
            "Stack: %s plan=%s writer=%s verify=%s embed=%s",
            self.settings.ollama_base_url,
            self.settings.ollama_model,
            w,
            v,
            self.settings.embedding_provider,
        )
        assert_ollama_reachable(self.settings)
        self._llm = get_chat_model(self.settings)
        self._writer_llm = get_writer_chat_model(self.settings)
        self._verifier_llm = (
            get_verifier_chat_model(self.settings) if self.settings.verification_enabled else None
        )
        self._embeddings = get_embeddings(self.settings)
        self._graph = build_research_graph(
            self.settings,
            self._llm,
            self._embeddings,
            writer_llm=self._writer_llm,
            verifier_llm=self._verifier_llm,
            checkpointer=checkpointer,
        )

    def _preflight_backends(self) -> None:
        """Re-check before each run (Ollama may stop; avoids WinError 10061 on stale clients)."""
        assert_ollama_reachable(self.settings)

    def run(self, user_query: str, *, thread_id: str | None = None) -> dict[str, Any]:
        """Execute full plan → gather → reflect loop → report."""
        self._preflight_backends()
        tid = thread_id or str(uuid.uuid4())
        config: dict[str, Any] = {"configurable": {"thread_id": tid}}
        initial: dict[str, Any] = {
            "user_query": user_query.strip(),
            "plan": [],
            "last_gather_plan_len": 0,
            "evidence": [],
            "gather_count": 0,
            "max_iterations": self.settings.max_reflection_iterations,
            "need_more": False,
            "revision_count": 0,
            "errors": [],
        }
        log.info("Starting research thread_id=%s", tid)
        result = self._graph.invoke(initial, config)
        result["_thread_id"] = tid
        return result

    def shutdown(self) -> None:
        """Close underlying HTTP clients (Ollama httpx) to avoid ResourceWarning on exit."""
        for attr in ("_llm", "_writer_llm", "_verifier_llm", "_embeddings"):
            lc = getattr(self, attr, None)
            if lc is None:
                continue
            ollama_wrapped = getattr(lc, "_client", None)
            if ollama_wrapped is not None:
                inner = getattr(ollama_wrapped, "_client", None)
                if inner is not None and hasattr(inner, "close"):
                    try:
                        inner.close()
                    except Exception:
                        log.debug("shutdown: httpx close failed for %s", attr, exc_info=True)

    @staticmethod
    def state_summary(state: dict[str, Any]) -> dict[str, Any]:
        """Lightweight progress payload for SSE (avoids huge evidence in every tick)."""
        return {
            "gather_round": state.get("gather_count", 0),
            "plan_steps": len(state.get("plan", [])),
            "evidence_items": len(state.get("evidence", [])),
            "has_report": bool(state.get("report")),
            "need_more": state.get("need_more"),
            "verification_passed": state.get("verification_passed"),
            "revision_count": state.get("revision_count", 0),
        }

    def result_for_api(self, state: dict[str, Any], thread_id: str) -> dict[str, Any]:
        """Stable JSON shape for clients."""
        return {
            "thread_id": thread_id,
            "user_query": state.get("user_query"),
            "report": state.get("report"),
            "errors": list(state.get("errors", [])),
            "plan": state.get("plan", []),
            "evidence": build_api_evidence_list(state, self.settings),
            "evidence_brief": state.get("evidence_brief"),
            "reflection_rationale": state.get("reflection_rationale"),
            "gather_count": state.get("gather_count"),
            "max_iterations": state.get("max_iterations"),
            "verification_passed": state.get("verification_passed"),
            "verification_notes": state.get("verification_notes"),
            "revision_count": state.get("revision_count", 0),
        }

    def stream_run(
        self,
        user_query: str,
        *,
        thread_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield SSE-friendly events; last event is type ``complete`` or ``error``."""
        tid = thread_id or str(uuid.uuid4())
        config: dict[str, Any] = {"configurable": {"thread_id": tid}}
        initial: dict[str, Any] = {
            "user_query": user_query.strip(),
            "plan": [],
            "last_gather_plan_len": 0,
            "evidence": [],
            "gather_count": 0,
            "max_iterations": self.settings.max_reflection_iterations,
            "need_more": False,
            "revision_count": 0,
            "errors": [],
        }
        log.info("Streaming research thread_id=%s", tid)
        yield {"type": "started", "thread_id": tid}
        try:
            self._preflight_backends()
        except RuntimeError as e:
            yield {"type": "error", "thread_id": tid, "message": str(e)}
            return
        last: dict[str, Any] = {}
        try:
            for state in self._graph.stream(initial, config, stream_mode="values"):
                if isinstance(state, dict):
                    last = state
                    yield {
                        "type": "progress",
                        "thread_id": tid,
                        "summary": self.state_summary(state),
                    }
            yield {
                "type": "complete",
                "thread_id": tid,
                "result": self.result_for_api(last, tid),
            }
        except Exception as e:
            log.exception("stream_run failed thread_id=%s", tid)
            yield {"type": "error", "thread_id": tid, "message": str(e)}
