"""Compile the LangGraph workflow."""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from reflective_research.config.settings import Settings
from reflective_research.domain.state import ResearchState
from reflective_research.graph.nodes import (
    make_evidence_brief_node,
    make_finalize_node,
    make_gather_node,
    make_plan_node,
    make_reflect_node,
    make_revise_node,
    make_route_after_reflect,
    make_route_after_verify,
    make_synthesize_node,
    make_verify_node,
)

log = logging.getLogger(__name__)


def build_research_graph(
    settings: Settings,
    llm: BaseChatModel,
    embeddings: Embeddings,
    *,
    writer_llm: BaseChatModel | None = None,
    verifier_llm: BaseChatModel | None = None,
    checkpointer: MemorySaver | None = None,
):
    write = writer_llm or llm
    graph = StateGraph(ResearchState)
    graph.add_node("plan", make_plan_node(settings, llm))
    graph.add_node("gather", make_gather_node(settings, embeddings))
    graph.add_node("reflect", make_reflect_node(settings, llm))
    graph.add_node("synthesize", make_synthesize_node(settings, write))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "gather")
    graph.add_edge("gather", "reflect")
    reflect_targets: dict[str, str] = {"gather": "gather"}
    if settings.evidence_brief_enabled:
        graph.add_node("brief", make_evidence_brief_node(settings, llm))
        graph.add_edge("brief", "synthesize")
        reflect_targets["brief"] = "brief"
    else:
        reflect_targets["synthesize"] = "synthesize"
    graph.add_conditional_edges(
        "reflect",
        make_route_after_reflect(settings),
        reflect_targets,
    )

    if settings.verification_enabled and verifier_llm is not None:
        graph.add_node("verify", make_verify_node(settings, verifier_llm))
        graph.add_node("revise", make_revise_node(settings, write))
        graph.add_node("finalize", make_finalize_node())
        graph.add_edge("synthesize", "verify")
        graph.add_conditional_edges(
            "verify",
            make_route_after_verify(settings),
            {
                "end": END,
                "revise": "revise",
                "finalize": "finalize",
            },
        )
        graph.add_edge("revise", "verify")
        graph.add_edge("finalize", END)
        log.info(
            "Graph: verify + revise loop enabled (max revisions=%s)",
            settings.max_verification_revisions,
        )
    else:
        graph.add_edge("synthesize", END)
        if not settings.verification_enabled:
            log.info("Graph: verification disabled (synthesize -> END)")
        else:
            log.warning(
                "Graph: verification on but no verifier_llm; synthesize -> END",
            )

    cp = checkpointer or MemorySaver()
    compiled = graph.compile(checkpointer=cp)
    log.info("Research graph compiled (checkpointer=%s)", type(cp).__name__)
    return compiled
