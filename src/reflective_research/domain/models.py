"""Pydantic contracts for LLM structured I/O."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One unit of work in the research plan."""

    tool: Literal["search", "rag"] = Field(description="search = web, rag = vector store")
    query: str = Field(min_length=1, max_length=2000)
    purpose: str = Field(
        default="",
        description="Why this step exists (helps reflection and synthesis)",
    )


class ResearchPlan(BaseModel):
    steps: list[PlanStep] = Field(
        min_length=1,
        max_length=16,
        description="Prefer 6–12 diverse steps for in-depth topics (angles, comparisons, edge cases).",
    )


class ReflectionDecision(BaseModel):
    need_more: bool = Field(description="True if more retrieval is required before answering")
    rationale: str = Field(description="What is missing, contradictory, or weak")
    new_steps: list[PlanStep] = Field(
        default_factory=list,
        description="Additional steps if need_more; empty otherwise",
    )


class EvidenceItem(BaseModel):
    """Single piece of evidence the synthesizer can cite."""

    id: str
    content: str
    source_type: Literal["search", "rag", "system", "error"]
    source_ref: str = Field(description="URL, file path, or label")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceResearchBrief(BaseModel):
    """Intermediate extraction: grounded bullets only (deep-research style)."""

    anchored_facts: list[str] = Field(
        default_factory=list,
        max_length=48,
        description="Each line: one short claim supported by evidence; must include [n] citations.",
    )
    coverage_note: str = Field(
        default="",
        max_length=1200,
        description="Optional 1–3 sentences on what the snippets do not cover.",
    )


class VerificationOutcome(BaseModel):
    """Second-pass (cross-model) review of the draft report vs evidence."""

    grounded_ok: bool = Field(
        description="True only if no factual claims appear that are not supported by evidence text",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Short bullets naming unsupported or overstated claims",
    )
    summary: str = Field(
        default="",
        description="One-line verdict for logs / reviser",
    )
