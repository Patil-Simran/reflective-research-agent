"""LangGraph state schema with reducers."""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict


def evidence_reducer(left: list[dict], right: list[dict]) -> list[dict]:
    """Deduplicate by evidence id while concatenating."""
    seen: set[str] = {e["id"] for e in left}
    out = list(left)
    for item in right:
        eid = item.get("id")
        if eid and eid not in seen:
            seen.add(eid)
            out.append(item)
    return out


class ResearchState(TypedDict):
    """Graph state (checkpoint-serializable: use dicts for nested models)."""

    user_query: str
    plan: list[dict]  # PlanStep.model_dump()
    last_gather_plan_len: int  # only execute plan[last_gather_plan_len:] in gather
    evidence: Annotated[list[dict], evidence_reducer]  # EvidenceItem.model_dump()
    gather_count: int
    max_iterations: int
    need_more: bool
    revision_count: int  # revise passes after failed verification
    reflection_rationale: NotRequired[str]
    #: Optional distilled facts (with [n] cites) before long-form synthesis.
    evidence_brief: NotRequired[str]
    report: NotRequired[str]
    verification_passed: NotRequired[bool]
    verification_notes: NotRequired[str]
    errors: Annotated[list[str], operator.add]
