"""API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=16_000)
    thread_id: str | None = Field(
        default=None,
        description="Optional thread id for checkpoint isolation (same id = same LangGraph thread).",
    )
