"""FastAPI application: REST + Server-Sent Events for live research progress."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from reflective_research.api.schemas import ResearchRequest
from reflective_research.config.settings import Settings, get_settings
from reflective_research.llm.dependencies_health import readiness_report
from reflective_research.logging_config import configure_logging
from reflective_research.service import ResearchService

log = logging.getLogger(__name__)

_service: ResearchService | None = None


def get_service() -> ResearchService:
    global _service
    if _service is None:
        _service = ResearchService()
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    settings = get_settings()
    configure_logging(settings)
    # Warm singleton (optional; avoids first-request latency)
    try:
        get_service()
    except Exception:
        log.exception("Service warm-up failed — first request will retry.")
    yield
    if _service is not None:
        try:
            _service.shutdown()
        finally:
            _service = None


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(
        title="Reflective Research Agent",
        description="LangGraph research API with optional SSE streaming.",
        version="0.1.0",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in s.cors_origins.split(",") if o.strip()]
    cors_kw: dict[str, Any] = {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    if s.cors_allow_localhost_regex:
        # Vite often moves to 5174+ if 5173 is taken — no token needed, dev-only pattern.
        cors_kw["allow_origin_regex"] = r"https?://(127\.0\.0\.1|localhost)(:\d+)?$"
    app.add_middleware(CORSMiddleware, **cors_kw)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/health/ready")
    def health_ready() -> JSONResponse:
        """503 when required backends are misconfigured (Ollama down, Google key missing, …)."""
        st = get_settings()
        ok, body = readiness_report(st)
        payload = {"ready": ok, **body}
        if ok:
            return JSONResponse(payload)
        return JSONResponse(status_code=503, content=payload)

    @app.post("/api/research")
    def research_sync(body: ResearchRequest) -> dict[str, Any]:
        try:
            svc = get_service()
            raw = svc.run(body.query.strip(), thread_id=body.thread_id)
            tid = raw.pop("_thread_id", body.thread_id or "")
            return svc.result_for_api(raw, str(tid))
        except Exception as e:
            log.exception("research_sync failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/research/stream")
    def research_stream(body: ResearchRequest) -> StreamingResponse:
        svc = get_service()

        def event_generator():
            for ev in svc.stream_run(body.query.strip(), thread_id=body.thread_id):
                line = json.dumps(ev, default=str)
                yield f"data: {line}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
