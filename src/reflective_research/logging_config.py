"""Structured-friendly logging setup."""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import UTC, datetime
from typing import Any

from reflective_research.config.settings import Settings


class JsonFormatter(logging.Formatter):
    """One JSON object per line for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings) -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*renamed to `ddgs`.*",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Pydantic V1.*",
        category=UserWarning,
        module="langchain_core._api.deprecation",
    )

    root = logging.getLogger()
    root.handlers.clear()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)

    # Quiet noisy third-party loggers in production
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
