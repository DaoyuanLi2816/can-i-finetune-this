"""Small wrapper to give the package a consistent logger and a JSON-safe encoder."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOGGER_NAME = "canifinetune"
_DEFAULT_LEVEL = os.environ.get("CANIFINETUNE_LOG_LEVEL", "INFO").upper()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger (or a child)."""
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
        handler.setFormatter(fmt)
        root.addHandler(handler)
        root.setLevel(_DEFAULT_LEVEL)
        root.propagate = False
    if name and name != _LOGGER_NAME:
        return root.getChild(name)
    return root


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp used in saved JSON results."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def to_json(obj: Any, *, indent: int = 2) -> str:
    """Dump ``obj`` to JSON with helpful default for paths / pydantic models."""
    return json.dumps(obj, indent=indent, default=_default, sort_keys=False)
