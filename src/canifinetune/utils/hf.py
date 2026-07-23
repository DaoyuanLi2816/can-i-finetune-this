"""Hugging Face helpers that work without huggingface_hub being authenticated."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .logging import get_logger

log = get_logger("utils.hf")


def hf_cache_dir() -> Path:
    """Best-effort HF cache root (respects ``HF_HOME`` if set)."""
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"])
    return Path.home() / ".cache" / "huggingface"


def fetch_model_config(model_id: str, *, revision: str = "main") -> dict[str, Any] | None:
    """Return the model's ``config.json`` as a dict.

    Tries (in order):
      1. The local Hub cache (no network).
      2. ``huggingface_hub.hf_hub_download`` if available.

    Returns ``None`` if the config can't be obtained. The caller should fall
    back to its own known-model table or to a manual override.
    """
    # 1) Look for a cached snapshot.
    try:
        from huggingface_hub import try_to_load_from_cache

        local = try_to_load_from_cache(repo_id=model_id, filename="config.json", revision=revision)
        if isinstance(local, str) and Path(local).is_file():
            return _read_json(Path(local))
    except Exception as e:  # pragma: no cover - optional path
        log.debug("try_to_load_from_cache failed for %s: %s", model_id, e)

    # 2) Download via huggingface_hub if installed.
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=revision,
        )
        return _read_json(Path(path))
    except Exception as e:
        log.info("hf_hub_download config.json failed for %s: %s", model_id, e)
        return None


@lru_cache(maxsize=128)
def fetch_model_parameter_count(model_id: str, *, revision: str = "main") -> int | None:
    """Return the exact safetensors parameter count without downloading weights.

    The Hub exposes aggregate tensor metadata through ``model_info``. Older Hub
    clients may not support ``expand``, so retry with their default response
    shape before giving up.
    """
    try:
        from huggingface_hub import model_info

        try:
            info = model_info(model_id, revision=revision, expand=["safetensors"])
        except TypeError:  # huggingface_hub before ``expand`` support
            info = model_info(model_id, revision=revision)
        safetensors = getattr(info, "safetensors", None)
        if safetensors is None:
            return None
        total = getattr(safetensors, "total", None)
        if total:
            return int(total)
        parameters = getattr(safetensors, "parameters", None) or {}
        if parameters:
            return int(sum(int(v) for v in parameters.values()))
    except Exception as e:
        log.info("Hub parameter metadata failed for %s: %s", model_id, e)
    return None


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))
