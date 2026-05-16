"""Resolve a HF model id to the architecture fields the estimator needs.

We try three strategies in order:

1. A short curated table of well-known open models (so estimator works offline).
2. ``huggingface_hub.hf_hub_download`` to fetch only ``config.json`` (KB scale).
3. A user-supplied override blob.

All three return the same :class:`ModelMetadata` shape.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..utils.hf import fetch_model_config
from ..utils.logging import get_logger
from .formulas import ArchHints

log = get_logger("estimator.metadata")


@dataclass
class ModelMetadata:
    model_id: str
    family: str
    arch: ArchHints
    total_params: int
    source: str = "unknown"  # "known" | "hf-config" | "override" | "heuristic"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["arch"] = asdict(self.arch)
        return d


# ---------------------------------------------------------------------------
# Curated table of public open-weight models.
#
# Source of values: each model's ``config.json`` on the Hugging Face Hub at the
# time of writing. ``total_params`` is rounded to the nearest 0.01 B and
# verified against the model card.
# ---------------------------------------------------------------------------

KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "sshleifer/tiny-gpt2": {
        "family": "gpt2",
        "hidden_size": 2,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 4,
        "vocab_size": 50257,
        "total_params": 100_000,
    },
    "hf-internal-testing/tiny-random-LlamaForCausalLM": {
        "family": "llama",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 32000,
        "total_params": 1_000_000,
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "family": "qwen2",
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "total_params": 494_032_768,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "family": "qwen2",
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "total_params": 1_543_714_304,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "family": "qwen2",
        "hidden_size": 2048,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "intermediate_size": 11008,
        "vocab_size": 151936,
        "total_params": 3_085_938_688,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "family": "qwen2",
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "intermediate_size": 18944,
        "vocab_size": 152064,
        "total_params": 7_615_616_512,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "family": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "total_params": 8_030_261_248,
    },
    "mistralai/Mistral-7B-v0.1": {
        "family": "mistral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "total_params": 7_241_732_096,
    },
    "microsoft/phi-2": {
        "family": "phi",
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "intermediate_size": 10240,
        "vocab_size": 51200,
        "total_params": 2_779_683_840,
    },
    "google/gemma-2b": {
        "family": "gemma",
        "hidden_size": 2048,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "intermediate_size": 16384,
        "vocab_size": 256000,
        "total_params": 2_506_172_416,
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "family": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "vocab_size": 32000,
        "total_params": 1_100_048_384,
    },
}


_OVERRIDES: dict[str, dict[str, Any]] = {}


def register_known_model(model_id: str, spec: dict[str, Any]) -> None:
    """Add or override a model's spec (used in tests and by users)."""
    _OVERRIDES[model_id] = dict(spec)


def _from_spec(model_id: str, spec: dict[str, Any], *, source: str) -> ModelMetadata:
    arch = ArchHints(
        hidden_size=int(spec["hidden_size"]),
        num_hidden_layers=int(spec["num_hidden_layers"]),
        num_attention_heads=int(spec["num_attention_heads"]),
        num_key_value_heads=int(
            spec.get("num_key_value_heads", spec["num_attention_heads"])
        ),
        intermediate_size=int(spec["intermediate_size"]),
        vocab_size=int(spec["vocab_size"]),
    )
    total = int(spec.get("total_params") or _estimate_param_count(arch))
    return ModelMetadata(
        model_id=model_id,
        family=str(spec.get("family", _guess_family(model_id))),
        arch=arch,
        total_params=total,
        source=source,
        notes=str(spec.get("notes", "")),
    )


def _from_hf_config(model_id: str, cfg: dict[str, Any]) -> ModelMetadata | None:
    """Pull arch hints from a HF ``config.json`` payload."""
    try:
        hidden = int(
            cfg.get("hidden_size")
            or cfg.get("n_embd")
            or cfg.get("d_model")
        )
        layers = int(
            cfg.get("num_hidden_layers")
            or cfg.get("n_layer")
            or cfg.get("num_layers")
        )
        heads = int(
            cfg.get("num_attention_heads")
            or cfg.get("n_head")
            or cfg.get("num_heads")
        )
        kv = int(cfg.get("num_key_value_heads") or heads)
        ffn = int(
            cfg.get("intermediate_size")
            or cfg.get("ffn_dim")
            or cfg.get("n_inner")
            or 4 * hidden
        )
        vocab = int(cfg.get("vocab_size") or 32000)
    except (TypeError, ValueError) as e:
        log.warning("Cannot read arch from HF config for %s: %s", model_id, e)
        return None

    arch = ArchHints(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        intermediate_size=ffn,
        vocab_size=vocab,
    )
    total = int(_estimate_param_count(arch))
    family = str(cfg.get("model_type", _guess_family(model_id)))
    return ModelMetadata(
        model_id=model_id,
        family=family,
        arch=arch,
        total_params=total,
        source="hf-config",
        notes="param count estimated from arch; consult model card for exact value",
    )


def _estimate_param_count(arch: ArchHints) -> int:
    """Rough param count from arch hints (used when total_params is unknown)."""
    h = arch.hidden_size
    ff = arch.intermediate_size
    layers = arch.num_hidden_layers
    vocab = arch.vocab_size
    a = max(1, arch.num_attention_heads)
    kv = max(1, arch.num_key_value_heads)
    head_dim = h // a
    kv_dim = head_dim * kv
    # Per-layer params:
    #   attn: q (h*h) + k (h*kv) + v (h*kv) + o (h*h) = 2*h*h + 2*h*kv
    #   ffn (SwiGLU-style): gate (h*ff) + up (h*ff) + down (ff*h) = 3*h*ff
    #   norms: 2*h
    per_layer = 2 * h * h + 2 * h * kv_dim + 3 * h * ff + 2 * h
    # Embeddings (tied with lm_head in many models, but we count once and trust
    # the table for famous families).
    embeds = vocab * h
    final_norm = h
    return per_layer * layers + embeds + final_norm


_FAMILY_FROM_ID = [
    (re.compile(r"qwen", re.I), "qwen2"),
    (re.compile(r"llama|tinyllama", re.I), "llama"),
    (re.compile(r"mistral", re.I), "mistral"),
    (re.compile(r"phi", re.I), "phi"),
    (re.compile(r"gpt-?2|gpt2", re.I), "gpt2"),
    (re.compile(r"gemma", re.I), "gemma"),
]


def _guess_family(model_id: str) -> str:
    for pat, fam in _FAMILY_FROM_ID:
        if pat.search(model_id):
            return fam
    return "llama"


def fetch_metadata(
    model_id: str,
    *,
    override: dict[str, Any] | None = None,
    use_network: bool = True,
) -> ModelMetadata:
    """Resolve metadata for ``model_id``.

    Strategy: override > _OVERRIDES > KNOWN_MODELS > HF cache > HF download.

    Raises:
        ValueError: when no metadata can be resolved and the caller did not
        provide an override.
    """
    if override:
        return _from_spec(model_id, override, source="override")
    if model_id in _OVERRIDES:
        return _from_spec(model_id, _OVERRIDES[model_id], source="override")
    if model_id in KNOWN_MODELS:
        return _from_spec(model_id, KNOWN_MODELS[model_id], source="known")

    if use_network:
        cfg = fetch_model_config(model_id)
        if cfg is not None:
            md = _from_hf_config(model_id, cfg)
            if md is not None:
                return md

    raise ValueError(
        f"Cannot resolve metadata for {model_id!r}. Either pass --override "
        f"with hidden_size/num_hidden_layers/... or run with network access "
        f"so the HF config.json can be downloaded."
    )


@dataclass
class _Override:
    """Helper used by the CLI to bundle CLI flags into a spec dict."""

    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    total_params: int | None = None
    family: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> dict[str, Any] | None:
        d = {k: v for k, v in self.__dict__.items() if v is not None and k != "extras"}
        d.update(self.extras)
        required = {
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "vocab_size",
        }
        if not required.issubset(d):
            return None
        return d
