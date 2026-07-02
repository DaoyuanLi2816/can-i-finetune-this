"""Byte / parameter unit conversions used across the estimator."""

from __future__ import annotations

BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024


def bytes_to_gb(b: float) -> float:
    """Convert bytes to gibibytes (1024**3)."""
    return float(b) / BYTES_PER_GB


def gb_to_bytes(gb: float) -> float:
    """Convert gibibytes to bytes."""
    return float(gb) * BYTES_PER_GB


def bytes_to_mb(b: float) -> float:
    """Convert bytes to mebibytes."""
    return float(b) / BYTES_PER_MB


def humanize_bytes(b: float) -> str:
    """Render bytes as a short human-readable string (KiB / MiB / GiB)."""
    b = float(b)
    if abs(b) >= BYTES_PER_GB:
        return f"{b / BYTES_PER_GB:.2f} GiB"
    if abs(b) >= BYTES_PER_MB:
        return f"{b / BYTES_PER_MB:.2f} MiB"
    if abs(b) >= BYTES_PER_KB:
        return f"{b / BYTES_PER_KB:.2f} KiB"
    return f"{b:.0f} B"


# Canonical dtype tables. Values are bytes per scalar (or per parameter).
# Quantized formats include both the raw weight bytes and a separate
# `_quant_overhead` table for absmax / zero-point / lookup table storage,
# so callers can decide whether to lump them together.

DTYPE_BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "float32": 4.0,
    "tf32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "half": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "nf4": 0.5,
    "fp4": 0.5,
    "int4": 0.5,
}

# Approximate per-parameter overhead for storing quantization metadata
# (absmax scalars, optional zero points, block tables). Values are in bytes.
# References:
#  * NF4 with blocksize 64 and fp32 absmax: 4 bytes / 64 params ~= 0.0625 B/param
#  * NF4 + double quantization stores the absmax in int8 (1 byte / 64 params)
#    plus a small fp32 second-level block: ~0.017 B/param (measured via
#    bitsandbytes quant_state on Qwen2.5-1.5B).
#  * INT8 LLM.int8() / bitsandbytes Linear8bitLt keeps both the int8 weight
#    and a small fp16 outlier matrix; we approximate the overhead at 0.15 B/param
#    to cover that and statistics.
QUANT_OVERHEAD_BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 0.0,
    "float32": 0.0,
    "fp16": 0.0,
    "float16": 0.0,
    "bf16": 0.0,
    "bfloat16": 0.0,
    "fp8": 0.0,
    "int8": 0.15,
    "nf4": 0.0625,
    "nf4_double_quant": 0.017,
    "fp4": 0.0625,
    "int4": 0.0625,
}


def dtype_bytes(dtype: str) -> float:
    """Bytes per scalar for ``dtype`` (e.g. ``"bf16"`` → 2.0)."""
    key = dtype.lower()
    if key not in DTYPE_BYTES_PER_PARAM:
        raise ValueError(
            f"Unknown dtype {dtype!r}. Known: {sorted(DTYPE_BYTES_PER_PARAM)}"
        )
    return DTYPE_BYTES_PER_PARAM[key]


def quant_overhead(quantization: str) -> float:
    """Per-parameter quantization metadata overhead in bytes."""
    key = quantization.lower()
    return QUANT_OVERHEAD_BYTES_PER_PARAM.get(key, 0.0)
