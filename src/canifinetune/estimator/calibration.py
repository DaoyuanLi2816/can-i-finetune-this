"""Load, save, and apply benchmark-derived calibration to estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platformdirs import user_cache_path
from pydantic import BaseModel, Field

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from .memory import EstimateRequest, MemoryBreakdown

log = get_logger("estimator.calibration")


class CalibrationSample(BaseModel):
    """One benchmark-derived (estimate, measured) pair."""

    gpu_name: str
    torch_version: str
    cuda_version: str
    model_family: str
    method: str
    seq_len: int
    micro_batch_size: int
    estimated_total_gb: float
    measured_total_gb: float

    @property
    def ratio(self) -> float:
        return self.measured_total_gb / max(1e-6, self.estimated_total_gb)


class Calibration(BaseModel):
    """A bundle of calibration samples + a derived multiplicative correction."""

    samples: list[CalibrationSample] = Field(default_factory=list)
    activation_scale: float = 1.0
    weights_scale: float = 1.0
    overhead_scale: float = 1.0
    note: str = ""

    def has_data(self) -> bool:
        return bool(self.samples)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def default_calibration_path() -> Path:
    """Per-user calibration cache (outside the repo)."""
    return user_cache_path("canifinetune") / "calibration.json"


def load_calibration(path: Path | None = None) -> Calibration:
    p = Path(path) if path else default_calibration_path()
    if not p.is_file():
        return Calibration()
    try:
        return Calibration.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read calibration %s: %s — using empty.", p, e)
        return Calibration()


def save_calibration(calib: Calibration, path: Path | None = None) -> Path:
    p = Path(path) if path else default_calibration_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(calib.model_dump_json(indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Apply calibration to an estimate
# ---------------------------------------------------------------------------

def _select_relevant(samples: list[CalibrationSample], request: EstimateRequest, family: str | None) -> list[CalibrationSample]:
    out = []
    for s in samples:
        if family and s.model_family.lower() != family.lower():
            continue
        if s.method.lower() != request.method.lower():
            continue
        out.append(s)
    return out


def fit_calibration_from_samples(samples: list[CalibrationSample]) -> Calibration:
    """Fit simple multiplicative scalars from a set of samples.

    Strategy: average of ``measured/estimated`` ratios per category. We split
    the correction crudely into "activation_scale" for short-seq vs "weights_scale"
    for long-seq, but in practice a single scale applied to total memory is the
    most reliable thing we can fit with few samples.
    """
    if not samples:
        return Calibration()
    ratios = [s.ratio for s in samples]
    mean_ratio = sum(ratios) / len(ratios)
    # Spread the correction across components proportionally. We bias the
    # activation scaling because activations are the noisiest component.
    activation_scale = float(mean_ratio)
    weights_scale = 1.0  # weights are easy to estimate exactly
    overhead_scale = 1.0
    return Calibration(
        samples=list(samples),
        activation_scale=activation_scale,
        weights_scale=weights_scale,
        overhead_scale=overhead_scale,
        note=(
            f"Fit from {len(samples)} sample(s); mean measured/estimated ratio "
            f"= {mean_ratio:.3f}. Applied as activation scale; "
            "estimator total = static_model + (activations*scale) + other."
        ),
    )


def apply_calibration(
    breakdown: MemoryBreakdown,
    calibration: Calibration | None,
    *,
    request: EstimateRequest = None,  # type: ignore[assignment]
) -> MemoryBreakdown:
    """Return ``breakdown`` adjusted by ``calibration``. No-op if unset."""
    if calibration is None or not calibration.has_data():
        return breakdown

    from .memory import MemoryBreakdown  # local import to avoid cycle

    s = calibration
    new_act = breakdown.activations_gb * s.activation_scale
    new_static = breakdown.static_model_gb * s.weights_scale
    new_overhead = breakdown.cuda_overhead_gb * s.overhead_scale
    delta = (new_act - breakdown.activations_gb) + (new_static - breakdown.static_model_gb) + (
        new_overhead - breakdown.cuda_overhead_gb
    )
    return MemoryBreakdown(
        static_model_gb=round(new_static, 4),
        quantization_overhead_gb=breakdown.quantization_overhead_gb,
        trainable_params_mb=breakdown.trainable_params_mb,
        gradients_gb=breakdown.gradients_gb,
        optimizer_gb=breakdown.optimizer_gb,
        activations_gb=round(new_act, 4),
        cuda_overhead_gb=round(new_overhead, 4),
        safety_margin_gb=breakdown.safety_margin_gb,
        total_estimated_gb=round(breakdown.total_estimated_gb + delta, 4),
    )


# ---------------------------------------------------------------------------
# Build calibration from benchmark-result JSON files
# ---------------------------------------------------------------------------

def calibration_from_result_files(paths: list[Path]) -> Calibration:
    """Read benchmark result JSONs and fit a calibration."""
    samples: list[CalibrationSample] = []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Skipping %s (cannot read): %s", p, e)
            continue
        sample = _result_to_sample(data)
        if sample:
            samples.append(sample)
    return fit_calibration_from_samples(samples)


def _result_to_sample(data: dict[str, Any]) -> CalibrationSample | None:
    try:
        return CalibrationSample(
            gpu_name=str(data.get("gpu", {}).get("name") or "unknown"),
            torch_version=str(data.get("env", {}).get("torch_version") or ""),
            cuda_version=str(data.get("env", {}).get("cuda_version") or ""),
            model_family=str(data.get("model_family") or "unknown"),
            method=str(data.get("method") or "unknown"),
            seq_len=int(data.get("config", {}).get("seq_len") or 0),
            micro_batch_size=int(data.get("config", {}).get("micro_batch_size") or 1),
            estimated_total_gb=float(data.get("estimated_total_gb") or 0.0),
            measured_total_gb=float(
                data.get("measured", {}).get("peak_total_gb") or 0.0
            ),
        )
    except Exception as e:
        log.warning("Skipping result (incomplete): %s", e)
        return None
