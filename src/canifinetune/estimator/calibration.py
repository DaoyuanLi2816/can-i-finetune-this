"""Load, save, and apply benchmark-derived calibration to estimates."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platformdirs import user_cache_path
from pydantic import BaseModel, Field

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from .memory import MemoryBreakdown

log = get_logger("estimator.calibration")


class CalibrationSample(BaseModel):
    """One benchmark-derived (estimate, measured) pair."""

    config_signature: dict[str, Any] = Field(default_factory=dict)
    estimator_version: str = "0.4.0"
    gpu_name: str
    torch_version: str
    cuda_version: str
    model_family: str
    method: str
    seq_len: int
    micro_batch_size: int
    estimated_total_gb: float
    measured_total_gb: float
    gpu_vram_gb: float = 0.0
    estimated_static_gb: float = 0.0
    estimated_dynamic_gb: float = 0.0
    estimated_cuda_overhead_gb: float = 0.0
    measured_allocated_gb: float = 0.0
    measured_reserved_gb: float = 0.0

    @property
    def ratio(self) -> float:
        if self.estimated_dynamic_gb > 0 and self.measured_allocated_gb > 0:
            observed_dynamic = self.measured_allocated_gb - self.estimated_static_gb
            return observed_dynamic / self.estimated_dynamic_gb
        return self.measured_total_gb / max(1e-6, self.estimated_total_gb)

    @property
    def overhead_ratio(self) -> float | None:
        if self.estimated_cuda_overhead_gb <= 0 or self.measured_reserved_gb <= 0:
            return None
        allocator_gap = max(0.0, self.measured_reserved_gb - self.measured_allocated_gb)
        return allocator_gap / self.estimated_cuda_overhead_gb


class Calibration(BaseModel):
    """A bundle of calibration samples + a derived multiplicative correction."""

    schema_version: int = 3
    estimator_version: str = "0.4.0"
    configuration_profiles: list[dict[str, Any]] = Field(default_factory=list)
    samples: list[CalibrationSample] = Field(default_factory=list)
    activation_scale: float = 1.0
    weights_scale: float = 1.0
    overhead_scale: float = 1.0
    model_families: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    gpu_vram_gb: float = 0.0
    note: str = ""

    def has_data(self) -> bool:
        return bool(self.samples)

    def is_compatible(
        self,
        *,
        model_family: str,
        method: str,
        gpu_vram_gb: float,
        config_signature: dict | None = None,
    ) -> bool:
        if self.estimator_version != "0.4.0":
            return False
        if self.configuration_profiles and config_signature not in self.configuration_profiles:
            return False
        if not self.has_data():
            return False
        if self.model_families and model_family not in self.model_families:
            return False
        if self.methods and method not in self.methods:
            return False
        return not (
            self.gpu_vram_gb and abs(gpu_vram_gb - self.gpu_vram_gb) / self.gpu_vram_gb > 0.15
        )


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
        raw = p.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if "schema_version" not in payload:
            log.warning(
                "Ignoring legacy calibration %s; rerun `canifinetune calibrate` "
                "to create component-aware schema v2 data.",
                p,
            )
            return Calibration(note="Legacy calibration ignored; rerun `canifinetune calibrate`.")
        if payload.get("schema_version") != 3:
            return Calibration(
                note="Legacy calibration ignored; rerun calibrate with matching new measurements."
            )
        return Calibration.model_validate(payload)
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


def fit_calibration_from_samples(samples: list[CalibrationSample]) -> Calibration:
    """Fit simple multiplicative scalars from a set of samples.

    Static weights, gradients, and optimizer states are deterministic enough
    to leave untouched. Fit the dynamic residual against
    ``max_memory_allocated`` and the allocator gap against
    ``max_memory_reserved - max_memory_allocated``. Legacy samples without a
    component breakdown fall back to their total ratio.
    """
    if not samples:
        return Calibration()
    ratios = [s.ratio for s in samples if math.isfinite(s.ratio) and s.ratio > 0]
    if not ratios:
        return Calibration(samples=list(samples), note="No usable calibration ratios.")
    activation_scale = float(statistics.median(ratios))
    overhead_ratios = [
        ratio
        for sample in samples
        if (ratio := sample.overhead_ratio) is not None and math.isfinite(ratio)
    ]
    overhead_scale = float(statistics.median(overhead_ratios)) if overhead_ratios else 1.0
    gpu_sizes = [s.gpu_vram_gb for s in samples if s.gpu_vram_gb > 0]
    return Calibration(
        samples=list(samples),
        activation_scale=activation_scale,
        weights_scale=1.0,
        overhead_scale=overhead_scale,
        configuration_profiles=[s.config_signature for s in samples if s.config_signature],
        estimator_version=samples[0].estimator_version
        if len({s.estimator_version for s in samples}) == 1
        else "mixed",
        model_families=sorted({s.model_family for s in samples if s.model_family}),
        methods=sorted({s.method for s in samples if s.method}),
        gpu_vram_gb=float(statistics.median(gpu_sizes)) if gpu_sizes else 0.0,
        note=(
            f"Fit from {len(samples)} sample(s); median dynamic scale "
            f"= {activation_scale:.3f}, allocator-overhead scale = {overhead_scale:.3f}. "
            "Static memory and safety margin are not scaled."
        ),
    )


def apply_calibration(
    breakdown: MemoryBreakdown,
    calibration: Calibration | None,
    *,
    model_family: str = "",
    method: str = "",
    gpu_vram_gb: float = 0.0,
    config_signature: dict | None = None,
) -> MemoryBreakdown:
    """Return ``breakdown`` adjusted by ``calibration``. No-op if unset."""
    if calibration is None or not calibration.is_compatible(
        model_family=model_family,
        method=method,
        gpu_vram_gb=gpu_vram_gb,
        config_signature=config_signature,
    ):
        return breakdown

    from .memory import MemoryBreakdown  # local import to avoid cycle

    s = calibration
    new_act = breakdown.activations_gb * s.activation_scale
    new_logits = breakdown.logits_gb * s.activation_scale
    new_overhead = breakdown.cuda_overhead_gb * s.overhead_scale
    delta = (
        (new_act - breakdown.activations_gb)
        + (new_logits - breakdown.logits_gb)
        + (new_overhead - breakdown.cuda_overhead_gb)
    )
    return MemoryBreakdown(
        static_model_gb=breakdown.static_model_gb,
        quantization_overhead_gb=breakdown.quantization_overhead_gb,
        trainable_params_mb=breakdown.trainable_params_mb,
        gradients_gb=breakdown.gradients_gb,
        optimizer_gb=breakdown.optimizer_gb,
        activations_gb=round(new_act, 4),
        logits_gb=round(new_logits, 4),
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
    if not isinstance(data, dict):
        return None
    if data.get("success") is False or data.get("oom", {}).get("happened"):
        return None
    if data.get("provenance", {}).get("source") == "community":
        return None
    if data.get("schema_version") == 2:
        if data.get("status") != "success" or data.get("estimator_version") != "0.4.0":
            return None
        if not data.get("effective_configuration") or data.get("config", {}).get("forward_only"):
            return None
        if data.get("completed_steps") != data.get("config", {}).get("steps") or not data.get(
            "estimated_breakdown"
        ):
            return None
        if not all(
            data.get("measured", {}).get(key, 0) > 0
            for key in ("peak_allocated_gb", "peak_reserved_gb")
        ):
            return None
        if data.get("config", {}).get("loss_backend", "stock") != "stock":
            return None
        if data.get("provenance", {}).get("cohort") == "validation":
            log.warning("Validation cohort is reserved for evaluation; cannot fit it.")
            return None
    measured = data.get("measured", {})
    measured_allocated = float(measured.get("peak_allocated_gb") or 0.0)
    measured_reserved = float(
        measured.get("peak_reserved_gb") or measured.get("peak_total_gb") or 0.0
    )
    measured_gb = measured_reserved or measured_allocated
    if measured_gb < 0.3:
        # Tiny smoke runs (e.g. sshleifer/tiny-gpt2) are dominated by fixed
        # overheads and would skew the fitted ratio.
        log.info("Skipping smoke-scale result (%.2f GB measured) for calibration.", measured_gb)
        return None
    try:
        cfg = data.get("config", {})
        gpu = data.get("gpu", {})
        breakdown = dict(data.get("estimated_breakdown") or {})
        static_gb = sum(
            float(breakdown.get(key) or 0.0)
            for key in (
                "static_model_gb",
                "quantization_overhead_gb",
                "gradients_gb",
                "optimizer_gb",
            )
        )
        dynamic_gb = float(breakdown.get("activations_gb") or 0.0) + float(
            breakdown.get("logits_gb") or 0.0
        )
        return CalibrationSample(
            estimator_version=data.get("estimator_version", "legacy"),
            config_signature=calibration_signature(data.get("effective_configuration") or cfg)
            if data.get("schema_version") == 2
            else {},
            gpu_name=str(gpu.get("name") or "unknown"),
            torch_version=str(data.get("env", {}).get("torch_version") or ""),
            cuda_version=str(data.get("env", {}).get("cuda_version") or ""),
            model_family=str(data.get("model_family") or "unknown"),
            method=str(data.get("method") or "unknown"),
            seq_len=int(cfg.get("seq_len") or 0),
            micro_batch_size=int(cfg.get("micro_batch_size") or 1),
            estimated_total_gb=float(data.get("estimated_total_gb") or 0.0),
            measured_total_gb=float(
                measured.get("peak_total_gb") or measured_reserved or measured_allocated
            ),
            gpu_vram_gb=float(gpu.get("total_vram_gb") or 0.0),
            estimated_static_gb=static_gb,
            estimated_dynamic_gb=dynamic_gb,
            estimated_cuda_overhead_gb=float(breakdown.get("cuda_overhead_gb") or 0.0),
            measured_allocated_gb=measured_allocated or measured_gb,
            measured_reserved_gb=measured_reserved or measured_gb,
        )
    except Exception as e:
        log.warning("Skipping result (incomplete): %s", e)
        return None


def calibration_signature(config: dict) -> dict:
    return {
        key: config.get(key)
        for key in (
            "base_dtype",
            "quantization",
            "optimizer",
            "gradient_checkpointing",
            "attention_implementation",
            "loss_backend",
            "lora_target_scope",
            "seq_len",
            "micro_batch_size",
            "lora_rank",
            "training_backend",
            "gradient_accumulation_steps",
            "lora_alpha",
            "lora_dropout",
            "loss_mode",
        )
    }
