"""High-level ``estimate()`` API used by the CLI and the recommender."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..utils.units import bytes_to_gb
from . import formulas as F
from .calibration import Calibration, apply_calibration
from .model_metadata import ModelMetadata, fetch_metadata

Method = Literal["full", "lora", "qlora"]
Confidence = Literal["low", "medium", "high"]


class EstimateRequest(BaseModel):
    """Inputs to :func:`estimate`. Validated with pydantic."""

    model_id: str = Field(..., description="HF model id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    method: Method = "qlora"
    gpu_vram_gb: float = Field(..., gt=0.0, description="Total GPU VRAM in gibibytes.")
    seq_len: int = Field(2048, gt=0)
    micro_batch_size: int = Field(1, gt=0)
    gradient_accumulation_steps: int = Field(1, gt=0)

    base_dtype: str = "bf16"
    quantization: str = "nf4_double_quant"  # only used for qlora
    lora_rank: int = 16
    lora_target_scope: Literal["attention", "all_linear", "conservative"] = "attention"

    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    attention_implementation: str = "sdpa"
    activation_dtype: str = "bf16"

    # If you've cached calibration, the CLI passes it in; library users
    # can pass a :class:`Calibration` directly.
    calibration: Calibration | None = None

    # Optional override dict for unknown models (hidden_size, ...).
    override: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _normalize_method(self) -> EstimateRequest:
        if self.method == "full":
            self.quantization = "none"
        if self.method == "lora" and self.quantization.lower() not in {"none", "fp16", "bf16", "fp32"}:
            # LoRA on non-quantized base.
            self.quantization = "bf16"
        return self


class MemoryBreakdown(BaseModel):
    static_model_gb: float
    quantization_overhead_gb: float
    trainable_params_mb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    cuda_overhead_gb: float
    safety_margin_gb: float
    total_estimated_gb: float


class Estimate(BaseModel):
    request: EstimateRequest
    metadata: dict[str, Any]
    memory: MemoryBreakdown
    feasible: Literal["yes", "marginal", "no"]
    feasibility_ratio: float
    confidence: Confidence
    assumptions: list[str]
    warnings: list[str] = Field(default_factory=list)
    calibration_applied: bool = False


def _compute_breakdown(req: EstimateRequest, md: ModelMetadata) -> MemoryBreakdown:
    arch = md.arch
    total_params = md.total_params

    # 1) Weights.
    if req.method == "qlora":
        weights_b = F.weights_bytes(
            num_params=total_params,
            base_dtype=req.base_dtype,
            quantization=req.quantization,
        )
        weights_baseline = F.weights_bytes(
            num_params=total_params, base_dtype=req.base_dtype, quantization=None
        )
        quant_overhead_b = max(0.0, weights_baseline * 0.0)  # baseline only used for ratio
        # The overhead vs. raw int4 storage is captured inside weights_bytes,
        # but we surface a separate "quantization_overhead" number relative to
        # the raw quantized weight count.
        from ..utils.units import dtype_bytes
        raw_q = total_params * dtype_bytes("nf4")
        quant_overhead_b = max(0.0, weights_b - raw_q)
    elif req.method == "lora":
        weights_b = F.weights_bytes(
            num_params=total_params,
            base_dtype=req.base_dtype,
            quantization=None,
        )
        quant_overhead_b = 0.0
    else:  # full
        weights_b = F.weights_bytes(
            num_params=total_params, base_dtype=req.base_dtype, quantization=None
        )
        quant_overhead_b = 0.0

    # 2) Trainable params.
    if req.method == "full":
        trainable_params = total_params
    else:
        trainable_params = F.lora_trainable_params(
            arch=arch,
            family=md.family,
            rank=req.lora_rank,
            scope=req.lora_target_scope,
        )

    # 3) Gradients.
    grad_b = F.gradients_bytes(trainable_params=trainable_params, grad_dtype="bf16")

    # 4) Optimizer states.
    opt_b = F.optimizer_bytes(trainable_params=trainable_params, optimizer=req.optimizer)

    # 5) Activations.
    act_b = F.activations_bytes(
        seq_len=req.seq_len,
        batch_size=req.micro_batch_size,
        arch=arch,
        activation_dtype=req.activation_dtype,
        use_gradient_checkpointing=req.gradient_checkpointing,
        attention_implementation=req.attention_implementation,
    )

    # 6) CUDA / fragmentation / safety.
    cuda_b = F.cuda_overhead_bytes(available_vram_gb=req.gpu_vram_gb)
    safety_b = F.safety_margin_bytes(available_vram_gb=req.gpu_vram_gb)

    total_b = weights_b + grad_b + opt_b + act_b + cuda_b + safety_b

    return MemoryBreakdown(
        static_model_gb=round(bytes_to_gb(weights_b), 4),
        quantization_overhead_gb=round(bytes_to_gb(quant_overhead_b), 4),
        trainable_params_mb=round(trainable_params / 1e6, 3),
        gradients_gb=round(bytes_to_gb(grad_b), 4),
        optimizer_gb=round(bytes_to_gb(opt_b), 4),
        activations_gb=round(bytes_to_gb(act_b), 4),
        cuda_overhead_gb=round(bytes_to_gb(cuda_b), 4),
        safety_margin_gb=round(bytes_to_gb(safety_b), 4),
        total_estimated_gb=round(bytes_to_gb(total_b), 4),
    )


def _classify_feasibility(*, total_gb: float, gpu_vram_gb: float) -> tuple[str, float]:
    ratio = total_gb / max(1e-6, gpu_vram_gb)
    if ratio <= 0.85:
        return "yes", ratio
    if ratio <= 0.97:
        return "marginal", ratio
    return "no", ratio


def _build_assumptions(req: EstimateRequest, md: ModelMetadata) -> list[str]:
    out = [
        f"base model has {md.total_params:,} parameters (source: {md.source})",
        f"weights stored as {req.quantization if req.method == 'qlora' else req.base_dtype}",
        f"gradient_checkpointing={'on' if req.gradient_checkpointing else 'off'}",
        f"attention_implementation={req.attention_implementation}",
        f"activation_dtype={req.activation_dtype}",
        f"optimizer={req.optimizer}",
        f"lora_rank={req.lora_rank}, target_scope={req.lora_target_scope}",
    ]
    if req.method == "full":
        out.append(
            "full fine-tune: gradients and optimizer states cover *all* params; "
            "this almost never fits on consumer GPUs for >1B params."
        )
    return out


def _build_warnings(
    req: EstimateRequest, md: ModelMetadata, breakdown: MemoryBreakdown
) -> list[str]:
    warnings: list[str] = []
    if req.method == "lora" and req.quantization.lower() not in {"none", "fp16", "bf16", "fp32"}:
        warnings.append(
            "lora method ignores --quantization other than fp16/bf16/fp32. Use 'qlora' for 4-bit."
        )
    if req.seq_len >= 8192:
        warnings.append(
            "seq_len>=8192: activations dominate the estimate; static numbers are less reliable. "
            "Run `canifinetune bench` to calibrate."
        )
    if req.micro_batch_size > 4 and req.method != "full":
        warnings.append(
            "micro_batch_size>4 is uncommon for LoRA/QLoRA on consumer GPUs; verify with bench."
        )
    if breakdown.activations_gb / max(breakdown.total_estimated_gb, 1e-6) > 0.6:
        warnings.append(
            "Activations dominate (>60% of total). Confidence on activation estimate is lower than weights."
        )
    return warnings


def _choose_confidence(
    req: EstimateRequest, breakdown: MemoryBreakdown, calibrated: bool
) -> str:
    if calibrated:
        return "high"
    if req.seq_len <= 2048 and req.micro_batch_size <= 2:
        return "medium"
    return "low"


def estimate(req: EstimateRequest) -> Estimate:
    """Static memory + feasibility estimate."""
    md = fetch_metadata(req.model_id, override=req.override)
    breakdown = _compute_breakdown(req, md)
    breakdown = apply_calibration(breakdown, req.calibration, request=req)
    feasibility, ratio = _classify_feasibility(
        total_gb=breakdown.total_estimated_gb, gpu_vram_gb=req.gpu_vram_gb
    )
    confidence = _choose_confidence(req, breakdown, calibrated=req.calibration is not None)
    assumptions = _build_assumptions(req, md)
    warnings = _build_warnings(req, md, breakdown)

    return Estimate(
        request=req,
        metadata={
            "model_id": md.model_id,
            "family": md.family,
            "total_params": md.total_params,
            "arch": asdict(md.arch),
            "source": md.source,
            "notes": md.notes,
        },
        memory=breakdown,
        feasible=feasibility,
        feasibility_ratio=round(ratio, 4),
        confidence=confidence,
        assumptions=assumptions,
        warnings=warnings,
        calibration_applied=req.calibration is not None,
    )
