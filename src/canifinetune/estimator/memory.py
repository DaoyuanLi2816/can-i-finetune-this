"""High-level ``estimate()`` API used by the CLI and the recommender."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..configuration import TrainingConfig, resolve_targets
from ..utils.units import bytes_to_gb
from . import formulas as F
from .calibration import Calibration, apply_calibration, calibration_signature
from .model_metadata import ModelMetadata, fetch_metadata

Method = Literal["full", "lora", "qlora"]
Confidence = Literal["low", "medium", "high"]


class EstimateRequest(TrainingConfig):
    """Inputs to :func:`estimate`. Validated with pydantic."""

    gpu_vram_gb: float = Field(..., gt=0.0, description="Total GPU VRAM in gibibytes.")
    available_vram_gb: float | None = Field(None, gt=0)
    activation_dtype: str | None = None

    # If you've cached calibration, the CLI passes it in; library users
    # can pass a :class:`Calibration` directly.
    calibration: Calibration | None = None

    # Optional override dict for unknown models (hidden_size, ...).
    override: dict[str, Any] | None = None
    use_network: bool = True
    revision: str = "main"

    @model_validator(mode="after")
    def _normalize_method(self) -> EstimateRequest:
        if self.activation_dtype is not None and self.activation_dtype != self.base_dtype:
            raise ValueError("activation_dtype must match base_dtype in the shared runtime")
        object.__setattr__(self, "activation_dtype", self.base_dtype)
        if self.available_vram_gb is not None and self.available_vram_gb > self.gpu_vram_gb:
            raise ValueError("available_vram_gb cannot exceed total gpu_vram_gb")
        return self


class MemoryBreakdown(BaseModel):
    static_model_gb: float
    quantization_overhead_gb: float
    trainable_params_mb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    logits_gb: float = 0.0
    cuda_overhead_gb: float
    safety_margin_gb: float
    total_estimated_gb: float


class Estimate(BaseModel):
    estimator_version: str = "0.4.0"
    units: str = "GiB (legacy *_gb fields)"
    confidence_basis: str = "Heuristic evidence grade, not a calibrated probability interval. Historical data informed the formulas."
    evidence_scope: str = "Historical RTX 4080 dense causal LM measurements; independent and cross-hardware accuracy is not guaranteed."
    requested_configuration: dict[str, Any] = Field(default_factory=dict)
    planned_configuration: dict[str, Any] = Field(default_factory=dict)
    measurement_target: str = (
        "process torch peak reserved, load plus bounded training; safety margin reported separately"
    )
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
    quantized = req.method == "qlora"

    # 1) Weights. For QLoRA only the Linear layers are 4-bit; embeddings,
    #    lm_head, and norms stay full precision and are upcast to fp32 by
    #    PEFT's prepare_model_for_kbit_training.
    weights_b = F.weights_bytes(
        num_params=total_params,
        base_dtype=req.base_dtype,
        quantization=req.quantization if quantized else None,
        arch=arch,
    )
    # Quantization side costs surfaced separately: absmax / quant-state
    # metadata plus the transient bf16 dequantization workspace.
    if quantized:
        from ..utils.units import quant_overhead

        linear_params = max(0, total_params - F.embedding_params(arch))
        quant_overhead_b = linear_params * quant_overhead(req.quantization) + (
            F.dequant_workspace_bytes(arch=arch, quantization=req.quantization)
        )
    else:
        quant_overhead_b = 0.0

    # 2) Trainable params.
    if req.method == "full":
        trainable_params = total_params
        adapter_b = 0.0  # already counted in weights
    else:
        defaults = F.default_target_modules(md.family, scope=req.lora_target_scope, strict=True)
        if req.target_modules is not None and req.target_modules != defaults:
            raise ValueError(
                "custom target_modules need an architecture-specific memory model; use a supported target scope"
            )
        trainable_params = F.lora_trainable_params(
            arch=arch,
            family=md.family,
            rank=req.lora_rank,
            scope=req.lora_target_scope,
        )
        adapter_b = F.adapter_weights_bytes(trainable_params=trainable_params)

    # 3) Gradients (fp32 for LoRA adapters, bf16 for full fine-tunes where
    #    full-training gradients follow the loaded parameter dtype).
    grad_b = F.gradients_bytes(
        trainable_params=trainable_params,
        grad_dtype=req.base_dtype if req.method == "full" else "fp32",
    )

    # 4) Optimizer states.
    opt_b = F.optimizer_bytes(
        trainable_params=trainable_params,
        optimizer=req.optimizer,
        parameter_dtype=req.base_dtype if req.method == "full" else "fp32",
    )

    # 5) Activations.
    act_b = F.activations_bytes(
        seq_len=req.seq_len,
        batch_size=req.micro_batch_size,
        arch=arch,
        activation_dtype=req.base_dtype,
        use_gradient_checkpointing=req.gradient_checkpointing,
        attention_implementation=req.attention_implementation,
        family=md.family,
        quantized_base=quantized,
    )

    # 6) Logits + cross-entropy chain. Unaffected by gradient checkpointing;
    #    dominates for large-vocab models at long seq_len.
    logits_b = F.logits_loss_bytes(
        seq_len=req.seq_len,
        batch_size=req.micro_batch_size,
        vocab_size=arch.vocab_size,
        logits_dtype=req.base_dtype,
    )
    # Liger is a different loss allocation path. Until separately calibrated,
    # retain the stock upper-planning proxy and explicitly mark low confidence.

    # 7) CUDA / fragmentation / safety.
    cuda_b = F.cuda_overhead_bytes(available_vram_gb=req.gpu_vram_gb)
    safety_b = F.safety_margin_bytes(available_vram_gb=req.gpu_vram_gb)

    total_b = (
        weights_b
        + quant_overhead_b
        + adapter_b
        + grad_b
        + opt_b
        + act_b
        + logits_b
        + cuda_b
        + safety_b
    )

    return MemoryBreakdown(
        static_model_gb=round(bytes_to_gb(weights_b + adapter_b), 4),
        quantization_overhead_gb=round(bytes_to_gb(quant_overhead_b), 4),
        trainable_params_mb=round(trainable_params / 1e6, 3),
        gradients_gb=round(bytes_to_gb(grad_b), 4),
        optimizer_gb=round(bytes_to_gb(opt_b), 4),
        activations_gb=round(bytes_to_gb(act_b), 4),
        logits_gb=round(bytes_to_gb(logits_b), 4),
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
        f"logits/loss chain models the stock HF Trainer path "
        f"(full fp32 logits for vocab_size={md.arch.vocab_size:,}); "
        "fused-CE kernels (e.g. Liger) would shrink the 'logits' component",
    ]
    if req.method != "full":
        # LoRA rank / target scope are meaningless for a full fine-tune (no
        # adapters are added; every parameter is trainable), so only surface
        # this assumption when it actually applies.
        out.append(f"lora_rank={req.lora_rank}, target_scope={req.lora_target_scope}")
    if req.method == "qlora":
        out.append(
            "qlora: embeddings/lm_head/norms are NOT quantized and are upcast "
            "to fp32 by peft's prepare_model_for_kbit_training"
        )
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
    if md.arch.num_local_experts > 1:
        warnings.append(
            f"Mixture-of-experts model: {md.arch.num_local_experts} experts are stored and "
            f"{md.arch.num_experts_per_tok} expert(s) are active per token. "
            f"Parameter count source: {md.parameter_count_source}."
        )
    if md.parameter_count_source == "estimated":
        warnings.append(
            "Exact safetensors parameter metadata was unavailable; the total parameter "
            "count was estimated from config.json."
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
    dynamic_gb = breakdown.activations_gb + breakdown.logits_gb
    if dynamic_gb / max(breakdown.total_estimated_gb, 1e-6) > 0.6:
        warnings.append(
            "Activations + logits dominate (>60% of total). Confidence on these terms is "
            "lower than on weights; a fused-CE kernel or shorter seq_len shrinks them."
        )
    return warnings


def _choose_confidence(
    req: EstimateRequest,
    md: ModelMetadata,
    breakdown: MemoryBreakdown,
    calibrated: bool,
) -> str:
    if req.loss_backend == "liger":
        return "low"
    if calibrated:
        return "medium"
    if md.arch.num_local_experts > 1:
        return "low"
    if req.seq_len <= 2048 and req.micro_batch_size <= 2:
        return "medium"
    return "low"


def estimate(req: EstimateRequest) -> Estimate:
    """Static memory + feasibility estimate."""
    md = fetch_metadata(
        req.model_id,
        override=req.override,
        use_network=req.use_network,
        revision=req.revision,
    )
    breakdown = _compute_breakdown(req, md)
    calibration_applied = bool(
        req.loss_backend == "stock"
        and req.calibration
        and req.calibration.is_compatible(
            model_family=md.family,
            method=req.method,
            gpu_vram_gb=req.gpu_vram_gb,
            config_signature=calibration_signature(req.training_fields()),
        )
    )
    breakdown = apply_calibration(
        breakdown,
        req.calibration if req.loss_backend == "stock" else None,
        model_family=md.family,
        method=req.method,
        gpu_vram_gb=req.gpu_vram_gb,
        config_signature=calibration_signature(req.training_fields()),
    )
    feasibility, ratio = _classify_feasibility(
        total_gb=breakdown.total_estimated_gb,
        gpu_vram_gb=req.available_vram_gb or req.gpu_vram_gb,
    )
    confidence = _choose_confidence(
        req,
        md,
        breakdown,
        calibrated=calibration_applied,
    )
    assumptions = _build_assumptions(req, md)
    warnings = _build_warnings(req, md, breakdown)
    if req.loss_backend == "liger":
        warnings.append(
            "Liger fused loss: stock logits proxy retained as a conservative planning assumption; no fused-loss accuracy claim or stock calibration applies."
        )
    if calibration_applied:
        warnings.append(
            "Calibration is a fit to supplied runs, not independent accuracy evidence or a statistical confidence interval."
        )

    return Estimate(
        request=req,
        requested_configuration=req.training_fields(),
        planned_configuration={
            **req.training_fields(),
            "target_modules": resolve_targets(req, md.family),
        },
        metadata={
            "model_id": md.model_id,
            "family": md.family,
            "total_params": md.total_params,
            "arch": asdict(md.arch),
            "source": md.source,
            "parameter_count_source": md.parameter_count_source,
            "notes": md.notes,
        },
        memory=breakdown,
        feasible=feasibility,
        feasibility_ratio=round(ratio, 4),
        confidence=confidence,
        assumptions=assumptions,
        warnings=warnings,
        calibration_applied=calibration_applied,
    )
