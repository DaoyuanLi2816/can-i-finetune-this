"""The actual local benchmark loop.

``run_bench(config)`` loads a model (with optional 4-bit quantization), wraps
it with a PEFT LoRA adapter (when applicable), runs a handful of forward /
backward / optimizer steps on synthetic tokens, and returns a JSON-serializable
:class:`BenchResult` with peak VRAM at each stage.

Everything heavy is imported lazily so this module is safe to import on CPU-only
installs.
"""

from __future__ import annotations

import contextlib
import gc
import time
import traceback
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..configuration import TrainingConfig
from ..estimator.memory import EstimateRequest, estimate
from ..estimator.model_metadata import fetch_metadata
from ..training.runtime import (
    attach_adapter,
    effective_configuration,
    failure_kind,
    load_model,
    software_stack,
)
from ..utils.gpu import probe_cuda
from ..utils.logging import get_logger, utc_now_iso
from .memory_trace import MemorySnapshot, empty_cache, reset_peak, snapshot
from .oom import OomReport, is_oom, make_oom_report
from .synthetic_data import make_batch

log = get_logger("bench.runner")


class BenchConfig(TrainingConfig):
    method: Literal["full", "lora", "qlora"] = "lora"
    seq_len: int = Field(128, ge=2)
    steps: int = Field(2, gt=0)
    lora_rank: int = Field(8, gt=0)
    lora_alpha: int = Field(16, gt=0)
    lora_dropout: float = Field(0.0, ge=0, lt=1)
    device: str = "cuda"
    forward_only: bool = False
    record_estimate: bool = True


class BenchResult(BaseModel):
    schema_version: int = 2
    estimator_version: str = "0.4.0"
    units: str = "GiB"
    status: str = "not_run"
    completed_steps: int = 0
    workload: str = "synthetic full-length all-token causal labels; load + first optimizer allocation + bounded optimizer updates"
    provenance: dict[str, Any] = Field(
        default_factory=lambda: {
            "source": "local",
            "review_status": "unreviewed",
            "cohort": "unspecified",
        }
    )
    requested_configuration: dict[str, Any] = Field(default_factory=dict)
    effective_configuration: dict[str, Any] = Field(default_factory=dict)
    estimated_feasible: str = "unknown"
    estimated_process_reserved_gb: float | None = None
    prediction_budget_gb: float | None = None
    config: BenchConfig
    model_family: str
    timestamp: str
    env: dict[str, Any]
    gpu: dict[str, Any]
    snapshots: list[dict[str, Any]] = Field(default_factory=list)
    tokens_per_second: float = 0.0
    avg_step_time_s: float = 0.0
    oom: dict[str, Any] = Field(default_factory=lambda: OomReport().to_dict())
    measured: dict[str, Any] = Field(default_factory=dict)
    estimated_total_gb: float | None = None
    estimated_breakdown: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    success: bool = True
    method: str = "lora"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def _safe_clear() -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        empty_cache()


def _build_optimizer(params, name: str):
    import torch

    name = name.lower()
    if name in {"paged_adamw_8bit", "adamw_8bit"}:
        try:
            import bitsandbytes as bnb

            cls = bnb.optim.PagedAdamW8bit if name == "paged_adamw_8bit" else bnb.optim.AdamW8bit
            return cls(params, lr=2e-4)
        except ImportError as e:
            raise RuntimeError(f"{name} requires a working bitsandbytes installation") from e
    if name in {"adamw_torch", "adamw_torch_fused", "adamw"}:
        return torch.optim.AdamW(params, lr=2e-4, fused=name == "adamw_torch_fused")
    if name == "sgd":
        return torch.optim.SGD(params, lr=1e-3)
    raise ValueError(f"Unsupported optimizer {name!r}")


def _build_model(cfg: BenchConfig):
    model = load_model(cfg, cfg.device)
    return model, model.config


def _attach_lora(model, cfg: BenchConfig, family: str):
    return attach_adapter(model, cfg)


def _torch_env() -> dict[str, Any]:
    out: dict[str, Any] = software_stack()
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_version"] = getattr(torch.version, "cuda", "") or ""
        out["cudnn_version"] = getattr(torch.backends.cudnn, "version", lambda: "")()
        out["bf16_supported"] = (
            bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False
        )
    except Exception:
        out["torch_version"] = "not installed"
    return out


def _gpu_snapshot_dict() -> dict[str, Any]:
    info = probe_cuda()
    if info.gpus:
        return info.gpus[0].to_dict()
    return {"available": False, "name": "unknown"}


def _estimate_request(cfg, gpu_total_gb, gpu_free_gb=None):
    return EstimateRequest(
        **cfg.training_fields(),
        gpu_vram_gb=gpu_total_gb,
        available_vram_gb=gpu_free_gb,
        use_network=not cfg.local_files_only,
    )


def run_bench(cfg: BenchConfig) -> BenchResult:
    """Run the benchmark and return a fully populated :class:`BenchResult`."""
    snapshots: list[MemorySnapshot] = []
    notes: list[str] = []

    gpu = _gpu_snapshot_dict()
    env = _torch_env()

    # Pre-compute the static estimate for the same config so the result file
    # is directly usable by ``canifinetune calibrate``.
    estimated_total = None
    estimated_breakdown: dict[str, Any] = {}
    try:
        md = fetch_metadata(
            cfg.model_id, revision=cfg.revision, use_network=not cfg.local_files_only
        )
        family = md.family
        if cfg.record_estimate:
            gpu_total_gb = float(gpu.get("total_vram_gb") or 0.0)
            if gpu_total_gb > 0:
                est = estimate(
                    _estimate_request(cfg, gpu_total_gb, gpu.get("free_vram_gb") or None)
                )
                estimated_breakdown = est.memory.model_dump()
                estimated_total = est.memory.total_estimated_gb
    except Exception as e:
        notes.append(f"Could not resolve model metadata in advance: {e}")
        family = "unknown"

    result = BenchResult(
        config=cfg,
        model_family=family,
        timestamp=utc_now_iso(),
        env=env,
        gpu=gpu,
        snapshots=[],
        oom=OomReport().to_dict(),
        estimated_total_gb=estimated_total,
        estimated_breakdown=estimated_breakdown,
        notes=notes,
        method=cfg.method,
        requested_configuration=cfg.training_fields(),
        estimated_feasible=est.feasible
        if cfg.record_estimate and estimated_total is not None
        else "unknown",
        prediction_budget_gb=(gpu.get("free_vram_gb") or gpu.get("total_vram_gb"))
        if cfg.device.startswith("cuda")
        else None,
        estimated_process_reserved_gb=(estimated_total - estimated_breakdown["safety_margin_gb"])
        if estimated_total is not None
        else None,
    )

    try:
        import torch
    except Exception as e:
        result.success = False
        result.status = "dependency_error"
        result.notes.append(f"torch not importable: {e}")
        return result

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        result.success = False
        result.status = "configuration_error"
        result.notes.append("CUDA not available; bench requires a GPU.")
        return result

    if cfg.loss_mode != "all" or cfg.use_liger_kernel:
        result.success = False
        result.status = "configuration_error"
        result.notes.append(
            "Synthetic bench supports all-token stock loss only; assistant/Liger need dataset-backed qualification"
        )
        return result

    _safe_clear()
    reset_peak()
    snapshots.append(snapshot("before_load"))

    try:
        model, hf_cfg = _build_model(cfg)
    except Exception as e:
        if is_oom(e):
            result.oom = make_oom_report("model_load", e).to_dict()
        result.success = False
        result.notes.append(f"model load failed: {type(e).__name__}: {e}")
        result.notes.append(traceback.format_exc(limit=2))
        result.snapshots = [s.to_dict() for s in snapshots]
        result.status = failure_kind(e)
        _safe_clear()
        return result

    family = getattr(hf_cfg, "model_type", family) or family
    result.model_family = family
    snapshots.append(snapshot("after_load"))

    try:
        model = _attach_lora(model, cfg, family)
    except Exception as e:
        result.success = False
        result.notes.append(f"LoRA attach failed: {type(e).__name__}: {e}")
        result.snapshots = [s.to_dict() for s in snapshots]
        if is_oom(e):
            result.oom = make_oom_report("adapter_attach", e).to_dict()
        result.status = failure_kind(e)
        del model
        _safe_clear()
        return result
    snapshots.append(snapshot("after_lora_attach"))
    try:
        result.effective_configuration = effective_configuration(model, cfg, cfg.device)
        if cfg.use_liger_kernel:
            raise ValueError(
                "Liger benchmark is not qualified; use the experimental recipe path and keep its stock upper-planning proxy separate"
            )
    except Exception as e:
        result.success = False
        result.status = "configuration_error"
        result.notes.append(str(e))
        del model
        _safe_clear()
        return result

    optimizer = None
    if not cfg.forward_only:
        try:
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = _build_optimizer(trainable, cfg.optimizer)
        except Exception as e:
            result.success = False
            result.notes.append(f"optimizer init failed: {type(e).__name__}: {e}")
            result.snapshots = [s.to_dict() for s in snapshots]
            if is_oom(e):
                result.oom = make_oom_report("optimizer_init", e).to_dict()
            result.status = failure_kind(e)
            del model
            _safe_clear()
            return result
        snapshots.append(snapshot("after_optimizer_init"))

    model.train()
    vocab_size = int(getattr(hf_cfg, "vocab_size", 32000))

    step_times: list[float] = []
    total_tokens = 0
    last_loss: float | None = None
    scaler = None

    for step in range(cfg.steps * cfg.gradient_accumulation_steps):
        try:
            batch = make_batch(
                batch_size=cfg.micro_batch_size,
                seq_len=cfg.seq_len,
                vocab_size=vocab_size,
                device=cfg.device,
                seed=step,
            )
        except Exception as e:
            if is_oom(e):
                result.oom = make_oom_report("make_batch", e).to_dict()
            result.notes.append(f"batch generation failed at step {step}: {e}")
            result.success = False
            break

        t0 = time.perf_counter()
        try:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
                cfg.base_dtype
            ]
            with torch.autocast(
                "cuda",
                dtype=dtype,
                enabled=cfg.device.startswith("cuda") and cfg.base_dtype != "fp32",
            ):
                outputs = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss in benchmark")
            last_loss = float(loss.detach().to("cpu").item())
            if step == 0:
                snapshots.append(snapshot("after_first_forward"))
            if not cfg.forward_only:
                if scaler is None:
                    scaler = torch.amp.GradScaler(
                        "cuda", enabled=cfg.device.startswith("cuda") and cfg.base_dtype == "fp16"
                    )
                scaler.scale(loss / cfg.gradient_accumulation_steps).backward()
                if step == 0:
                    snapshots.append(snapshot("after_first_backward"))
                assert optimizer is not None
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if step + 1 == cfg.gradient_accumulation_steps:
                        snapshots.append(snapshot("after_first_optimizer_step"))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = time.perf_counter() - t0
            step_times.append(dt)
            total_tokens += cfg.micro_batch_size * cfg.seq_len
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                result.completed_steps += 1
        except Exception as e:
            stage = "forward" if step == 0 else f"step_{step}"
            if is_oom(e):
                result.oom = make_oom_report(stage, e).to_dict()
                result.notes.append(
                    f"OOM at stage {stage}. Try lower micro_batch_size, smaller seq_len, "
                    "or enable gradient_checkpointing/QLoRA."
                )
            else:
                result.notes.append(f"step {step} failed: {type(e).__name__}: {e}")
            result.success = False
            break

    snapshots.append(snapshot("after_run"))

    final = snapshots[-1]
    peak_alloc = max((s.max_allocated_gb for s in snapshots), default=0.0)
    peak_reserved = max((s.max_reserved_gb for s in snapshots), default=0.0)
    result.measured = {
        "peak_allocated_gb": round(peak_alloc, 4),
        "peak_reserved_gb": round(peak_reserved, 4),
        "peak_total_gb": round(
            peak_reserved, 4
        ),  # legacy alias for process reserved, never device usage
        "final_allocated_gb": round(final.allocated_gb, 4),
        "final_reserved_gb": round(final.reserved_gb, 4),
        "loss_last_step": last_loss,
    }

    if result.success:
        try:
            result.effective_configuration = effective_configuration(model, cfg, cfg.device)
            if optimizer is not None:
                result.effective_configuration["optimizer_class"] = (
                    f"{type(optimizer).__module__}.{type(optimizer).__name__}"
                )
                result.effective_configuration["optimizer_state_dtypes"] = sorted(
                    {
                        str(value.dtype).replace("torch.", "")
                        for state in optimizer.state.values()
                        for value in state.values()
                        if isinstance(value, torch.Tensor)
                    }
                )
        except Exception as exc:
            result.success = False
            result.notes.append(f"effective configuration after updates: {exc}")

    if not cfg.device.startswith("cuda") or not result.success:
        # CPU is not a zero-VRAM observation. An OOM partial trace is not an exact peak.
        result.measured = {"loss_last_step": last_loss}
    result.status = (
        "success" if result.success else ("oom" if result.oom.get("happened") else "runtime_error")
    )

    if step_times:
        result.avg_step_time_s = round(sum(step_times) / len(step_times), 4)
        result.tokens_per_second = round(total_tokens / max(1e-6, sum(step_times)), 2)

    result.snapshots = [s.to_dict() for s in snapshots]

    # Cleanup; the runner expects to be callable again in-process.
    with contextlib.suppress(UnboundLocalError):
        del model, optimizer
    _safe_clear()

    return result


def result_path_for(
    out_dir: Path | str,
    cfg: BenchConfig,
    *,
    suffix: str = "",
) -> Path:
    """Compute a deterministic file path for this benchmark result."""
    base = Path(out_dir)
    import hashlib

    safe_model = Path(cfg.model_id).name.replace("\\", "__").replace("/", "__")
    name = (
        f"{safe_model}_{cfg.method}_s{cfg.seq_len}_b{cfg.micro_batch_size}"
        f"_r{cfg.lora_rank}_steps{cfg.steps}"
    )
    # Distinguish runs that differ only in checkpointing / quant / scope so
    # A/B comparisons don't overwrite each other. Defaults stay short.
    if not cfg.gradient_checkpointing:
        name += "_nockpt"
    if cfg.method == "qlora" and cfg.quantization != "nf4_double_quant":
        name += f"_{cfg.quantization}"
    if cfg.lora_target_scope != "attention":
        name += f"_{cfg.lora_target_scope}"
    if cfg.forward_only:
        name += "_fwdonly"
    if suffix:
        name = f"{name}_{suffix}"
    identity = hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:10]
    return base / f"{name}_{identity}_{time.time_ns()}.json"
