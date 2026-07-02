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

from ..estimator.memory import EstimateRequest, estimate
from ..estimator.model_metadata import fetch_metadata
from ..utils.gpu import probe_cuda
from ..utils.logging import get_logger, utc_now_iso
from .memory_trace import MemorySnapshot, empty_cache, reset_peak, snapshot
from .oom import OomReport, is_oom, make_oom_report
from .synthetic_data import make_batch

log = get_logger("bench.runner")


class BenchConfig(BaseModel):
    model_id: str
    method: Literal["full", "lora", "qlora"] = "lora"
    seq_len: int = Field(128, gt=0)
    micro_batch_size: int = Field(1, gt=0)
    steps: int = Field(2, gt=0)
    lora_rank: int = Field(8, gt=0)
    lora_alpha: int = Field(16, gt=0)
    lora_dropout: float = 0.0
    lora_target_scope: Literal["attention", "all_linear", "conservative"] = "attention"
    optimizer: str = "paged_adamw_8bit"
    quantization: str = "nf4_double_quant"
    base_dtype: str = "bf16"
    gradient_checkpointing: bool = True
    attention_implementation: str = "sdpa"
    device: str = "cuda"
    # If set, the runner will only do `forward()` (no backward) — used for
    # extra-tiny smoke tests.
    forward_only: bool = False
    # If True, the runner records the static-estimate alongside the measurement.
    record_estimate: bool = True


class BenchResult(BaseModel):
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
    estimated_total_gb: float = 0.0
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


def _make_lora_config(cfg: BenchConfig, family: str) -> Any:
    from peft import LoraConfig  # type: ignore

    from ..estimator.formulas import default_target_modules

    target_modules = default_target_modules(family, scope=cfg.lora_target_scope)
    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )


def _build_optimizer(params, name: str):
    import torch  # type: ignore

    name = name.lower()
    if name in {"paged_adamw_8bit", "adamw_8bit"}:
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.PagedAdamW8bit(params, lr=2e-4)
        except Exception as e:
            log.warning("bitsandbytes 8-bit optimizer unavailable (%s); falling back to AdamW", e)
            return torch.optim.AdamW(params, lr=2e-4)
    if name in {"adamw_torch", "adamw_torch_fused", "adamw"}:
        return torch.optim.AdamW(params, lr=2e-4)
    if name == "sgd":
        return torch.optim.SGD(params, lr=1e-3)
    return torch.optim.AdamW(params, lr=2e-4)


def _model_dtype_kwarg(value: Any) -> dict[str, Any]:
    """Return ``{"dtype": value}`` for transformers ≥ 5.x, ``{"torch_dtype": value}`` for older.

    transformers 5.0 renamed ``torch_dtype`` → ``dtype``. We prefer the new
    name; if the installed transformers complains, the caller falls back.
    """
    return {"dtype": value}


def _build_model(cfg: BenchConfig):
    import torch  # type: ignore
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

    kwargs: dict[str, Any] = {"trust_remote_code": False}

    # bf16 vs fp16 dtype
    if cfg.base_dtype.lower() == "bf16":
        dtype_value = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif cfg.base_dtype.lower() == "fp16":
        dtype_value = torch.float16
    elif cfg.base_dtype.lower() == "fp32":
        dtype_value = torch.float32
    else:
        dtype_value = torch.float32
    kwargs.update(_model_dtype_kwarg(dtype_value))

    if cfg.method == "qlora":
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            compute_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=cfg.quantization == "nf4_double_quant",
            )
            kwargs.update(_model_dtype_kwarg(compute_dtype))
        except Exception as e:
            log.warning("BitsAndBytesConfig unavailable (%s); falling back to LoRA without 4-bit.", e)

    if cfg.attention_implementation in {"sdpa", "flash_attention_2", "eager"}:
        kwargs["attn_implementation"] = cfg.attention_implementation

    # For QLoRA, bitsandbytes places shards on GPU automatically when
    # device_map is set. For LoRA / full, we load on CPU and `.to(device)` after.
    if cfg.method == "qlora" and cfg.device.startswith("cuda"):
        kwargs["device_map"] = {"": 0}

    def _load(**extra: Any):
        merged = {**kwargs, **extra}
        return AutoModelForCausalLM.from_pretrained(cfg.model_id, **merged)

    try:
        model = _load()
    except TypeError as e:
        kwargs.pop("attn_implementation", None)
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        try:
            model = _load()
        except TypeError:
            log.warning("Retrying without dtype/torch_dtype after %s", e)
            kwargs.pop("torch_dtype", None)
            kwargs.pop("dtype", None)
            model = _load()

    if cfg.method != "qlora" and cfg.device.startswith("cuda"):
        model = model.to(cfg.device)

    # For QLoRA we need to prepare the model for 4-bit fine-tuning.
    if cfg.method == "qlora":
        try:
            from peft import prepare_model_for_kbit_training  # type: ignore

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.gradient_checkpointing,
            )
        except Exception as e:
            log.warning("prepare_model_for_kbit_training failed: %s", e)

    if cfg.gradient_checkpointing and not getattr(model, "is_gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            log.warning("gradient_checkpointing_enable failed: %s", e)

    config = AutoConfig.from_pretrained(cfg.model_id)
    return model, config


def _attach_lora(model, cfg: BenchConfig, family: str):
    if cfg.method == "full":
        return model
    from peft import get_peft_model  # type: ignore

    lora_cfg = _make_lora_config(cfg, family)
    return get_peft_model(model, lora_cfg)


def _torch_env() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch  # type: ignore

        out["torch_version"] = torch.__version__
        out["cuda_version"] = getattr(torch.version, "cuda", "") or ""
        out["cudnn_version"] = getattr(torch.backends.cudnn, "version", lambda: "")()
        out["bf16_supported"] = bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False
    except Exception:
        out["torch_version"] = "not installed"
    return out


def _gpu_snapshot_dict() -> dict[str, Any]:
    info = probe_cuda()
    if info.gpus:
        return info.gpus[0].to_dict()
    return {"available": False, "name": "unknown"}


def _build_estimate_total(cfg: BenchConfig, gpu_total_gb: float) -> float:
    if gpu_total_gb <= 0.0:
        return 0.0
    req = EstimateRequest(
        model_id=cfg.model_id,
        method=cfg.method,
        gpu_vram_gb=gpu_total_gb,
        seq_len=cfg.seq_len,
        micro_batch_size=cfg.micro_batch_size,
        base_dtype=cfg.base_dtype,
        quantization=cfg.quantization if cfg.method == "qlora" else "bf16",
        lora_rank=cfg.lora_rank,
        lora_target_scope=cfg.lora_target_scope,
        optimizer=cfg.optimizer,
        gradient_checkpointing=cfg.gradient_checkpointing,
        attention_implementation=cfg.attention_implementation,
    )
    return estimate(req).memory.total_estimated_gb


def run_bench(cfg: BenchConfig) -> BenchResult:
    """Run the benchmark and return a fully populated :class:`BenchResult`."""
    snapshots: list[MemorySnapshot] = []
    notes: list[str] = []

    gpu = _gpu_snapshot_dict()
    env = _torch_env()

    # Pre-compute the static estimate for the same config so the result file
    # is directly usable by ``canifinetune calibrate``.
    estimated_total = 0.0
    try:
        md = fetch_metadata(cfg.model_id)
        family = md.family
        if cfg.record_estimate:
            estimated_total = _build_estimate_total(
                cfg, gpu_total_gb=float(gpu.get("total_vram_gb") or 0.0)
            )
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
        notes=notes,
        method=cfg.method,
    )

    try:
        import torch  # type: ignore
    except Exception as e:
        result.success = False
        result.notes.append(f"torch not importable: {e}")
        return result

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        result.success = False
        result.notes.append("CUDA not available; bench requires a GPU.")
        return result

    _safe_clear()
    reset_peak()
    snapshots.append(snapshot("before_load"))

    try:
        model, hf_cfg = _build_model(cfg)
    except BaseException as e:
        if is_oom(e):
            result.oom = make_oom_report("model_load", e).to_dict()
        result.success = False
        result.notes.append(f"model load failed: {type(e).__name__}: {e}")
        result.notes.append(traceback.format_exc(limit=2))
        result.snapshots = [s.to_dict() for s in snapshots]
        return result

    family = getattr(hf_cfg, "model_type", family) or family
    result.model_family = family
    snapshots.append(snapshot("after_load"))

    try:
        model = _attach_lora(model, cfg, family)
    except BaseException as e:
        result.success = False
        result.notes.append(f"LoRA attach failed: {type(e).__name__}: {e}")
        result.snapshots = [s.to_dict() for s in snapshots]
        return result
    snapshots.append(snapshot("after_lora_attach"))

    try:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = _build_optimizer(trainable, cfg.optimizer)
    except BaseException as e:
        result.success = False
        result.notes.append(f"optimizer init failed: {type(e).__name__}: {e}")
        result.snapshots = [s.to_dict() for s in snapshots]
        return result
    snapshots.append(snapshot("after_optimizer_init"))

    model.train()
    vocab_size = int(getattr(hf_cfg, "vocab_size", 32000))

    step_times: list[float] = []
    total_tokens = 0
    last_loss: float | None = None

    for step in range(cfg.steps):
        try:
            batch = make_batch(
                batch_size=cfg.micro_batch_size,
                seq_len=cfg.seq_len,
                vocab_size=vocab_size,
                device=cfg.device,
                seed=step,
            )
        except BaseException as e:
            if is_oom(e):
                result.oom = make_oom_report("make_batch", e).to_dict()
            result.notes.append(f"batch generation failed at step {step}: {e}")
            break

        t0 = time.perf_counter()
        try:
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            )
            loss = outputs.loss
            last_loss = float(loss.detach().to("cpu").item())
            if step == 0:
                snapshots.append(snapshot("after_first_forward"))
            if not cfg.forward_only:
                loss.backward()
                if step == 0:
                    snapshots.append(snapshot("after_first_backward"))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step == 0:
                    snapshots.append(snapshot("after_first_optimizer_step"))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = time.perf_counter() - t0
            step_times.append(dt)
            total_tokens += cfg.micro_batch_size * cfg.seq_len
        except BaseException as e:
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
        "peak_total_gb": round(peak_reserved, 4),  # reserved ~= what the allocator holds
        "final_allocated_gb": round(final.allocated_gb, 4),
        "final_reserved_gb": round(final.reserved_gb, 4),
        "loss_last_step": last_loss,
    }

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
    safe_model = cfg.model_id.replace("/", "__")
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
    return base / f"{name}.json"
