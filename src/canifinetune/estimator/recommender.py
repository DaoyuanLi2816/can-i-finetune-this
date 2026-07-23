"""Search and degradation logic on top of :func:`estimate`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any

from pydantic import BaseModel

from .memory import Estimate, EstimateRequest, estimate


@dataclass(frozen=True)
class SearchSpace:
    method: tuple[str, ...] = ("qlora", "lora")
    seq_len: tuple[int, ...] = (512, 1024, 2048, 4096)
    micro_batch_size: tuple[int, ...] = (1, 2)
    lora_rank: tuple[int, ...] = (8, 16, 32, 64)
    gradient_checkpointing: tuple[bool, ...] = (True, False)
    optimizer: tuple[str, ...] = ("paged_adamw_8bit", "adamw_torch")
    quantization: tuple[str, ...] = ("nf4_double_quant", "nf4", "bf16")
    lora_target_scope: tuple[str, ...] = ("attention", "all_linear")


class RecommendedConfig(BaseModel):
    feasible: str
    feasibility_ratio: float
    confidence: str
    estimate: Estimate

    @property
    def headline(self) -> str:
        r = self.estimate.request
        return (
            f"{r.method}, seq_len={r.seq_len}, micro_batch={r.micro_batch_size}, "
            f"rank={r.lora_rank}, ckpt={'on' if r.gradient_checkpointing else 'off'}, "
            f"opt={r.optimizer}, quant={r.quantization}"
        )


def _iter_search(base: EstimateRequest, space: SearchSpace) -> Iterable[EstimateRequest]:
    combos = product(
        space.method,
        space.seq_len,
        space.micro_batch_size,
        space.lora_rank,
        space.gradient_checkpointing,
        space.optimizer,
        space.quantization,
        space.lora_target_scope,
    )
    for m, s, b, r, ck, opt, q, scope in combos:
        # Skip nonsensical combos.
        if m == "lora" and q not in {"none", "bf16", "fp16", "fp32"}:
            continue
        if m == "qlora" and q in {"bf16", "fp16", "fp32", "none"}:
            continue
        if m == "qlora" and opt == "adamw_torch":
            # QLoRA almost always pairs with paged 8-bit optimizer.
            continue
        yield EstimateRequest.model_validate(
            {
                **base.model_dump(),
                "method": m,
                "seq_len": s,
                "micro_batch_size": b,
                "lora_rank": r,
                "gradient_checkpointing": ck,
                "optimizer": opt,
                "quantization": q,
                "lora_target_scope": scope,
            }
        )


def recommend_configs(
    *,
    model_id: str,
    gpu_vram_gb: float,
    space: SearchSpace | None = None,
    top_k: int = 5,
    use_network: bool = True,
    override: dict[str, Any] | None = None,
    base_overrides: dict[str, Any] | None = None,
) -> list[RecommendedConfig]:
    """Return up to ``top_k`` feasible configs, preferring large seq_len then small rank.

    Args:
        model_id: HF model id.
        gpu_vram_gb: GPU VRAM in gibibytes.
        space: Search space (defaults to a wide-but-bounded grid).
        top_k: Maximum number of configs to return.
        use_network: Whether to allow HF config fetch in the inner estimator.
        override: Manual arch spec for unknown models.
        base_overrides: Extra fields forced on every candidate (e.g.
            ``{"attention_implementation": "flash"}``).
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    space = space or SearchSpace()
    base = EstimateRequest(
        model_id=model_id,
        method="qlora",
        gpu_vram_gb=gpu_vram_gb,
        override=override,
        use_network=use_network,
    )
    if base_overrides:
        base = EstimateRequest.model_validate({**base.model_dump(), **base_overrides})

    feasible: list[RecommendedConfig] = []
    marginal: list[RecommendedConfig] = []

    for cand in _iter_search(base, space):
        est = estimate(cand)
        rec = RecommendedConfig(
            feasible=est.feasible,
            feasibility_ratio=est.feasibility_ratio,
            confidence=est.confidence,
            estimate=est,
        )
        if est.feasible == "yes":
            feasible.append(rec)
        elif est.feasible == "marginal":
            marginal.append(rec)

    # Rank: prefer larger seq_len, larger rank, larger batch (more useful for training).
    def quality(rec: RecommendedConfig) -> tuple[int, int, int, float]:
        r = rec.estimate.request
        return (r.seq_len, r.lora_rank, r.micro_batch_size, -rec.feasibility_ratio)

    feasible.sort(key=quality, reverse=True)
    marginal.sort(key=quality, reverse=True)
    return (feasible + marginal)[:top_k]


# ---------------------------------------------------------------------------
# Degradation suggestions when a specific config is infeasible
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DegradationStep:
    description: str
    request: EstimateRequest
    estimate: Estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "request": self.request.model_dump(),
            "estimated_total_gb": self.estimate.memory.total_estimated_gb,
            "feasible": self.estimate.feasible,
            "feasibility_ratio": self.estimate.feasibility_ratio,
        }


def _try_step(req: EstimateRequest, description: str, **changes: Any) -> DegradationStep:
    updated = EstimateRequest.model_validate({**req.model_dump(), **changes})
    est = estimate(updated)
    return DegradationStep(description=description, request=updated, estimate=est)


def suggest_degradations(req: EstimateRequest) -> list[DegradationStep]:
    """Return a sequence of progressively cheaper configs.

    The list stops as soon as a step is feasible, so the caller can show
    "easiest path to feasibility".
    """
    steps: list[DegradationStep] = []
    state = req.model_copy()

    def push(desc: str, **kw: Any) -> bool:
        nonlocal state
        step = _try_step(state, desc, **kw)
        steps.append(step)
        state = step.request
        return step.estimate.feasible == "yes"

    # A full fine-tune is dominated by full-model gradients + optimizer
    # states; no batch/seq tweak can save it. Suggest PEFT first.
    if state.method == "full" and push(
        "Switch from full fine-tune to QLoRA (NF4 + double quant + paged 8-bit AdamW)",
        method="qlora",
        quantization="nf4_double_quant",
        optimizer="paged_adamw_8bit",
    ):
        return steps

    if state.micro_batch_size > 1 and push("Drop micro_batch_size to 1", micro_batch_size=1):
        return steps

    if not state.gradient_checkpointing and push(
        "Turn on gradient_checkpointing", gradient_checkpointing=True
    ):
        return steps

    if state.seq_len > 1024 and push("Halve seq_len", seq_len=max(1024, state.seq_len // 2)):
        return steps

    if (
        state.method != "full"
        and state.lora_rank > 8
        and push("Halve LoRA rank", lora_rank=max(8, state.lora_rank // 2))
    ):
        return steps

    if (
        state.method != "full"
        and state.lora_target_scope == "all_linear"
        and push(
            "Restrict LoRA target_modules to attention only",
            lora_target_scope="attention",
        )
    ):
        return steps

    if state.method == "lora" and push(
        "Switch to QLoRA (NF4 + double quant + paged 8-bit AdamW)",
        method="qlora",
        quantization="nf4_double_quant",
        optimizer="paged_adamw_8bit",
    ):
        return steps

    if state.optimizer != "paged_adamw_8bit" and push(
        "Switch optimizer to paged_adamw_8bit", optimizer="paged_adamw_8bit"
    ):
        return steps

    if state.seq_len > 512 and push("Drop seq_len to 512", seq_len=512):
        return steps

    if state.method != "full" and state.lora_rank > 4 and push("Drop LoRA rank to 4", lora_rank=4):
        return steps

    # Last resort: caller has to pick a smaller model. We surface that as a
    # final step with a synthetic description but no further config change.
    steps.append(
        DegradationStep(
            description=(
                "Still infeasible. Try a smaller base model (e.g. "
                "Qwen/Qwen2.5-0.5B-Instruct, TinyLlama, or sshleifer/tiny-gpt2 "
                "for smoke tests). CPU offload is possible but will be 5-20x "
                "slower than full-GPU training."
            ),
            request=state,
            estimate=estimate(state),
        )
    )
    return steps
