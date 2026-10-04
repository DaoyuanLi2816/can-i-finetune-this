"""Shared, torch-free meanings for estimate, recommendation, bench and recipe."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)

    model_id: str = Field(min_length=1)
    revision: str = "main"
    method: Literal["full", "lora", "qlora"] = "qlora"
    seq_len: int = Field(2048, ge=2)
    micro_batch_size: int = Field(1, gt=0)
    gradient_accumulation_steps: int = Field(1, gt=0)
    base_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    quantization: str = "nf4_double_quant"
    lora_rank: int = Field(16, gt=0)
    lora_alpha: int = Field(32, gt=0)
    lora_dropout: float = Field(0.05, ge=0, lt=1)
    lora_target_scope: Literal["attention", "all_linear", "conservative"] = "attention"
    target_modules: list[str] | None = None
    optimizer: Literal[
        "paged_adamw_8bit", "adamw_8bit", "adamw_torch", "adamw_torch_fused", "adamw", "sgd"
    ] = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    attention_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
    training_backend: Literal["transformers"] = "transformers"
    loss_backend: Literal["stock", "liger"] = "stock"
    use_liger_kernel: bool = False
    loss_mode: Literal["all", "assistant"] = "all"
    truncation: Literal["error", "right"] = "error"
    local_files_only: bool = False

    @model_validator(mode="after")
    def validate_meanings(self):
        # Assignment uses object.__setattr__ to avoid recursive validation.
        if (
            not self.model_id.strip()
            or self.model_id != self.model_id.strip()
            or any(ord(c) < 32 for c in self.model_id)
        ):
            raise ValueError("model_id must be non-empty and have no surrounding whitespace")
        if not self.revision.strip() or any(ord(c) < 32 for c in self.revision):
            raise ValueError("revision must be non-empty and contain no control characters")
        if self.method != "qlora":
            object.__setattr__(self, "quantization", "none")
        elif self.quantization not in {"nf4", "nf4_double_quant", "fp4", "int4", "int8"}:
            raise ValueError("QLoRA supports nf4, nf4_double_quant, fp4/int4 or int8")
        if self.use_liger_kernel or self.loss_backend == "liger":
            object.__setattr__(self, "loss_backend", "liger")
            object.__setattr__(self, "use_liger_kernel", True)
        if self.target_modules is not None and (
            not self.target_modules or any(not t.strip() for t in self.target_modules)
        ):
            raise ValueError("target_modules must contain non-empty module names")
        return self

    def training_fields(self) -> dict:
        return {name: getattr(self, name) for name in TrainingConfig.model_fields}


def resolve_targets(config: TrainingConfig, family: str) -> list[str]:
    from .estimator.formulas import default_target_modules

    if config.method == "full":
        return []
    return config.target_modules or default_target_modules(
        family, scope=config.lora_target_scope, strict=True
    )


def validate_base_checkpoint(quantization_config: object) -> None:
    if quantization_config is not None:
        raise ValueError(
            "pre-quantized checkpoints are not qualified; select an unquantized base checkpoint "
            "and request quantization explicitly so estimation and loading agree"
        )
