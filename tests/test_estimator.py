"""Tests for the formula layer and the high-level estimator."""

from __future__ import annotations

import math

import pytest

from canifinetune.estimator.formulas import (
    activations_bytes,
    gradients_bytes,
    lora_trainable_params,
    optimizer_bytes,
    weights_bytes,
)
from canifinetune.estimator.memory import EstimateRequest, estimate


def test_weights_bytes_fp16_is_two_bytes_per_param():
    b = weights_bytes(num_params=1_000_000, base_dtype="fp16", quantization=None)
    assert b == pytest.approx(2_000_000)


def test_weights_bytes_bf16_matches_fp16():
    a = weights_bytes(num_params=10_000, base_dtype="bf16", quantization=None)
    b = weights_bytes(num_params=10_000, base_dtype="fp16", quantization=None)
    assert a == b


def test_weights_bytes_nf4_double_quant_smaller_than_fp16():
    raw = weights_bytes(num_params=1_000_000, base_dtype="fp16", quantization=None)
    q = weights_bytes(num_params=1_000_000, base_dtype="bf16", quantization="nf4_double_quant")
    assert q < raw / 3  # ~0.51 B/param vs 2 B/param


def test_weights_bytes_int8_is_close_to_one_byte_plus_overhead():
    b = weights_bytes(num_params=1_000_000, base_dtype="bf16", quantization="int8")
    assert 1.1 * 1e6 <= b <= 1.3 * 1e6


def test_lora_trainable_params_grows_with_rank(arch_qwen_1p5b):
    a = lora_trainable_params(arch=arch_qwen_1p5b, family="qwen2", rank=8)
    b = lora_trainable_params(arch=arch_qwen_1p5b, family="qwen2", rank=16)
    assert b == pytest.approx(2 * a, rel=0.01)


def test_lora_trainable_params_all_linear_is_bigger_than_attention(arch_qwen_1p5b):
    attn = lora_trainable_params(arch=arch_qwen_1p5b, family="qwen2", rank=16, scope="attention")
    full = lora_trainable_params(arch=arch_qwen_1p5b, family="qwen2", rank=16, scope="all_linear")
    assert full > attn


def test_gradients_only_for_trainable():
    g = gradients_bytes(trainable_params=1_000_000, grad_dtype="bf16")
    assert g == pytest.approx(2_000_000)


def test_optimizer_paged_adamw_8bit_is_smaller_than_torch_adamw():
    a = optimizer_bytes(trainable_params=1_000_000, optimizer="adamw_torch")
    b = optimizer_bytes(trainable_params=1_000_000, optimizer="paged_adamw_8bit")
    assert b < a


def test_activations_grow_with_seq_len(arch_qwen_1p5b):
    a = activations_bytes(seq_len=512, batch_size=1, arch=arch_qwen_1p5b)
    b = activations_bytes(seq_len=2048, batch_size=1, arch=arch_qwen_1p5b)
    assert b > a


def test_gradient_checkpointing_reduces_activations(arch_qwen_1p5b):
    base = activations_bytes(
        seq_len=2048, batch_size=1, arch=arch_qwen_1p5b, use_gradient_checkpointing=False
    )
    ckpt = activations_bytes(
        seq_len=2048, batch_size=1, arch=arch_qwen_1p5b, use_gradient_checkpointing=True
    )
    assert ckpt < base / 3  # checkpointing typically cuts at least 3x


def test_estimate_qwen_qlora_fits_on_16gb():
    est = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=2048,
            micro_batch_size=1,
            lora_rank=16,
        )
    )
    assert est.feasible == "yes"
    assert est.memory.total_estimated_gb < 12.0
    assert est.feasibility_ratio <= 0.85
    assert est.confidence in {"medium", "low", "high"}
    # Static model must dominate weight footprint with QLoRA.
    assert est.memory.static_model_gb < 2.0


def test_estimate_full_finetune_qwen_1p5b_does_not_fit_on_16gb():
    est = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            method="full",
            gpu_vram_gb=16.0,
            seq_len=2048,
            micro_batch_size=1,
            # Full fine-tune typically uses standard AdamW (fp32 m/v + master weights).
            optimizer="adamw_torch",
        )
    )
    # Full fine-tune of 1.5 B parameters with AdamW = ~18 GB optimizer + ~3 GB grads + 3 GB weights.
    assert est.feasible == "no"


def test_estimate_qlora_7b_marginal_or_yes_on_16gb_with_checkpointing():
    est = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=1024,
            micro_batch_size=1,
            lora_rank=16,
            gradient_checkpointing=True,
        )
    )
    assert est.feasible in {"yes", "marginal", "no"}
    # 6.5B linear params packed 4-bit (~3.1 GB) + untied embedding & lm_head
    # upcast to fp32 by kbit-prepare (~4.1 GB). Measured 7.21 GiB on a 4080.
    assert 6.5 < est.memory.static_model_gb < 8.0


def test_estimate_warnings_include_high_seq_len():
    est = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=8192,
            micro_batch_size=1,
        )
    )
    assert any("activations" in w.lower() or "seq_len" in w.lower() for w in est.warnings)


def test_estimate_unknown_model_with_override():
    est = estimate(
        EstimateRequest(
            model_id="unknown-org/unknown-model-9000B",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=512,
            micro_batch_size=1,
            override={
                "family": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 1024,
                "total_params": 100_000,
            },
        )
    )
    assert est.feasible == "yes"
    assert est.metadata["model_id"] == "unknown-org/unknown-model-9000B"


def test_estimate_total_is_sum_of_components():
    est = estimate(
        EstimateRequest(
            model_id="test/tiny-llama",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=128,
            micro_batch_size=1,
        )
    )
    parts = est.memory
    s = (
        parts.static_model_gb
        + parts.quantization_overhead_gb
        + parts.gradients_gb
        + parts.optimizer_gb
        + parts.activations_gb
        + parts.logits_gb
        + parts.cuda_overhead_gb
        + parts.safety_margin_gb
    )
    assert math.isclose(s, parts.total_estimated_gb, abs_tol=0.05)
