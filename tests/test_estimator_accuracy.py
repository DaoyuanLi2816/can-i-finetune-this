"""Regression tests anchoring the estimator to real GPU measurements.

The measured numbers are ``torch.cuda.max_memory_allocated`` /
``max_memory_reserved`` peaks from ``canifinetune bench`` runs on an RTX 4080
(16 GB), torch 2.6.0+cu124, transformers 5.8.1, peft 0.19.1, bitsandbytes
0.49.2 (result JSONs under ``benchmarks/``). They pin the *shape* of the
formulas — the logits/loss chain, the fp32-upcast embeddings under QLoRA, the
checkpointed-activation slope — so a future "simplification" cannot silently
reintroduce the 2-4x under-estimates these tests were written against.

If a stack upgrade legitimately moves real memory use, re-run
``scripts/collect_baselines.py`` / ``canifinetune bench`` and update the
table together with the bands.
"""

from __future__ import annotations

import pytest

from canifinetune.estimator.memory import EstimateRequest, estimate

# (model_id, method, seq_len, micro_batch, ckpt, measured_max_allocated_gb)
MEASURED_RTX4080 = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "qlora", 512, 1, True, 2.794),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qlora", 1024, 1, True, 4.041),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qlora", 2048, 1, True, 6.530),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qlora", 4096, 1, True, 11.507),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qlora", 1024, 2, True, 6.531),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qlora", 1024, 1, False, 9.341),
    ("Qwen/Qwen2.5-0.5B-Instruct", "qlora", 2048, 1, True, 5.522),
    ("Qwen/Qwen2.5-0.5B-Instruct", "lora", 1024, 1, True, 3.032),
    ("Qwen/Qwen2.5-3B-Instruct", "qlora", 1024, 1, True, 5.174),
    ("Qwen/Qwen2.5-7B-Instruct", "qlora", 1024, 1, True, 10.002),
]


def _real_components_gb(est) -> float:
    """Estimated bytes that map onto ``max_memory_allocated`` (i.e. total
    minus the fragmentation headroom and the policy safety margin)."""
    m = est.memory
    return m.total_estimated_gb - m.cuda_overhead_gb - m.safety_margin_gb


@pytest.mark.parametrize(
    "model_id,method,seq_len,batch,ckpt,measured_gb",
    MEASURED_RTX4080,
    ids=[f"{m.split('/')[-1]}-{meth}-s{s}-b{b}-{'ckpt' if c else 'nockpt'}"
         for m, meth, s, b, c, _ in MEASURED_RTX4080],
)
def test_estimate_tracks_measured_peak(model_id, method, seq_len, batch, ckpt, measured_gb):
    est = estimate(
        EstimateRequest(
            model_id=model_id,
            method=method,
            gpu_vram_gb=16.0,
            seq_len=seq_len,
            micro_batch_size=batch,
            lora_rank=16,
            gradient_checkpointing=ckpt,
        )
    )
    ratio = _real_components_gb(est) / measured_gb
    assert 0.85 <= ratio <= 1.25, (
        f"estimate {_real_components_gb(est):.2f} GB vs measured {measured_gb:.2f} GB "
        f"(ratio {ratio:.2f}) for {model_id} {method} s{seq_len} b{batch} ckpt={ckpt}"
    )


def test_seq_len_slope_matches_measurement():
    """Doubling seq_len roughly doubles the dynamic memory (logits dominate);
    measured 4.04 -> 6.53 -> 11.51 GiB for 1024/2048/4096 on Qwen2.5-1.5B."""
    def real(seq):
        est = estimate(
            EstimateRequest(
                model_id="Qwen/Qwen2.5-1.5B-Instruct",
                method="qlora",
                gpu_vram_gb=16.0,
                seq_len=seq,
                micro_batch_size=1,
            )
        )
        return _real_components_gb(est)

    d1 = real(2048) - real(1024)  # measured ~2.5 GiB
    d2 = real(4096) - real(2048)  # measured ~5.0 GiB
    assert 2.0 <= d1 <= 3.2
    assert 2.0 * 0.85 <= d2 / d1 <= 2.0 * 1.15  # near-linear slope


def test_batch_and_seq_are_interchangeable():
    """b=2,s=1024 measured within 0.1 GiB of b=1,s=2048 — the estimator
    should agree instead of charging batch differently from sequence."""
    def real(seq, batch):
        est = estimate(
            EstimateRequest(
                model_id="Qwen/Qwen2.5-1.5B-Instruct",
                method="qlora",
                gpu_vram_gb=16.0,
                seq_len=seq,
                micro_batch_size=batch,
            )
        )
        return _real_components_gb(est)

    assert real(1024, 2) == pytest.approx(real(2048, 1), rel=0.05)


def test_qlora_static_includes_fp32_embeddings():
    """Qwen2.5-1.5B loads at 1.51 GiB after prepare_model_for_kbit_training
    (0.61 GiB packed 4-bit + 0.87 GiB fp32 tied embedding), not the 0.79 GiB
    an all-params-quantized model would predict."""
    est = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=1024,
            micro_batch_size=1,
        )
    )
    assert 1.35 <= est.memory.static_model_gb <= 1.75


def test_untied_lm_head_costs_a_second_embedding():
    """Qwen2.5-7B (untied) pays fp32 for embedding AND lm_head: measured
    7.21 GiB static vs 3.04 GiB packed linear weights."""
    est = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            method="qlora",
            gpu_vram_gb=24.0,
            seq_len=1024,
            micro_batch_size=1,
        )
    )
    assert 6.5 <= est.memory.static_model_gb <= 8.0


def test_logits_term_scales_with_vocab():
    base = {
        "method": "qlora", "gpu_vram_gb": 16.0, "seq_len": 1024, "micro_batch_size": 1,
        "override": {
            "family": "llama",
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 4096,
            "total_params": 500_000_000,
        },
    }
    small = estimate(EstimateRequest(
        model_id="x/small-vocab", **{**base, "override": {**base["override"], "vocab_size": 32_000}}
    ))
    large = estimate(EstimateRequest(
        model_id="x/large-vocab", **{**base, "override": {**base["override"], "vocab_size": 152_064}}
    ))
    assert large.memory.logits_gb == pytest.approx(
        small.memory.logits_gb * 152_064 / 32_000, rel=0.01
    )
    # At Qwen-like vocab the logits chain must be GiB-scale (measured ~2 GiB
    # of the 4.04 GiB peak at seq 1024).
    assert large.memory.logits_gb > 1.5


def test_gradient_checkpointing_does_not_shrink_logits():
    kwargs = {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "method": "qlora",
        "gpu_vram_gb": 16.0,
        "seq_len": 1024,
        "micro_batch_size": 1,
    }
    on = estimate(EstimateRequest(**kwargs, gradient_checkpointing=True))
    off = estimate(EstimateRequest(**kwargs, gradient_checkpointing=False))
    assert on.memory.logits_gb == off.memory.logits_gb
    assert on.memory.activations_gb < off.memory.activations_gb
