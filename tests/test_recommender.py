from __future__ import annotations

from canifinetune.estimator.memory import EstimateRequest
from canifinetune.estimator.recommender import (
    RecommendedConfig,
    recommend_configs,
    suggest_degradations,
)


def test_recommend_returns_feasible_configs():
    recs = recommend_configs(
        model_id="Qwen/Qwen2.5-1.5B-Instruct", gpu_vram_gb=16.0, top_k=5
    )
    assert recs, "expected at least one feasible config"
    assert all(isinstance(r, RecommendedConfig) for r in recs)
    assert all(r.estimate.feasible in {"yes", "marginal"} for r in recs)


def test_recommend_prefers_longer_seq_first():
    recs = recommend_configs(
        model_id="Qwen/Qwen2.5-1.5B-Instruct", gpu_vram_gb=16.0, top_k=3
    )
    seq_lens = [r.estimate.request.seq_len for r in recs]
    # The first one should not be smaller than the last in the returned ordering.
    assert seq_lens[0] >= seq_lens[-1]


def test_suggest_degradation_brings_full_finetune_to_feasibility():
    req = EstimateRequest(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        method="full",
        gpu_vram_gb=16.0,
        seq_len=4096,
        micro_batch_size=4,
        lora_rank=64,
        gradient_checkpointing=False,
        lora_target_scope="all_linear",
    )
    steps = suggest_degradations(req)
    assert steps, "should propose at least one degradation step"
    # The terminal step is either feasible OR explicitly recommends a smaller model.
    terminal = steps[-1]
    assert terminal.estimate.feasible in {"yes", "marginal", "no"}
    # Each step's request should differ from the previous (no infinite loops).
    seen = set()
    for s in steps:
        key = s.request.model_dump_json()
        assert key not in seen or s is steps[-1]
        seen.add(key)
