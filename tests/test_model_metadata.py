from __future__ import annotations

import pytest

import canifinetune.estimator.model_metadata as metadata

MIXTRAL_CONFIG = {
    "model_type": "mixtral",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "tie_word_embeddings": False,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
}


def test_hub_safetensors_count_wins_over_dense_config_estimate(monkeypatch):
    monkeypatch.setattr(metadata, "fetch_model_config", lambda *args, **kwargs: MIXTRAL_CONFIG)
    monkeypatch.setattr(
        metadata,
        "fetch_model_parameter_count",
        lambda *args, **kwargs: 46_702_792_704,
    )

    resolved = metadata.fetch_metadata("test-org/test-mixtral")

    assert resolved.total_params == 46_702_792_704
    assert resolved.parameter_count_source == "hub-safetensors"
    assert resolved.arch.num_local_experts == 8
    assert resolved.arch.num_experts_per_tok == 2


def test_moe_fallback_formula_counts_all_expert_weights():
    resolved = metadata._from_hf_config("test/mixtral", MIXTRAL_CONFIG)

    assert resolved is not None
    assert resolved.total_params > 40_000_000_000
    assert resolved.parameter_count_source == "estimated"


def test_use_network_false_does_not_call_hub(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network helper should not be called")

    monkeypatch.setattr(metadata, "fetch_model_config", fail)
    monkeypatch.setattr(metadata, "fetch_model_parameter_count", fail)

    with pytest.raises(ValueError, match="Cannot resolve metadata"):
        metadata.fetch_metadata("test/unknown-model", use_network=False)
