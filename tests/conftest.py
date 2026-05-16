"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from canifinetune.estimator.formulas import ArchHints
from canifinetune.estimator.model_metadata import register_known_model


@pytest.fixture(autouse=True)
def _register_local_test_model():
    """Register a small synthetic model so tests work without HF network."""
    register_known_model(
        "test/tiny-llama",
        {
            "family": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 1024,
            "total_params": 1_000_000,
        },
    )
    register_known_model(
        "test/qwen-mini",
        {
            "family": "qwen2",
            "hidden_size": 128,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "vocab_size": 4096,
            "total_params": 5_000_000,
        },
    )
    yield


@pytest.fixture
def arch_small() -> ArchHints:
    return ArchHints(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        vocab_size=1024,
    )


@pytest.fixture
def arch_qwen_1p5b() -> ArchHints:
    return ArchHints(
        hidden_size=1536,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        intermediate_size=8960,
        vocab_size=151936,
    )
