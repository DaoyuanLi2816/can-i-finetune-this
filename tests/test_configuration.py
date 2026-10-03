import json

import pytest

from canifinetune.bench.runner import BenchConfig
from canifinetune.configuration import TrainingConfig
from canifinetune.estimator.memory import EstimateRequest, estimate
from canifinetune.recipes import RecipeRequest, generate_recipe
from canifinetune.training.runtime import RunConfig


def test_configuration_agrees_across_entrypoints(tmp_path):
    shared = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "method": "lora",
        "base_dtype": "fp32",
        "optimizer": "adamw_torch",
        "seq_len": 128,
        "micro_batch_size": 2,
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "gradient_checkpointing": False,
        "attention_implementation": "eager",
        "loss_mode": "assistant",
    }
    request = EstimateRequest(**shared, gpu_vram_gb=16)
    bench = BenchConfig(**shared)
    recipe = RecipeRequest(**shared, gradient_accumulation_steps=1, output_dir=tmp_path / "recipe")
    generate_recipe(recipe)
    loaded = RunConfig.from_yaml(recipe.output_dir / "config.yaml")
    for key, value in request.training_fields().items():
        if key == "target_modules":
            continue
        assert bench.training_fields()[key] == value
        assert loaded.training_fields()[key] == value
    assert loaded.target_modules == estimate(request).planned_configuration["target_modules"]
    assert loaded.quantization == "none"


def test_invalid_and_unknown_configuration_is_rejected():
    for fields in (
        {"base_dtype": "guess"},
        {"attention_implementation": "auto"},
        {"optimizer": "rmsprop"},
        {"seq_len": 1},
        {"surprise_option": True},
        {"method": "qlora", "quantization": "bf16"},
    ):
        with pytest.raises(ValueError):
            TrainingConfig(model_id="test/model", **fields)


def test_config_paths_and_unknown_yaml_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        json.dumps(
            {"model_id": "test/model", "dataset_path": "data.jsonl", "output_dir": "trained"}
        )
    )
    loaded = RunConfig.from_yaml(path)
    assert loaded.dataset_path == str(tmp_path / "data.jsonl")
    assert loaded.output_dir == str(tmp_path / "trained")
    path.write_text('{"model_id": "test/model", "typo": 1}')
    with pytest.raises(ValueError, match="typo"):
        RunConfig.from_yaml(path)


def test_free_budget_does_not_change_overhead_or_repeat_safety():
    full = estimate(EstimateRequest(model_id="Qwen/Qwen2.5-1.5B-Instruct", gpu_vram_gb=16))
    free = estimate(
        EstimateRequest(model_id="Qwen/Qwen2.5-1.5B-Instruct", gpu_vram_gb=16, available_vram_gb=6)
    )
    assert free.memory == full.memory
    assert free.feasible == "no" and full.feasible == "yes"


def test_liger_has_explicit_unvalidated_memory_proxy():
    result = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-0.5B-Instruct", gpu_vram_gb=16, use_liger_kernel=True
        )
    )
    assert result.confidence == "low"
    assert result.planned_configuration["loss_backend"] == "liger"
    assert any("proxy" in warning for warning in result.warnings)


def test_secondary_cuda_device_is_rejected_before_loading(monkeypatch):
    import sys
    from types import SimpleNamespace

    from canifinetune.configuration import TrainingConfig
    from canifinetune.training.runtime import validate_device

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(ValueError, match="secondary CUDA device"):
        validate_device(TrainingConfig(model_id="org/model"), "cuda:1")
