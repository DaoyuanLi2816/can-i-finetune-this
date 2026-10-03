from __future__ import annotations

import json
from pathlib import Path

from canifinetune.estimator.calibration import (
    CalibrationSample,
    calibration_from_result_files,
    fit_calibration_from_samples,
    load_calibration,
    save_calibration,
)
from canifinetune.estimator.memory import EstimateRequest, estimate


def test_fit_from_samples_mean_ratio():
    samples = [
        CalibrationSample(
            gpu_name="RTX 4080",
            torch_version="2.4.1",
            cuda_version="12.1",
            model_family="qwen2",
            method="qlora",
            seq_len=1024,
            micro_batch_size=1,
            estimated_total_gb=4.0,
            measured_total_gb=4.4,
        ),
        CalibrationSample(
            gpu_name="RTX 4080",
            torch_version="2.4.1",
            cuda_version="12.1",
            model_family="qwen2",
            method="qlora",
            seq_len=1024,
            micro_batch_size=1,
            estimated_total_gb=2.0,
            measured_total_gb=2.6,
        ),
    ]
    calib = fit_calibration_from_samples(samples)
    # Expected mean ratio = (4.4/4 + 2.6/2)/2 = (1.1 + 1.3)/2 = 1.2
    assert abs(calib.activation_scale - 1.2) < 0.01
    assert calib.has_data()


def test_load_save_round_trip(tmp_path: Path):
    s = CalibrationSample(
        gpu_name="x",
        torch_version="2",
        cuda_version="12",
        model_family="qwen2",
        method="qlora",
        seq_len=128,
        micro_batch_size=1,
        estimated_total_gb=1.0,
        measured_total_gb=1.1,
    )
    calib = fit_calibration_from_samples([s])
    p = tmp_path / "calib.json"
    save_calibration(calib, p)
    re = load_calibration(p)
    assert re.has_data()
    assert abs(re.activation_scale - calib.activation_scale) < 1e-6


def test_calibration_from_result_files(tmp_path: Path):
    fake = {
        "config": {"seq_len": 1024, "micro_batch_size": 1},
        "gpu": {"name": "RTX 4080"},
        "env": {"torch_version": "2.4.1", "cuda_version": "12.1"},
        "model_family": "qwen2",
        "method": "qlora",
        "estimated_total_gb": 4.0,
        "measured": {"peak_total_gb": 4.6},
    }
    fp = tmp_path / "x.json"
    fp.write_text(json.dumps(fake), encoding="utf-8")
    calib = calibration_from_result_files([fp])
    assert calib.has_data()
    assert calib.samples[0].measured_total_gb == 4.6
    assert calib.estimator_version == "legacy"
    assert not calib.is_compatible(model_family="qwen2", method="qlora", gpu_vram_gb=16)


def test_calibration_adjusts_estimate():
    base = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=2048,
            micro_batch_size=1,
        )
    )

    sample = CalibrationSample(
        gpu_name="RTX 4080",
        torch_version="2.4.1",
        cuda_version="12.1",
        model_family="qwen2",
        method="qlora",
        seq_len=2048,
        micro_batch_size=1,
        estimated_total_gb=base.memory.total_estimated_gb,
        measured_total_gb=base.memory.total_estimated_gb * 1.25,
    )
    calib = fit_calibration_from_samples([sample])

    adjusted = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=2048,
            micro_batch_size=1,
            calibration=calib,
        )
    )
    # With a 1.25x correction the calibrated estimate should be larger than baseline.
    assert adjusted.memory.total_estimated_gb > base.memory.total_estimated_gb
    assert adjusted.calibration_applied
    assert adjusted.memory.static_model_gb == base.memory.static_model_gb


def test_calibration_is_not_applied_to_an_unseen_family():
    sample = CalibrationSample(
        gpu_name="RTX 4080",
        torch_version="2.6",
        cuda_version="12.4",
        model_family="qwen2",
        method="qlora",
        seq_len=1024,
        micro_batch_size=1,
        estimated_total_gb=4.0,
        measured_total_gb=8.0,
    )
    calib = fit_calibration_from_samples([sample])

    adjusted = estimate(
        EstimateRequest(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            method="qlora",
            gpu_vram_gb=16.0,
            seq_len=1024,
            calibration=calib,
        )
    )

    assert not adjusted.calibration_applied


def test_legacy_calibration_file_is_ignored(tmp_path: Path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "samples": [],
                "activation_scale": 0.5,
                "weights_scale": 0.5,
                "overhead_scale": 0.5,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_calibration(path)

    assert not loaded.has_data()
    assert "Legacy calibration ignored" in loaded.note


def test_current_measurement_fit_requires_complete_matching_workload():
    from canifinetune.configuration import TrainingConfig
    from canifinetune.estimator.calibration import _result_to_sample, calibration_signature

    config = TrainingConfig(
        model_id="org/model", method="lora", optimizer="adamw_torch"
    ).training_fields()
    data = {
        "schema_version": 2,
        "estimator_version": "0.4.0",
        "success": True,
        "status": "success",
        "config": {**config, "steps": 3},
        "effective_configuration": config,
        "completed_steps": 3,
        "method": "lora",
        "model_family": "qwen2",
        "gpu": {"total_vram_gb": 16},
        "measured": {"peak_allocated_gb": 3, "peak_reserved_gb": 3.4},
        "estimated_total_gb": 3.6,
        "estimated_breakdown": {
            "static_model_gb": 2,
            "gradients_gb": 0.1,
            "optimizer_gb": 0.1,
            "activations_gb": 0.5,
            "logits_gb": 0.3,
            "cuda_overhead_gb": 0.4,
            "safety_margin_gb": 0.2,
        },
    }
    sample = _result_to_sample(data)
    assert sample is not None
    calibration = fit_calibration_from_samples([sample])
    assert calibration.is_compatible(
        model_family="qwen2",
        method="lora",
        gpu_vram_gb=16,
        config_signature=calibration_signature(config),
    )
    assert not calibration.is_compatible(
        model_family="qwen2",
        method="lora",
        gpu_vram_gb=16,
        config_signature=calibration_signature({**config, "seq_len": 512}),
    )
    assert _result_to_sample({**data, "completed_steps": 2}) is None
    assert _result_to_sample({**data, "estimator_version": "0.3.0"}) is None
    assert _result_to_sample({**data, "measured": {"peak_reserved_gb": 3.4}}) is None
