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
