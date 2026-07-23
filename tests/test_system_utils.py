from __future__ import annotations

import subprocess

from canifinetune import doctor
from canifinetune.utils import gpu
from canifinetune.utils.subprocess import try_run


def test_parse_nvidia_smi_multiple_gpus():
    parsed = gpu._parse_nvidia_smi(
        "NVIDIA RTX 4080, 16384, 12000, 595.97, 8.9\nNVIDIA RTX 3090, 24576, 20000, 595.97, 8.6\n"
    )

    assert [item.name for item in parsed] == ["NVIDIA RTX 4080", "NVIDIA RTX 3090"]
    assert parsed[0].total_vram_gb == 16
    assert parsed[1].index == 1


def test_probe_cuda_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gpu, "_probe_torch_cuda", lambda: gpu.CudaInfo())
    expected = [gpu.GpuInfo(name="fallback", available=True)]
    monkeypatch.setattr(gpu, "probe_gpus_via_nvidia_smi", lambda: expected)

    assert gpu.probe_cuda().gpus == expected


def test_try_run_missing_executable(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda exe: None)
    code, stdout, stderr = try_run(["missing-command"])
    assert code == 127
    assert stdout == ""
    assert "not found" in stderr


def test_try_run_timeout(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda exe: "fake")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("fake", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    code, _, stderr = try_run(["fake"], timeout=1)
    assert code == 124
    assert "timed out" in stderr


def test_doctor_reports_missing_cuda_and_bitsandbytes(monkeypatch):
    cuda = gpu.CudaInfo(torch_available=True, torch_cuda_available=False)
    monkeypatch.setattr(doctor, "probe_cuda", lambda: cuda)
    monkeypatch.setattr(doctor, "_serialize_cuda", lambda info: info.to_dict())
    monkeypatch.setattr(
        doctor,
        "_probe_library",
        lambda name: doctor.LibInfo(
            name=name,
            installed=name != "bitsandbytes",
            version="1.0" if name != "bitsandbytes" else "",
        ),
    )
    monkeypatch.setattr(
        doctor,
        "_try_tiny_model_load",
        lambda: {"attempted": True, "ok": True, "model": "tiny"},
    )
    monkeypatch.setattr(
        doctor,
        "host_info",
        lambda: {"platform": "test", "platform_release": "1"},
    )

    report = doctor.run_doctor()

    assert any("cannot see CUDA" in issue for issue in report.issues)
    assert any("bitsandbytes is not installed" in issue for issue in report.issues)
