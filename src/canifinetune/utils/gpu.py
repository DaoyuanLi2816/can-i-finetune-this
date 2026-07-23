"""Probe CUDA / NVIDIA GPU info without requiring torch to be installed."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any

from .subprocess import try_run

NVIDIA_SMI_QUERY = "name,memory.total,memory.free,driver_version,compute_cap"


@dataclass
class GpuInfo:
    index: int = 0
    name: str = "Unknown"
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    driver_version: str = ""
    compute_capability: str = ""
    available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_vram_gb": round(self.total_vram_gb, 3),
            "free_vram_gb": round(self.free_vram_gb, 3),
            "driver_version": self.driver_version,
            "compute_capability": self.compute_capability,
            "available": self.available,
        }


@dataclass
class CudaInfo:
    torch_available: bool = False
    torch_version: str = ""
    torch_cuda_available: bool = False
    torch_cuda_version: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "torch_cuda_available": self.torch_cuda_available,
            "torch_cuda_version": self.torch_cuda_version,
            "gpus": [g.to_dict() for g in self.gpus],
        }


def _parse_nvidia_smi(stdout: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for idx, raw_line in enumerate(stdout.strip().splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        name, mem_total, mem_free, driver, compute_cap = parts[:5]
        try:
            total_mb = float(mem_total.split()[0])
            free_mb = float(mem_free.split()[0])
        except (ValueError, IndexError):
            total_mb = 0.0
            free_mb = 0.0
        gpus.append(
            GpuInfo(
                index=idx,
                name=name,
                total_vram_gb=total_mb / 1024.0,
                free_vram_gb=free_mb / 1024.0,
                driver_version=driver,
                compute_capability=compute_cap,
                available=True,
            )
        )
    return gpus


def probe_gpus_via_nvidia_smi() -> list[GpuInfo]:
    """Use ``nvidia-smi --query-gpu`` to enumerate GPUs without torch."""
    code, stdout, _ = try_run(
        [
            "nvidia-smi",
            f"--query-gpu={NVIDIA_SMI_QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0 or not stdout.strip():
        return []
    return _parse_nvidia_smi(stdout)


def _driver_version_via_nvidia_smi() -> str:
    """Best-effort actual NVIDIA driver version (e.g. ``560.94``).

    Torch does not expose this anywhere; ``torch.version.cuda`` is the CUDA
    *runtime* version bundled with the wheel, which is a different number and
    must not be reported as the driver version.
    """
    code, stdout, _ = try_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if code != 0 or not stdout.strip():
        return ""
    return stdout.strip().splitlines()[0].strip()


def _probe_torch_cuda() -> CudaInfo:
    info = CudaInfo()
    try:
        import torch
    except Exception:  # pragma: no cover - torch optional
        return info
    info.torch_available = True
    info.torch_version = getattr(torch, "__version__", "")
    cuda_available = bool(getattr(torch.version, "cuda", None)) and torch.cuda.is_available()
    info.torch_cuda_available = cuda_available
    info.torch_cuda_version = getattr(torch.version, "cuda", "") or ""
    if cuda_available:
        # NOTE: this is the actual NVIDIA driver version, not
        # torch.version.cuda (which is the CUDA runtime the torch wheel was
        # built against) — the two are easy to conflate but not the same
        # number (e.g. driver 560.94 vs CUDA runtime 12.4).
        driver_version = _driver_version_via_nvidia_smi()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info.gpus.append(
                GpuInfo(
                    index=i,
                    name=props.name,
                    total_vram_gb=total / (1024**3),
                    free_vram_gb=free / (1024**3),
                    driver_version=driver_version,
                    compute_capability=f"{props.major}.{props.minor}",
                    available=True,
                )
            )
    return info


def probe_cuda() -> CudaInfo:
    """Return CUDA/GPU info, preferring torch but falling back to ``nvidia-smi``."""
    info = _probe_torch_cuda()
    if not info.gpus:
        info.gpus = probe_gpus_via_nvidia_smi()
    return info


def host_info() -> dict[str, str]:
    """Static host metadata, useful for tagging benchmark results."""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
