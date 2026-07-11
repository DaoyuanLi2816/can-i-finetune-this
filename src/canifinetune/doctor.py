"""Environment probe used by ``canifinetune doctor``.

This module never *requires* torch or any training dep — it discovers what is
installed and reports it, instead of raising.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from .utils.gpu import CudaInfo, GpuInfo, host_info, probe_cuda
from .utils.subprocess import try_run

_OPTIONAL_LIBS = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "trl",
    "datasets",
    "huggingface_hub",
    "pydantic",
    "typer",
    "rich",
    "flash_attn",
]


@dataclass
class LibInfo:
    name: str
    installed: bool
    version: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    host: dict[str, str] = field(default_factory=dict)
    python: dict[str, Any] = field(default_factory=dict)
    cuda: dict[str, Any] = field(default_factory=dict)
    libraries: list[LibInfo] = field(default_factory=list)
    tiny_model_load: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "python": self.python,
            "cuda": self.cuda,
            "libraries": [lib.to_dict() for lib in self.libraries],
            "tiny_model_load": self.tiny_model_load,
            "issues": self.issues,
        }


def _python_info() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
    }


def _probe_library(name: str) -> LibInfo:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return LibInfo(name=name, installed=False, note=f"import failed: {e.__class__.__name__}")
    version = getattr(mod, "__version__", "")
    if not version:
        # Some packages (e.g. rich) don't set a top-level __version__; fall
        # back to the installed-distribution metadata so the table doesn't
        # show a blank version for a library that is clearly installed.
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = ""
    return LibInfo(name=name, installed=True, version=str(version))


def _try_tiny_model_load() -> dict[str, Any]:
    """Try to instantiate a tiny model from local config.

    We deliberately avoid downloading anything: we synthesize a 2-layer GPT-2
    config in memory and instantiate it on CPU. That validates the transformers
    install end-to-end.
    """
    out: dict[str, Any] = {"attempted": True, "ok": False, "model": None, "error": None}
    try:
        from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore
    except Exception as e:
        out["error"] = f"transformers not importable: {e}"
        return out
    try:
        cfg = GPT2Config(
            vocab_size=1024, n_positions=64, n_embd=32, n_layer=2, n_head=2
        )
        model = GPT2LMHeadModel(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        out["ok"] = True
        out["model"] = "in-memory GPT-2 (2 layers, hidden=32)"
        out["params"] = int(n_params)
    except Exception as e:
        out["error"] = f"model instantiation failed: {e}"
    return out


def _check_nvidia_smi() -> dict[str, Any]:
    code, stdout, _ = try_run(["nvidia-smi", "--version"])
    if code == 0:
        return {"available": True, "version_line": stdout.splitlines()[0] if stdout else ""}
    return {"available": False}


def _serialize_cuda(info: CudaInfo) -> dict[str, Any]:
    d = info.to_dict()
    d["nvidia_smi"] = _check_nvidia_smi()
    return d


def run_doctor() -> DoctorReport:
    """Collect everything ``canifinetune doctor`` reports."""
    report = DoctorReport()
    report.host = host_info()
    report.python = _python_info()
    cuda = probe_cuda()
    report.cuda = _serialize_cuda(cuda)
    report.libraries = [_probe_library(n) for n in _OPTIONAL_LIBS]
    report.tiny_model_load = _try_tiny_model_load()

    # Sanity-check derived issues.
    if not cuda.torch_available:
        report.issues.append(
            "torch is not installed. Install with `pip install canifinetune[train]` "
            "or `uv pip install -e .[train]` to run benchmarks."
        )
    elif not cuda.torch_cuda_available:
        report.issues.append(
            "torch is installed but cannot see CUDA. Install the matching CUDA wheel, e.g. "
            "`pip install --index-url https://download.pytorch.org/whl/cu124 torch`."
        )
    if cuda.gpus:
        gpu0: GpuInfo = cuda.gpus[0]
        if gpu0.total_vram_gb and gpu0.total_vram_gb < 4.0:
            report.issues.append(
                f"GPU has only {gpu0.total_vram_gb:.1f} GB VRAM; only tiny models can be fine-tuned."
            )
    else:
        report.issues.append("No NVIDIA GPU detected.")

    bnb = next((lib for lib in report.libraries if lib.name == "bitsandbytes"), None)
    if bnb and not bnb.installed:
        report.issues.append(
            "bitsandbytes is not installed. QLoRA (4-bit) requires it. "
            "On Windows, prefer `pip install bitsandbytes>=0.43.1` (recent versions ship Windows wheels)."
        )

    if not report.tiny_model_load.get("ok"):
        report.issues.append(
            "Tiny in-memory transformers model failed to instantiate. "
            f"Detail: {report.tiny_model_load.get('error')}"
        )
    return report
