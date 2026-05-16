"""Local benchmark runner: measure actual VRAM during a few training steps."""

from .runner import BenchConfig, BenchResult, run_bench

__all__ = ["BenchConfig", "BenchResult", "run_bench"]
