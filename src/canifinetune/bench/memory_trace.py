"""Thin wrappers around ``torch.cuda`` memory stats.

Importing this module does not import torch; ``snapshot`` and friends import
torch lazily, so the rest of the package stays usable on CPU-only installs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MemorySnapshot:
    stage: str
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    max_reserved_gb: float
    free_gb: float
    total_gb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "allocated_gb": round(self.allocated_gb, 4),
            "reserved_gb": round(self.reserved_gb, 4),
            "max_allocated_gb": round(self.max_allocated_gb, 4),
            "max_reserved_gb": round(self.max_reserved_gb, 4),
            "free_gb": round(self.free_gb, 4),
            "total_gb": round(self.total_gb, 4),
        }


def reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def snapshot(stage: str) -> MemorySnapshot:
    import torch

    if not torch.cuda.is_available():
        return MemorySnapshot(
            stage=stage,
            allocated_gb=0,
            reserved_gb=0,
            max_allocated_gb=0,
            max_reserved_gb=0,
            free_gb=0,
            total_gb=0,
        )
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    max_res = torch.cuda.max_memory_reserved() / (1024**3)
    free_b, total_b = torch.cuda.mem_get_info()
    return MemorySnapshot(
        stage=stage,
        allocated_gb=alloc,
        reserved_gb=reserved,
        max_allocated_gb=max_alloc,
        max_reserved_gb=max_res,
        free_gb=free_b / (1024**3),
        total_gb=total_b / (1024**3),
    )


def empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
