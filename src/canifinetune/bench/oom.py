"""Helpers to capture CUDA OOM without losing whatever progress was made."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OomReport:
    happened: bool = False
    stage: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {"happened": self.happened, "stage": self.stage, "message": self.message}


def is_oom(exc: BaseException) -> bool:
    """Return True if ``exc`` is a CUDA out-of-memory error."""
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda oom" in msg:
        return True
    try:
        import torch  # type: ignore

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return False


def make_oom_report(stage: str, exc: BaseException) -> OomReport:
    return OomReport(happened=True, stage=stage, message=str(exc)[:600])
