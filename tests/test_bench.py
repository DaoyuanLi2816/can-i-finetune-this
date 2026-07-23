from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from canifinetune.bench import memory_trace
from canifinetune.bench.memory_trace import MemorySnapshot
from canifinetune.bench.oom import is_oom, make_oom_report
from canifinetune.bench.runner import BenchConfig, _build_optimizer, run_bench
from canifinetune.bench.synthetic_data import make_text_dataset


def _snapshot(stage: str) -> MemorySnapshot:
    return MemorySnapshot(stage, 0, 0, 0, 0, 0, 16)


def test_memory_snapshot_without_cuda(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    snap = memory_trace.snapshot("cpu")
    memory_trace.reset_peak()
    memory_trace.empty_cache()

    assert snap.stage == "cpu"
    assert snap.allocated_gb == 0


def test_oom_helpers():
    exc = RuntimeError("CUDA out of memory while allocating")
    assert is_oom(exc)
    report = make_oom_report("forward", exc)
    assert report.happened
    assert report.stage == "forward"


def test_text_dataset_has_requested_rows():
    rows = make_text_dataset(3, seq_len_chars=32)
    assert len(rows) == 3
    assert all(set(row) == {"instruction", "input", "output"} for row in rows)


def test_bench_config_rejects_invalid_dropout():
    with pytest.raises(ValueError):
        BenchConfig(model_id="x/y", lora_dropout=1.0)


def test_8bit_optimizer_fails_instead_of_silent_fallback(monkeypatch):
    fake_torch = SimpleNamespace(optim=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)

    with pytest.raises(RuntimeError, match="requires a working bitsandbytes"):
        _build_optimizer([], "paged_adamw_8bit")


def test_unknown_optimizer_is_rejected(monkeypatch):
    fake_torch = SimpleNamespace(optim=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        _build_optimizer([], "mystery")


def test_run_bench_reports_missing_cuda(monkeypatch):
    from canifinetune.bench import runner

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner, "_gpu_snapshot_dict", lambda: {"available": False})
    monkeypatch.setattr(runner, "_torch_env", lambda: {})

    result = run_bench(
        BenchConfig(
            model_id="sshleifer/tiny-gpt2",
            record_estimate=False,
        )
    )

    assert not result.success
    assert any("CUDA not available" in note for note in result.notes)


def test_batch_failure_marks_result_failed(monkeypatch):
    from canifinetune.bench import runner

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(runner, "_gpu_snapshot_dict", lambda: {"total_vram_gb": 16})
    monkeypatch.setattr(runner, "_torch_env", lambda: {})
    monkeypatch.setattr(runner, "_safe_clear", lambda: None)
    monkeypatch.setattr(runner, "reset_peak", lambda: None)
    monkeypatch.setattr(runner, "snapshot", _snapshot)

    class FakeModel:
        def parameters(self):
            return []

        def train(self):
            return None

    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda cfg: (FakeModel(), SimpleNamespace(vocab_size=100, model_type="gpt2")),
    )
    monkeypatch.setattr(runner, "_attach_lora", lambda model, cfg, family: model)
    monkeypatch.setattr(runner, "_build_optimizer", lambda params, name: object())
    monkeypatch.setattr(
        runner,
        "make_batch",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad batch")),
    )

    result = run_bench(
        BenchConfig(
            model_id="sshleifer/tiny-gpt2",
            record_estimate=False,
            optimizer="adamw_torch",
        )
    )

    assert not result.success
    assert any("batch generation failed" in note for note in result.notes)
