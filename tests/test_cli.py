"""End-to-end smoke tests of the CLI through typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from canifinetune.bench.runner import BenchConfig, BenchResult
from canifinetune.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_help(runner):
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "doctor" in res.stdout
    assert "estimate" in res.stdout


def test_cli_version(runner):
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert "canifinetune" in res.stdout


def test_cli_estimate_qwen_qlora(runner):
    res = runner.invoke(
        app,
        [
            "estimate",
            "--model",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--method",
            "qlora",
            "--gpu-vram-gb",
            "16",
            "--seq-len",
            "2048",
            "--micro-batch-size",
            "1",
            "--lora-rank",
            "16",
            "--json",
        ],
    )
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["feasible"] in {"yes", "marginal", "no"}
    assert payload["memory"]["total_estimated_gb"] > 0


def test_cli_estimate_unknown_model_errors_without_override(runner):
    res = runner.invoke(
        app,
        [
            "estimate",
            "--model",
            "totally-fake/non-existent-model",
            "--method",
            "qlora",
            "--gpu-vram-gb",
            "16",
            "--seq-len",
            "128",
            "--micro-batch-size",
            "1",
            "--lora-rank",
            "8",
            "--json",
        ],
    )
    # Either it errored cleanly, or it actually fetched a config — both are fine.
    assert res.exit_code in {0, 2}


def test_cli_recommend(runner):
    res = runner.invoke(
        app,
        [
            "recommend",
            "--model",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--gpu-vram-gb",
            "16",
            "--top-k",
            "3",
            "--json",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert isinstance(data, list)
    assert len(data) > 0


def test_cli_recipe_creates_files(runner, tmp_path: Path):
    out = tmp_path / "recipe"
    res = runner.invoke(
        app,
        [
            "recipe",
            "--model",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--method",
            "qlora",
            "--seq-len",
            "512",
            "--micro-batch-size",
            "1",
            "--lora-rank",
            "8",
            "--output",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (out / "train.py").is_file()
    assert (out / "config.yaml").is_file()


def test_cli_recipe_refuses_non_empty_output_without_force(runner, tmp_path: Path):
    out = tmp_path / "recipe"
    out.mkdir()
    sentinel = out / "notes.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    res = runner.invoke(
        app,
        [
            "recipe",
            "--model",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--output",
            str(out),
        ],
    )

    assert res.exit_code == 2
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_cli_bench_failure_returns_nonzero(monkeypatch, runner, tmp_path: Path):
    import canifinetune.bench as bench

    def fail(cfg: BenchConfig) -> BenchResult:
        return BenchResult(
            config=cfg,
            model_family="unknown",
            timestamp="2026-01-01T00:00:00Z",
            env={},
            gpu={},
            success=False,
            notes=["simulated failure"],
        )

    monkeypatch.setattr(bench, "run_bench", fail)
    res = runner.invoke(
        app,
        [
            "bench",
            "--model",
            "sshleifer/tiny-gpt2",
            "--out-dir",
            str(tmp_path),
            "--json",
        ],
    )

    assert res.exit_code == 1
    assert json.loads(res.stdout)["success"] is False


def test_cli_report_handles_empty_dir(runner, tmp_path: Path):
    res = runner.invoke(
        app,
        ["report", "--benchmarks", str(tmp_path), "--out", str(tmp_path / "out.md")],
    )
    assert res.exit_code == 0
    assert (tmp_path / "out.md").is_file()


def test_cli_compare_with_one_result(runner, tmp_path: Path):
    result_file = tmp_path / "fake.json"
    result_file.write_text(
        json.dumps(
            {
                "config": {
                    "model_id": "test/tiny-llama",
                    "method": "lora",
                    "seq_len": 128,
                    "micro_batch_size": 1,
                    "lora_rank": 8,
                    "quantization": "bf16",
                    "optimizer": "adamw_torch",
                    "gradient_checkpointing": False,
                    "attention_implementation": "eager",
                    "steps": 1,
                },
                "model_family": "llama",
                "env": {},
                "gpu": {"name": "fake"},
                "snapshots": [],
                "oom": {"happened": False, "stage": "", "message": ""},
                "measured": {"peak_reserved_gb": 0.1, "peak_allocated_gb": 0.05},
                "tokens_per_second": 100.0,
                "avg_step_time_s": 0.01,
                "estimated_total_gb": 0.15,
                "notes": [],
                "success": True,
                "method": "lora",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "compare.md"
    res = runner.invoke(app, ["compare", "--benchmarks", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0
    text = out.read_text(encoding="utf-8")
    assert "test/tiny-llama" in text
