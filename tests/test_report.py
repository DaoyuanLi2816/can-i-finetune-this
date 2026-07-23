from __future__ import annotations

import json
from pathlib import Path

from canifinetune.reports import (
    render_compare_html,
    render_compare_markdown,
    render_report_html,
    render_report_markdown,
)


def _write_fake_result(path: Path, *, model: str, seq: int, oom: bool) -> Path:
    data = {
        "config": {
            "model_id": model,
            "method": "qlora",
            "seq_len": seq,
            "micro_batch_size": 1,
            "lora_rank": 16,
            "quantization": "nf4_double_quant",
            "optimizer": "paged_adamw_8bit",
            "gradient_checkpointing": True,
            "attention_implementation": "sdpa",
            "steps": 3,
        },
        "model_family": "qwen2",
        "env": {"torch_version": "2.4.1+cu121", "cuda_version": "12.1", "bf16_supported": True},
        "gpu": {"name": "NVIDIA GeForce RTX 4080", "total_vram_gb": 16.0, "free_vram_gb": 14.5},
        "snapshots": [],
        "oom": {
            "happened": oom,
            "stage": "forward" if oom else "",
            "message": "fake" if oom else "",
        },
        "measured": {
            "peak_reserved_gb": 5.2,
            "peak_allocated_gb": 4.7,
            "final_allocated_gb": 4.1,
            "loss_last_step": 2.34,
        },
        "tokens_per_second": 1200.5,
        "avg_step_time_s": 0.21,
        "estimated_total_gb": 5.5,
        "notes": [],
        "success": not oom,
        "method": "qlora",
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_report_md_contains_each_result(tmp_path: Path):
    a = _write_fake_result(
        tmp_path / "a.json", model="Qwen/Qwen2.5-1.5B-Instruct", seq=1024, oom=False
    )
    b = _write_fake_result(tmp_path / "b.json", model="Qwen/Qwen2.5-3B-Instruct", seq=512, oom=True)
    md = render_report_markdown([a, b])
    assert "Qwen2.5-1.5B-Instruct" in md
    assert "Qwen2.5-3B-Instruct" in md
    assert "OOM" in md  # the OOM section should appear


def test_compare_md_table_rows_match_inputs(tmp_path: Path):
    a = _write_fake_result(
        tmp_path / "a.json", model="Qwen/Qwen2.5-1.5B-Instruct", seq=1024, oom=False
    )
    b = _write_fake_result(tmp_path / "b.json", model="Qwen/Qwen2.5-3B-Instruct", seq=512, oom=True)
    md = render_compare_markdown([a, b])
    # Two data rows + 2 header rows = 4 lines starting with '|'
    pipe_rows = [line for line in md.splitlines() if line.startswith("|")]
    assert len(pipe_rows) >= 4
    assert any("yes" in r for r in pipe_rows)  # OOM column for b
    assert any("no" in r for r in pipe_rows)  # OOM column for a


def test_html_renders_some_html_tags(tmp_path: Path):
    a = _write_fake_result(
        tmp_path / "a.json", model="Qwen/Qwen2.5-1.5B-Instruct", seq=1024, oom=False
    )
    html = render_report_html([a])
    assert "<html" in html.lower()
    assert "<h1>" in html
    assert "Qwen2.5-1.5B-Instruct" in html

    compare_html = render_compare_html([a])
    assert "<table>" in compare_html


def test_report_empty_dir(tmp_path: Path):
    md = render_compare_markdown([])
    assert "No benchmark" in md
