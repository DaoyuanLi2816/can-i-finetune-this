"""Markdown rendering for bench results and comparisons."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt_gb(v: Any) -> str:
    try:
        return f"{float(v):.2f} GB"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v: Any, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def render_report_markdown(result_paths: Iterable[Path]) -> str:
    """Render one Markdown document covering each bench result in detail."""
    sections: list[str] = []
    sections.append("# canifinetune benchmark report\n")
    sections.append(
        "Each section below corresponds to one benchmark result JSON. "
        "`estimated` is what the static estimator predicted, `measured` is "
        "what was observed on this machine.\n"
    )
    for p in result_paths:
        try:
            data = _read_result(Path(p))
        except Exception as e:
            sections.append(f"## {p}\n\n_Could not parse result: {e}_\n")
            continue
        sections.append(_render_one_result(Path(p), data))
    return "\n".join(sections).rstrip() + "\n"


def _render_one_result(path: Path, data: dict[str, Any]) -> str:
    cfg = data.get("config", {})
    gpu = data.get("gpu", {})
    env = data.get("env", {})
    measured = data.get("measured", {})
    oom = data.get("oom", {})
    success = data.get("success", True)
    status = "OK" if success and not oom.get("happened") else "FAILED"
    out: list[str] = []
    out.append(f"## {path.name}  —  {status}\n")
    out.append("**Configuration**\n")
    out.append("")
    out.append("| Field | Value |")
    out.append("| --- | --- |")
    out.append(f"| model | `{cfg.get('model_id', '?')}` |")
    out.append(f"| method | {cfg.get('method', '?')} |")
    out.append(f"| seq_len | {cfg.get('seq_len', '?')} |")
    out.append(f"| micro_batch_size | {cfg.get('micro_batch_size', '?')} |")
    out.append(f"| steps | {cfg.get('steps', '?')} |")
    out.append(f"| lora_rank | {cfg.get('lora_rank', '?')} |")
    out.append(f"| quantization | {cfg.get('quantization', '?')} |")
    out.append(f"| optimizer | {cfg.get('optimizer', '?')} |")
    out.append(f"| gradient_checkpointing | {cfg.get('gradient_checkpointing', '?')} |")
    out.append(f"| attention | {cfg.get('attention_implementation', '?')} |")
    out.append("")
    out.append("**Environment**\n")
    out.append("")
    out.append("| Field | Value |")
    out.append("| --- | --- |")
    out.append(f"| GPU | {gpu.get('name', '?')} ({_fmt_gb(gpu.get('total_vram_gb'))}) |")
    out.append(f"| torch | {env.get('torch_version', '?')} |")
    out.append(f"| CUDA | {env.get('cuda_version', '?')} |")
    out.append(f"| bf16 | {env.get('bf16_supported', '?')} |")
    out.append("")
    out.append("**Memory: estimated vs measured**\n")
    out.append("")
    out.append("| Metric | Value |")
    out.append("| --- | --- |")
    out.append(f"| estimated total | {_fmt_gb(data.get('estimated_total_gb'))} |")
    out.append(f"| measured peak (reserved) | {_fmt_gb(measured.get('peak_reserved_gb'))} |")
    out.append(f"| measured peak (allocated) | {_fmt_gb(measured.get('peak_allocated_gb'))} |")
    out.append(f"| final allocated | {_fmt_gb(measured.get('final_allocated_gb'))} |")
    out.append(f"| tokens/sec | {_fmt_num(data.get('tokens_per_second'))} |")
    out.append(f"| avg step time | {_fmt_num(data.get('avg_step_time_s'), 4)} s |")
    out.append(f"| last-step loss | {_fmt_num(measured.get('loss_last_step'), 4)} |")
    if oom.get("happened"):
        out.append("")
        out.append("**OOM**\n")
        out.append(f"- stage: `{oom.get('stage')}`")
        out.append(f"- message: `{oom.get('message')}`")
    notes = data.get("notes") or []
    if notes:
        out.append("")
        out.append("**Notes**\n")
        for n in notes:
            out.append(f"- {n}")
    out.append("")
    repro = data.get("reproduce_command")
    if repro:
        out.append("**Reproduce**\n")
        out.append("```bash")
        out.append(repro)
        out.append("```")
        out.append("")
    return "\n".join(out)


def render_compare_markdown(result_paths: Iterable[Path]) -> str:
    """Render a single Markdown table comparing multiple bench results."""
    rows: list[list[str]] = []
    headers = [
        "model",
        "method",
        "seq",
        "bs",
        "rank",
        "quant",
        "ckpt",
        "opt",
        "peak GB (meas)",
        "estimated GB",
        "tok/s",
        "OOM?",
    ]
    for p in result_paths:
        try:
            data = _read_result(Path(p))
        except Exception:
            continue
        cfg = data.get("config", {})
        measured = data.get("measured", {})
        oom = data.get("oom", {})
        rows.append(
            [
                f"`{cfg.get('model_id', '?')}`",
                cfg.get("method", "?"),
                str(cfg.get("seq_len", "?")),
                str(cfg.get("micro_batch_size", "?")),
                str(cfg.get("lora_rank", "?")),
                cfg.get("quantization", "?"),
                "on" if cfg.get("gradient_checkpointing") else "off",
                cfg.get("optimizer", "?"),
                _fmt_num(measured.get("peak_reserved_gb")),
                _fmt_num(data.get("estimated_total_gb")),
                _fmt_num(data.get("tokens_per_second")),
                "yes" if oom.get("happened") else "no",
            ]
        )

    if not rows:
        return "_No benchmark results found._\n"

    out: list[str] = []
    out.append("# canifinetune benchmark comparison\n")
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    out.append("")
    return "\n".join(out)
