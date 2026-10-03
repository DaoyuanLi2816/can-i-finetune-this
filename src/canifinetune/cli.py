"""Typer-based CLI: doctor / estimate / recommend / bench / calibrate / recipe / report / compare."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer.core import TyperGroup

from . import __version__
from .utils.logging import get_logger, to_json

# Rich renders table rules/truncation with Unicode characters (box-drawing,
# "…"). On Windows the default stdout/stderr codepage is often not UTF-8
# (e.g. cp936/cp1252), which turns those into "?"/mojibake instead of raising
# — reconfigure both streams to UTF-8 up front so every table/panel below
# renders correctly regardless of the OS locale.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        with contextlib.suppress(ValueError):
            _reconfigure(encoding="utf-8")

console = Console()
err_console = Console(stderr=True)
log = get_logger("cli")


class UserErrorGroup(TyperGroup):
    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (ValueError, OSError) as exc:
            err_console.print(f"error: {exc}", markup=False)
            raise typer.Exit(2) from exc


app = typer.Typer(
    cls=UserErrorGroup,
    name="canifinetune",
    help="Estimate, benchmark, and generate fine-tuning recipes for LLMs on consumer GPUs.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)


def _print_version_and_exit(value: bool) -> None:
    if value:
        console.print(f"canifinetune {__version__}")
        raise typer.Exit(0)


@app.callback()
def _root(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Print version and exit.",
        callback=_print_version_and_exit,
        is_eager=True,
    ),
) -> None:
    """canifinetune root callback."""
    return None


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command("doctor")
def cmd_doctor(
    json_out: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Show environment summary (Python, PyTorch, CUDA, GPU, libraries)."""
    from .doctor import run_doctor

    report = run_doctor()
    if json_out:
        print(to_json(report.to_dict()))
        return

    table = Table(title="canifinetune doctor", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Python", f"{report.python['version']} ({report.python['implementation']})")
    table.add_row("Executable", report.python["executable"])
    table.add_row(
        "Platform", f"{report.host.get('platform', '?')} {report.host.get('platform_release', '')}"
    )
    cuda = report.cuda
    table.add_row(
        "Torch",
        f"{cuda.get('torch_version', '-')} (CUDA available: {cuda.get('torch_cuda_available')})",
    )
    table.add_row("Torch CUDA", cuda.get("torch_cuda_version", "-"))
    for g in cuda.get("gpus", []):
        table.add_row(
            f"GPU {g['index']}",
            f"{g['name']}  {g['total_vram_gb']:.2f} GiB total / {g['free_vram_gb']:.2f} GiB free  "
            f"(cc {g['compute_capability']}, driver {g['driver_version']})",
        )
    console.print(table)

    lib_table = Table(title="Libraries", show_lines=False)
    lib_table.add_column("Library")
    lib_table.add_column("Installed")
    lib_table.add_column("Version")
    for lib in report.libraries:
        lib_table.add_row(
            lib.name,
            "yes" if lib.installed else "no",
            lib.version or lib.note,
        )
    console.print(lib_table)

    tm = report.tiny_model_load
    console.print(
        Panel.fit(
            f"Tiny in-memory transformers model: {'OK' if tm.get('ok') else 'FAILED'}\n"
            f"{tm.get('model') or tm.get('error') or ''}",
            title="Tiny model load",
        )
    )

    if report.issues:
        issues = "\n".join(f"- {x}" for x in report.issues)
        console.print(Panel.fit(issues, title="Issues", style="yellow"))
    else:
        console.print(Panel.fit("No blocking issues detected.", title="Issues", style="green"))


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


@app.command("estimate")
def cmd_estimate(
    model: str = typer.Option(..., "--model", help="HF model id, e.g. Qwen/Qwen2.5-1.5B-Instruct"),
    gpu_vram_gb: float = typer.Option(
        ..., "--gpu-vram-gb", min=0.001, help="GPU VRAM in GiB (e.g. 16 for an RTX 4080)."
    ),
    method: str = typer.Option("qlora", "--method", help="full | lora | qlora"),
    seq_len: int = typer.Option(2048, "--seq-len", min=1),
    micro_batch_size: int = typer.Option(1, "--micro-batch-size", min=1),
    lora_rank: int = typer.Option(16, "--lora-rank", min=1),
    lora_target_scope: str = typer.Option(
        "attention", "--target-scope", help="attention | all_linear | conservative"
    ),
    quantization: str = typer.Option("nf4_double_quant", "--quantization"),
    base_dtype: str = typer.Option("bf16", "--base-dtype"),
    optimizer: str = typer.Option("paged_adamw_8bit", "--optimizer"),
    gradient_checkpointing: bool = typer.Option(
        True, "--gradient-checkpointing/--no-gradient-checkpointing"
    ),
    attention_implementation: str = typer.Option("sdpa", "--attn"),
    use_calibration: bool = typer.Option(False, "--use-calibration"),
    calibration_path: Path | None = typer.Option(None, "--calibration-path"),
    override_json: Path | None = typer.Option(
        None, "--override-json", help="Path to JSON with arch override."
    ),
    available_vram_gb: float | None = typer.Option(None, "--available-vram-gb", min=0.001),
    offline: bool = typer.Option(False, "--offline"),
    revision: str = typer.Option("main", "--revision"),
    loss_backend: str = typer.Option("stock", "--loss-backend"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Static memory + feasibility estimate."""
    from .estimator.calibration import load_calibration
    from .estimator.memory import EstimateRequest, estimate

    calib = None
    if use_calibration:
        calib = load_calibration(calibration_path)
        if not calib.has_data():
            err_console.print(
                "[yellow]warning:[/yellow] --use-calibration set but no calibration data found "
                f"at {calibration_path or 'default cache path'}."
            )

    override = None
    if override_json:
        override = json.loads(override_json.read_text(encoding="utf-8"))

    req = EstimateRequest(
        model_id=model,
        available_vram_gb=available_vram_gb,
        use_network=not offline,
        local_files_only=offline,
        revision=revision,
        loss_backend=loss_backend,
        method=method,
        gpu_vram_gb=gpu_vram_gb,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        lora_rank=lora_rank,
        lora_target_scope=lora_target_scope,
        quantization=quantization,
        base_dtype=base_dtype,
        optimizer=optimizer,
        gradient_checkpointing=gradient_checkpointing,
        attention_implementation=attention_implementation,
        calibration=calib,
        override=override,
    )
    try:
        est = estimate(req)
    except ValueError as e:
        err_console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(2)

    if json_out:
        print(to_json(est.model_dump()))
        return

    _print_estimate(est)


def _print_estimate(est) -> None:
    color = {"yes": "green", "marginal": "yellow", "no": "red"}.get(est.feasible, "white")
    console.print(
        Panel.fit(
            f"[bold {color}]feasible: {est.feasible.upper()}[/bold {color}]    "
            f"ratio = {est.feasibility_ratio:.2f}    confidence = {est.confidence}",
            title=f"{est.request.model_id}  ({est.request.method})",
        )
    )

    mem = est.memory
    table = Table(title="Memory breakdown (GiB)", show_lines=False)
    table.add_column("Component", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("static model", f"{mem.static_model_gb:.3f}")
    table.add_row("quantization overhead", f"{mem.quantization_overhead_gb:.3f}")
    table.add_row("trainable params", f"{mem.trainable_params_mb:.1f} million")
    table.add_row("gradients", f"{mem.gradients_gb:.3f}")
    table.add_row("optimizer states", f"{mem.optimizer_gb:.3f}")
    table.add_row("activations", f"{mem.activations_gb:.3f}")
    table.add_row("logits / loss", f"{mem.logits_gb:.3f}")
    table.add_row("CUDA / fragmentation", f"{mem.cuda_overhead_gb:.3f}")
    table.add_row("safety margin", f"{mem.safety_margin_gb:.3f}")
    table.add_row("[bold]total[/bold]", f"[bold]{mem.total_estimated_gb:.3f}[/bold]")
    console.print(table)

    if est.assumptions:
        console.print(Panel.fit("\n".join(f"- {a}" for a in est.assumptions), title="Assumptions"))
    if est.warnings:
        console.print(
            Panel.fit("\n".join(f"- {w}" for w in est.warnings), title="Warnings", style="yellow")
        )

    if est.feasible != "yes":
        from .estimator.recommender import suggest_degradations

        steps = suggest_degradations(est.request)
        table = Table(title="Suggested degradations", show_lines=False)
        table.add_column("#")
        table.add_column("Change")
        table.add_column("Est. GiB", justify="right")
        table.add_column("Feasible")
        for i, s in enumerate(steps, 1):
            table.add_row(
                str(i),
                s.description,
                f"{s.estimate.memory.total_estimated_gb:.2f}",
                s.estimate.feasible,
            )
        console.print(table)


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------


@app.command("recommend")
def cmd_recommend(
    model: str = typer.Option(..., "--model"),
    gpu_vram_gb: float = typer.Option(..., "--gpu-vram-gb", min=0.001),
    top_k: int = typer.Option(5, "--top-k", min=1),
    offline: bool = typer.Option(False, "--offline"),
    json_out: bool = typer.Option(False, "--json"),
    override_json: Path | None = typer.Option(None, "--override-json"),
) -> None:
    """Search for feasible (method, seq_len, batch, rank, quant, ...) combinations."""
    from .estimator.recommender import recommend_configs

    override = None
    if override_json:
        override = json.loads(override_json.read_text(encoding="utf-8"))

    try:
        recs = recommend_configs(
            model_id=model,
            gpu_vram_gb=gpu_vram_gb,
            top_k=top_k,
            use_network=not offline,
            override=override,
        )
    except ValueError as e:
        err_console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(2)

    if json_out:
        print(to_json([r.model_dump() for r in recs]))
        return

    if not recs:
        console.print("[red]No feasible configurations found in the search grid.[/red]")
        return

    table = Table(
        title=f"Top {len(recs)} configurations for {model} on {gpu_vram_gb} GiB", show_lines=False
    )
    table.add_column("#")
    table.add_column("method")
    table.add_column("seq")
    table.add_column("bs")
    table.add_column("rank")
    table.add_column("scope")
    table.add_column("ckpt")
    table.add_column("quant")
    table.add_column("opt")
    table.add_column("est GiB", justify="right")
    table.add_column("feasible")
    for i, r in enumerate(recs, 1):
        req = r.estimate.request
        table.add_row(
            str(i),
            req.method,
            str(req.seq_len),
            str(req.micro_batch_size),
            str(req.lora_rank),
            req.lora_target_scope,
            "on" if req.gradient_checkpointing else "off",
            req.quantization,
            req.optimizer,
            f"{r.estimate.memory.total_estimated_gb:.2f}",
            r.estimate.feasible,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


@app.command("bench")
def cmd_bench(
    model: str = typer.Option(..., "--model"),
    method: str = typer.Option("lora", "--method", help="full | lora | qlora"),
    seq_len: int = typer.Option(128, "--seq-len", min=1),
    micro_batch_size: int = typer.Option(1, "--micro-batch-size", min=1),
    steps: int = typer.Option(2, "--steps", min=1),
    lora_rank: int = typer.Option(8, "--lora-rank", min=1),
    lora_target_scope: str = typer.Option("attention", "--target-scope"),
    quantization: str = typer.Option("nf4_double_quant", "--quantization"),
    base_dtype: str = typer.Option("bf16", "--base-dtype"),
    optimizer: str = typer.Option("paged_adamw_8bit", "--optimizer"),
    gradient_checkpointing: bool = typer.Option(
        True, "--gradient-checkpointing/--no-gradient-checkpointing"
    ),
    attention_implementation: str = typer.Option("sdpa", "--attn"),
    forward_only: bool = typer.Option(False, "--forward-only"),
    out_dir: Path = typer.Option(Path("benchmarks/results"), "--out-dir"),
    device: str = typer.Option("cuda", "--device"),
    offline: bool = typer.Option(False, "--offline"),
    revision: str = typer.Option("main", "--revision"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run a local smoke benchmark and save a result JSON."""
    from .bench import BenchConfig, run_bench
    from .bench.runner import result_path_for

    cfg = BenchConfig(
        model_id=model,
        device=device,
        local_files_only=offline,
        revision=revision,
        method=method,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        steps=steps,
        lora_rank=lora_rank,
        lora_target_scope=lora_target_scope,
        quantization=quantization,
        base_dtype=base_dtype,
        optimizer=optimizer,
        gradient_checkpointing=gradient_checkpointing,
        attention_implementation=attention_implementation,
        forward_only=forward_only,
    )
    result = run_bench(cfg)
    path = result_path_for(out_dir, cfg)
    result.save(path)

    if json_out:
        print(to_json(result.model_dump()))
    else:
        _print_bench_summary(result, path)
    if not result.success or (result.oom or {}).get("happened"):
        raise typer.Exit(1)


def _print_bench_summary(result, path: Path) -> None:
    measured = result.measured or {}
    oom = result.oom or {}
    status = "OK" if result.success and not oom.get("happened") else "FAILED"
    console.print(
        Panel.fit(
            f"[bold]{status}[/bold]\n"
            f"file: {path}\n"
            f"model: {result.config.model_id} ({result.model_family})\n"
            f"method: {result.method}  seq_len: {result.config.seq_len}  bs: {result.config.micro_batch_size}\n"
            f"peak reserved: {measured.get('peak_reserved_gb', '-')} GiB  "
            f"peak allocated: {measured.get('peak_allocated_gb', '-')} GiB\n"
            f"estimated total: {result.estimated_total_gb if result.estimated_total_gb is not None else 'unknown'} GiB\n"
            f"avg step: {result.avg_step_time_s} s  tokens/sec: {result.tokens_per_second}",
            title="bench",
        )
    )
    if oom.get("happened"):
        err_console.print(
            Panel.fit(
                f"OOM at stage: {oom.get('stage')}\n{oom.get('message')}",
                title="OOM",
                style="red",
            )
        )
    if result.notes:
        console.print(Panel.fit("\n".join(f"- {n}" for n in result.notes), title="Notes"))


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


@app.command("calibrate")
def cmd_calibrate(
    benchmarks: Path = typer.Option(Path("benchmarks/results"), "--benchmarks"),
    out: Path | None = typer.Option(None, "--out"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Aggregate benchmark JSONs into a calibration file."""
    from .estimator.calibration import (
        calibration_from_result_files,
        default_calibration_path,
        save_calibration,
    )

    files = sorted(Path(benchmarks).glob("*.json"))
    if not files:
        err_console.print(f"[yellow]warning:[/yellow] no JSON files under {benchmarks}.")
    calib = calibration_from_result_files(files)
    target = out or default_calibration_path()
    save_calibration(calib, target)

    if json_out:
        print(to_json(calib.model_dump()))
        return

    console.print(
        Panel.fit(
            f"samples: {len(calib.samples)}\n"
            f"activation_scale: {calib.activation_scale:.3f}\n"
            f"weights_scale: {calib.weights_scale:.3f}\n"
            f"overhead_scale: {calib.overhead_scale:.3f}\n"
            f"note: {calib.note}\n"
            f"saved to: {target}",
            title="calibration",
        )
    )


# ---------------------------------------------------------------------------
# recipe
# ---------------------------------------------------------------------------


@app.command("recipe")
def cmd_recipe(
    model: str = typer.Option(..., "--model"),
    method: str = typer.Option("qlora", "--method"),
    seq_len: int = typer.Option(2048, "--seq-len", min=1),
    micro_batch_size: int = typer.Option(1, "--micro-batch-size", min=1),
    grad_accum: int = typer.Option(8, "--grad-accum", min=1),
    lora_rank: int = typer.Option(16, "--lora-rank", min=1),
    lora_target_scope: str = typer.Option("attention", "--target-scope"),
    learning_rate: float = typer.Option(2e-4, "--lr"),
    max_steps: int = typer.Option(50, "--max-steps", min=1),
    optimizer: str = typer.Option("paged_adamw_8bit", "--optimizer"),
    quantization: str = typer.Option("nf4_double_quant", "--quantization"),
    base_dtype: str = typer.Option("bf16", "--base-dtype"),
    gradient_checkpointing: bool = typer.Option(
        True, "--gradient-checkpointing/--no-gradient-checkpointing"
    ),
    attention_implementation: str = typer.Option("sdpa", "--attn"),
    use_liger_kernel: bool = typer.Option(
        False,
        "--liger/--no-liger",
        help="Experimental Liger kernels in Transformers; stock memory proxy remains.",
    ),
    loss_mode: str = typer.Option(
        "all",
        "--loss-mode",
        help="all | assistant; native generation masks required for assistant chat loss",
    ),
    truncation: str = typer.Option("error", "--truncation", help="error | right"),
    device: str = typer.Option("cuda", "--device"),
    offline: bool = typer.Option(False, "--offline"),
    revision: str = typer.Option("main", "--revision"),
    gpu_vram_gb: float = typer.Option(16.0, "--gpu-vram-gb", min=0.001),
    output: Path = typer.Option(..., "--output"),
    force: bool = typer.Option(
        False, "--force", help="Overwrite files in a non-empty output directory."
    ),
) -> None:
    """Generate a self-contained training recipe folder."""
    from .recipes import RecipeRequest, generate_recipe

    req = RecipeRequest(
        model_id=model,
        loss_mode=loss_mode,
        truncation=truncation,
        device=device,
        local_files_only=offline,
        revision=revision,
        method=method,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        lora_rank=lora_rank,
        lora_target_scope=lora_target_scope,
        learning_rate=learning_rate,
        max_steps=max_steps,
        optimizer=optimizer,
        quantization=quantization,
        base_dtype=base_dtype,
        gradient_checkpointing=gradient_checkpointing,
        attention_implementation=attention_implementation,
        use_liger_kernel=use_liger_kernel,
        gpu_vram_gb=gpu_vram_gb,
        output_dir=output,
        overwrite=force,
    )
    try:
        res = generate_recipe(req)
    except FileExistsError as e:
        err_console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(2)
    console.print(
        Panel.fit(
            "\n".join(f"- {p}" for p in res.files),
            title=f"Recipe written to {res.output_dir}",
            style="green",
        )
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command("report")
def cmd_report(
    benchmarks: Path = typer.Option(Path("benchmarks/results"), "--benchmarks"),
    out: Path = typer.Option(Path("report.md"), "--out"),
    html: bool = typer.Option(False, "--html"),
) -> None:
    """Render a Markdown (or HTML) report from benchmark results."""
    from .reports import render_report_html, render_report_markdown

    files = sorted(Path(benchmarks).glob("*.json"))
    if not files:
        err_console.print(f"[yellow]warning:[/yellow] no JSON files under {benchmarks}.")
    content = render_report_html(files) if html else render_report_markdown(files)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    console.print(f"wrote {out} ({len(files)} result(s))")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command("compare")
def cmd_compare(
    benchmarks: Path = typer.Option(Path("benchmarks/results"), "--benchmarks"),
    out: Path = typer.Option(Path("compare.md"), "--out"),
    html: bool = typer.Option(False, "--html"),
) -> None:
    """Render a single comparison table of multiple benchmark results."""
    from .reports import render_compare_html, render_compare_markdown

    files = sorted(Path(benchmarks).glob("*.json"))
    content = render_compare_html(files) if html else render_compare_markdown(files)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    console.print(f"wrote {out} ({len(files)} result(s))")


@app.command("demo")
def cmd_demo(port: int = typer.Option(8765, "--port", min=1, max=65535)):
    """Serve a loopback interactive estimator using the same Python core."""
    from .demo import serve

    serve(port)


@app.command("smoke-model")
def cmd_smoke_model(output: Path = typer.Option(..., "--output")):
    """Create a tiny random model/tokenizer locally for offline qualification."""
    from .training.smoke import create_smoke_model

    try:
        create_smoke_model(output)
    except ImportError as exc:
        err_console.print(
            f"Missing training dependency: {exc}. Install canifinetune[train].", markup=False
        )
        raise typer.Exit(2) from exc
    console.print(f"created tiny random model: {output}")


@app.command("evidence-export")
def cmd_evidence_export(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, "--out"),
    include_public_model: bool = typer.Option(False, "--include-public-model"),
):
    """Preview/export redacted benchmark JSON. Never uploads anything."""
    from .evidence import export_evidence

    data = export_evidence(
        json.loads(input_path.read_text(encoding="utf-8")),
        include_public_model=include_public_model,
    )
    text = json.dumps(data, indent=2, allow_nan=False)
    if out:
        with out.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
        console.print(f"wrote {out}; preview before submitting manually")
    else:
        print(text)


@app.command("evidence-validate")
def cmd_evidence_validate(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
):
    """Validate evidence as data, without executing attached commands."""
    from .evidence import SharedEvidence

    SharedEvidence.model_validate_json(input_path.read_text(encoding="utf-8"))
    console.print("valid evidence schema; upload is not maintainer verification")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
