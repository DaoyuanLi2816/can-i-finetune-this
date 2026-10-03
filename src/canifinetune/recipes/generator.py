"""Render a complete recipe folder for a given (model, method, ...) request.

Each recipe is package-backed: ``train.py``, ``config.yaml``, ``run.sh``,
``eval_smoke.py``, ``requirements.txt``, ``README.md``, and a few short
docs. The training script is small, opinionated, and works on a single
consumer GPU using Hugging Face Transformers + PEFT + (optionally)
bitsandbytes for QLoRA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import Field

from .. import __version__
from ..configuration import TrainingConfig, resolve_targets
from ..estimator.memory import EstimateRequest, estimate
from ..estimator.model_metadata import fetch_metadata

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class RecipeRequest(TrainingConfig):
    gradient_accumulation_steps: int = Field(8, gt=0)
    learning_rate: float = Field(2e-4, gt=0)
    max_steps: int = Field(50, gt=0)
    device: str = "cuda"
    output_dir: Path = Field(..., description="Where to write the recipe folder.")
    gpu_vram_gb: float = Field(16.0, gt=0.0)
    project_name: str = "canifinetune-recipe"
    overwrite: bool = False


@dataclass
class GeneratedRecipe:
    output_dir: Path
    files: list[Path]


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )


def _render(env: Environment, template: str, ctx: dict) -> str:
    return env.get_template(template).render(**ctx)


def _build_context(req: RecipeRequest) -> dict:
    if Path(req.model_id).is_dir():
        req = req.model_copy(update={"model_id": str(Path(req.model_id).resolve())})
    md = fetch_metadata(req.model_id, revision=req.revision, use_network=not req.local_files_only)
    target_modules = resolve_targets(req, md.family)

    # Pre-compute the static estimate so the recipe's README shows it.
    er = EstimateRequest(
        **req.training_fields(),
        gpu_vram_gb=req.gpu_vram_gb,
        use_network=not req.local_files_only,
    )
    est = estimate(er)

    return {
        "package_version": __version__,
        "runtime_config": {
            **req.training_fields(),
            "target_modules": target_modules or None,
            "learning_rate": req.learning_rate,
            "max_steps": req.max_steps,
            "device": req.device,
            "output_dir": "output",
            "dataset_path": "data/sample.jsonl",
        },
        "req": req.model_dump(),
        "model_id": req.model_id,
        "method": req.method,
        "seq_len": req.seq_len,
        "micro_batch_size": req.micro_batch_size,
        "gradient_accumulation_steps": req.gradient_accumulation_steps,
        "lora_rank": req.lora_rank,
        "lora_alpha": req.lora_alpha,
        "lora_dropout": req.lora_dropout,
        "lora_target_scope": req.lora_target_scope,
        "target_modules": target_modules,
        "base_dtype": req.base_dtype,
        "quantization": req.quantization,
        "optimizer": req.optimizer,
        "gradient_checkpointing": req.gradient_checkpointing,
        "attention_implementation": req.attention_implementation,
        "use_liger_kernel": req.use_liger_kernel,
        "learning_rate": req.learning_rate,
        "max_steps": req.max_steps,
        "project_name": req.project_name,
        "model_family": md.family,
        "model_total_params": md.total_params,
        "model_source": md.source,
        "gpu_vram_gb": req.gpu_vram_gb,
        "estimate": est.model_dump(),
        "estimate_memory": est.memory.model_dump(),
        "estimate_feasible": est.feasible,
        "estimate_confidence": est.confidence,
        "uses_4bit": req.method == "qlora",
        "extra_deps": [
            *(
                ["bitsandbytes>=0.49.2,<0.50"]
                if req.method == "qlora"
                or "8bit" in req.optimizer
                or req.optimizer.startswith("paged")
                else []
            ),
            *(["liger-kernel>=0.5"] if req.use_liger_kernel else []),
        ],
    }


_FILES = [
    ("train.py.j2", "train.py"),
    ("config.yaml.j2", "config.yaml"),
    ("run.sh.j2", "run.sh"),
    ("eval_smoke.py.j2", "eval_smoke.py"),
    ("requirements.txt.j2", "requirements.txt"),
    ("README.md.j2", "README.md"),
    ("expected_vram.md.j2", "expected_vram.md"),
    ("dataset_format.md.j2", "dataset_format.md"),
    ("troubleshooting.md.j2", "troubleshooting.md"),
    ("sample_dataset.jsonl.j2", "data/sample.jsonl"),
]


def generate_recipe(req: RecipeRequest) -> GeneratedRecipe:
    out = Path(req.output_dir)
    if out.exists() and any(out.iterdir()) and not req.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty recipe directory {out}. "
            "Choose a new directory or pass overwrite=True/--force."
        )
    out.mkdir(parents=True, exist_ok=True)
    env = _env()
    ctx = _build_context(req)
    written: list[Path] = []
    for tpl, name in _FILES:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _render(env, tpl, ctx)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return GeneratedRecipe(output_dir=out, files=written)
