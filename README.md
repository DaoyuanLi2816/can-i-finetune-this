# canifinetune

[![CI](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/canifinetune.svg)](https://pypi.org/project/canifinetune/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/LICENSE)

**A single-GPU LLM fine-tuning preflight: estimate a memory budget, inspect the
assumptions, generate a recipe, then measure a bounded local run.**

Start from PyPI. Core needs no PyTorch and never downloads model weights:

```console
python -m pip install canifinetune==0.4.0
canifinetune estimate --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --gpu-vram-gb 16 --seq-len 2048 --offline
canifinetune demo
```

The first command produces an **8.420 GiB planning budget**, `yes` against 16 GiB,
with a heuristic `medium` evidence grade. `demo` prints
[http://127.0.0.1:8765](http://127.0.0.1:8765): choose a catalogue model, training
configuration and total/currently-free memory, inspect the breakdown, and copy
CLI commands. It calls the same Python estimator. No account, telemetry, uploads
or UI dependencies. This is a local application; it is not a hosted service.

A budget is not a guarantee that training will fit. All `*_gb` JSON fields use
**GiB**, including historical files. `confidence` is not a calibrated probability
or error interval. Historical RTX 4080 measurements informed the coefficients;
[prospective validation](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/docs/validation-0.4.0.md) reports fresh observations
separately, including substantial overestimates. No cross-GPU accuracy claim.

## From estimate to an executable recipe

```console
canifinetune doctor
canifinetune recommend --model Qwen/Qwen2.5-0.5B-Instruct --gpu-vram-gb 16 --offline --top-k 3
canifinetune recipe --model Qwen/Qwen2.5-0.5B-Instruct --method qlora --gpu-vram-gb 16 --seq-len 256 --max-steps 2 --grad-accum 1 --output my-recipe
```

Estimate/recommend/recipe use metadata only. Known catalogue models work offline;
other Hub models may fetch config and parameter metadata. `bench` and `train.py`
load weights and actually run training. Gated models require the user's own
access approval. Remote model code is never enabled.

For CUDA training, use a Python 3.12 environment and a driver compatible with
PyTorch's CUDA 12.4 wheel. Install the qualified stack:

```console
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.0/constraints/train-recommended.txt "canifinetune[train]==0.4.0"
python my-recipe/train.py --config my-recipe/config.yaml
python my-recipe/eval_smoke.py --output-dir my-recipe/output --max-new-tokens 8
```

These are individual commands usable in PowerShell or a Linux shell. Environment
creation/activation differs by platform; see the complete
[PyPI-only walkthrough](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/docs/quickstart.md). Native Windows CPU/CUDA is verified;
Linux CPU runs in CI. WSL GPU qualification is not claimed for this release.

An adapter is saved under `output/adapter`; a full fine-tune under `output/model`.
`output/run.json` records requested/effective configuration, data statistics,
software, resolved model revision, outcome and allocator peaks. Saving is staged;
failed runs cannot be evaluated as successful. A non-empty output directory is
rejected. We save inference artifacts, not optimizer/RNG state for full resume.

## Offline training smoke without downloading a model

Install the training extra with a CPU torch wheel (or the CUDA stack above), then:

```console
canifinetune smoke-model --output tiny-local
canifinetune recipe --model tiny-local --method lora --device cpu --base-dtype fp32 --optimizer adamw_torch --seq-len 128 --max-steps 2 --grad-accum 1 --offline --output cpu-recipe
python cpu-recipe/train.py --config cpu-recipe/config.yaml
python cpu-recipe/eval_smoke.py --output-dir cpu-recipe/output --max-new-tokens 2
```

This creates a random model locally, executes real updates, saves and reloads it.
It demonstrates pipeline behavior, not useful language capability.

## Supervision and configuration

Recipes use one Transformers Trainer backend, independently of whether TRL is
installed. Default `loss_mode: all` preserves the full-token training objective.
Instruction records have `instruction`, optional `input`, and a non-empty
`output`. Chat records preserve an optional initial system message and every
alternating user/assistant turn through the tokenizer's native chat template.
Other role arrangements/tools/multimodal content are rejected explicitly.

`--loss-mode assistant` uses offset mappings for instruction completions and
native `{% generation %}` token masks for chat. Templates without those masks
are rejected; there is no string-search fallback. `--truncation error` is the
default. Explicit `right` truncation must preserve response supervision after the
causal label shift. Padding is masked by position; genuine EOS survives when
PAD and EOS share an ID. Invalid rows report their line and stop the run.
See the generated `dataset_format.md` and [training contract](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/docs/training.md).

Dtype, attention, quantization, target scope, optimizer, checkpointing and backend
share a validated configuration. Invalid/unknown settings fail; attention or
dtype are never removed in a broad exception retry. CPU requires explicit fp32.
Unsupported BF16 requires a user-selected alternative and a new estimate.
Full FP16 weight training is rejected. Custom LoRA target lists outside supported
scopes lack a reliable static model and are rejected by estimation/recipe.
Pre-quantized checkpoints are rejected; choose an unquantized base and request
quantization explicitly to keep prediction and actual loading aligned.

Liger recipes remain **experimental** (`--liger`, optional Linux/Triton dependency).
Their stock logits allocation is retained as an explicit conservative planning
proxy with low confidence; stock calibration is not applied. Fused loss and Flash
Attention accuracy/CUDA compatibility were not qualified here. Bench rejects
Liger until a matching measurement path is qualified.

## Measure, compare and share evidence

```console
canifinetune bench --model Qwen/Qwen2.5-0.5B-Instruct --method qlora --seq-len 256 --optimizer adamw_torch --steps 3 --out-dir measurements
canifinetune report --benchmarks measurements --out report.md
canifinetune compare --benchmarks measurements --out compare.md
canifinetune calibrate --benchmarks measurements --out calibration.json
canifinetune evidence-export --input measurements/YOUR_RESULT.json
```

A benchmark covers model loading, first forward/backward, first optimizer-state
allocation and the requested short probe. Allocated and reserved peaks describe
this process's torch allocator; device free/total memory are a different scope.
OOM records are boundaries without an exact peak, and missing measurements are
never replaced with zero. Safety is separate from the process peak proxy.
Use `--available-vram-gb` for a currently busy device while retaining the actual
`--gpu-vram-gb` total; memory allowances are not recalculated from the free budget.

Evidence export is explicit opt-in, preview-first and local. It removes personal
paths, detailed exceptions, model identity, GPU UUIDs and unknown fields by
allowlist. `--include-public-model` separately opts into a public Hub identifier
and revision. Review the JSON before sharing; never include private samples or
credentials. [Contribution instructions](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/CONTRIBUTING.md) explain manual issue/PR
submission. Community uploads remain unreviewed and cannot enter calibration or
independent accuracy summaries automatically. There are no automatic uploads.

Calibration is a fit to supplied data, with configuration profiles, estimator
version, family/method and GPU-capacity checks. It is not independent validation.
Historical raw observations remain unchanged; old calibration caches need
regeneration. The report defines success/error and feasibility denominators,
excludes historical fits/community uploads, and reports insufficient samples
instead of inventing missing rates.

## Compatibility and development

Core: Python 3.10–3.14, no torch. Qualified training: Python 3.12, torch 2.6,
Transformers 4.57.6 / PEFT 0.18.1 / Accelerate 1.12 (minimum combination), or
Transformers 5.8.1 / PEFT 0.19.1 / Accelerate 1.13 (recommended), bitsandbytes
0.49.2. [Constraints](https://github.com/DaoyuanLi2816/can-i-finetune-this/tree/v0.4.0/constraints) reproduce both. Newer upstream releases are
not silently included in the supported range. TRL/datasets are no longer needed.

```console
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest -q -m "not training" --cov=canifinetune --cov-fail-under=70
python scripts/check_generated.py
```

After installing a qualified training stack: `pytest -q tests/integration`.
Core coverage excludes optional torch runtime/smoke creation; real training
behavior has a separate CI job. CI also installs wheels outside the checkout.
[Release verification](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/docs/releasing.md) preserves the same tested candidate
files through GitHub Release and PyPI Trusted Publishing, then checks public
hashes and installation.

Scope: one consumer NVIDIA GPU, causal HF models, full/tiny training and
LoRA/QLoRA. Distributed training, remote code and automatic gated-model approval
are unsupported. See [changelog](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/CHANGELOG.md), [troubleshooting](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/docs/troubleshooting.md)
and [historical development baselines](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.0/docs/rtx4080_baselines.md).

MIT. Maintainer: Daoyuan Li.
