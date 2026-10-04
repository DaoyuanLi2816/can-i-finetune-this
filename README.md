<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/assets/logo.png" alt="canifinetune GPU preflight logo" width="104">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/banner.svg" alt="canifinetune — Can I fine-tune this LLM on my GPU? Estimate a memory budget, measure local training, and generate a runnable recipe." width="880">
</p>

<p align="center">
  <a href="https://daoyuanli2816.github.io/can-i-finetune-this/">Documentation</a> ·
  <a href="https://daoyuanli2816.github.io/can-i-finetune-this/quickstart/">Quickstart</a> ·
  <a href="https://pypi.org/project/canifinetune/">PyPI</a> ·
  <a href="https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.1/README.zh-CN.md">中文</a>
</p>

[![CI](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/canifinetune.svg)](https://pypi.org/project/canifinetune/)
[![Docs](https://img.shields.io/badge/docs-online-0f766e)](https://daoyuanli2816.github.io/can-i-finetune-this/)
[![Python](https://img.shields.io/badge/core-Python%203.10–3.14-2563eb)](https://daoyuanli2816.github.io/can-i-finetune-this/compatibility/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.1/LICENSE)

**Can I fine-tune this LLM on my GPU? Make a plan before loading the weights.**

You have one consumer NVIDIA GPU and an open-weight model in mind. How much
memory will training need? Which sequence length, batch size and LoRA rank
should you start with? What should you change when the budget is tight?

`canifinetune` turns those questions into a memory breakdown, configuration
suggestions and a runnable recipe. Then it helps you measure an actual local run
and compare the observation with the plan. Core estimation needs **no PyTorch
and no model-weight download**.

## Your first result

Start in a virtual environment; you do not need to clone this repository.

```console
python -m pip install canifinetune==0.4.1
canifinetune estimate --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --gpu-vram-gb 16 --seq-len 2048 --offline
canifinetune demo
```

The estimate is **8.420 GiB**, `YES` against a 16 GiB budget, with heuristic
confidence `medium`. It includes the stock logits/loss allocation and a separate
safety allowance. A planning result is not a guarantee that training will fit.

`demo` serves a local interactive entry at [127.0.0.1:8765](http://127.0.0.1:8765).
Choose a model, training settings and total/currently-free memory; inspect the
breakdown and copy matching CLI commands. It uses the same Python estimator.

<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/demo-desktop.jpg" alt="Local preflight demo: model/configuration form, 8.42 GiB memory breakdown and matching commands" width="880">
</p>

## Choose your path

| You want to… | Start here | What you get |
| --- | --- | --- |
| **Check a memory budget** | [Estimate and recommend](https://daoyuanli2816.github.io/can-i-finetune-this/workflows/) | A component breakdown, assumptions and candidate configurations |
| **Run a real fine-tune** | [PyPI-only quickstart](https://daoyuanli2816.github.io/can-i-finetune-this/quickstart/) | A pinned 0.5B QLoRA recipe, real updates, an adapter and a reload check |
| **Try training without Hub weights** | [Offline CPU smoke](https://daoyuanli2816.github.io/can-i-finetune-this/quickstart/#linux-shell-wsl-shell) | A locally created tiny model and full update/save/reload execution |
| **Understand or contribute evidence** | [Measurements and limits](https://daoyuanli2816.github.io/can-i-finetune-this/evidence/) | Defined metrics, raw observations and opt-in redacted export |

## How it works

<picture>
  <source media="(max-width: 760px)" srcset="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/assets/architecture-mobile.svg">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/assets/architecture.svg" alt="Model metadata, GPU memory and one shared configuration feed the estimator and recommender; installed recipes and benchmarks load weights, record execution and produce reviewable evidence." width="1000">
</picture>

One configuration contract connects the plan to execution. The estimate accounts
for weights, quantization, trainable gradients, optimizer state, activations,
logits/loss and overhead. Generated recipes use one Transformers/PEFT runtime,
preserve system and multi-turn data, and record requested/effective settings.
Benchmarks cover loading, first optimizer-state allocation and the bounded updates.

- **Inspect the budget.** See where memory goes, including large-vocabulary loss
  buffers and the unquantized parts of QLoRA models.
- **Carry the configuration through.** Dtype, attention, targets, optimizer,
  checkpointing, quantization and supervision have explicit meanings and errors.
- **Get usable artifacts.** Recipes save a full model or adapter through a staged
  save, record the outcome and provide a reload/generation command.
- **Keep evidence traceable.** Reports separate measured peaks, planning safety,
  historical fits, independent observations and unreviewed community submissions.

[Architecture guide](https://daoyuanli2816.github.io/can-i-finetune-this/how-it-works/) ·
[Training/data contract](https://daoyuanli2816.github.io/can-i-finetune-this/training/)

## What has been measured

On a native Windows RTX 4080, four predeclared Qwen2.5-0.5B-Instruct cases covered
LoRA/QLoRA at sequence 256/512, with three updates each. Reserved peaks ranged
from **1.666 to 2.398 GiB**. Old/new estimator predictions were identical and
conservative: **46.2% MAPE**, four overestimates, no demonstrated accuracy gain.

This is a small prospective cohort on one GPU/model, separate from historical
measurements used during estimator development. It does not establish a general
OOM probability. Candidate/public-package qualification additionally executes
CPU full/LoRA and CUDA LoRA/QLoRA updates, saves and reloads. Tiny smoke proves the
pipeline, not useful fine-tuned language quality.

[Prospective validation and raw records](https://daoyuanli2816.github.io/can-i-finetune-this/validation-0.4.0/) ·
[Historical baselines](https://daoyuanli2816.github.io/can-i-finetune-this/rtx4080_baselines/)

## Install and compatibility

| Layer | Install | Scope |
| --- | --- | --- |
| Core | `pip install canifinetune==0.4.1` | Python 3.10–3.14; estimate, recommend, recipes, reports and local demo |
| Training | Qualified Torch wheel, then `pip install "canifinetune[train]==0.4.1"` with constraints | Python 3.12; real training and benchmarks |
| Reporting extras | `pip install "canifinetune[report]==0.4.1"` | Optional pandas/tabulate |

Training uses Torch 2.6, minimum/recommended Transformers/PEFT/Accelerate stacks
and bitsandbytes 0.49.2. The quickstart separates Windows and Linux/WSL commands.
Native Windows CPU/CUDA and Linux CPU are qualified; WSL GPU, other GPUs, Flash
Attention and Liger are not qualified. CPU requires fp32; pre-quantized bases,
remote model code and distributed training are outside supported execution.
Inference artifacts do not include full optimizer/RNG resume state.

[Compatibility and migration](https://daoyuanli2816.github.io/can-i-finetune-this/compatibility/) ·
[Troubleshooting](https://daoyuanli2816.github.io/can-i-finetune-this/troubleshooting/)

## Contribute and develop

Improve a documented model family, reproduce a bounded measurement, or make the
first-use workflow clearer. Measurements are **manual and opt-in**, redacted by
default and unreviewed until checked; uploads do not automatically enter fits.

```console
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest -q -m "not training" --cov=canifinetune --cov-fail-under=70
python scripts/check_generated.py
```

[Contributing](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.1/CONTRIBUTING.md) ·
[Changelog](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.1/CHANGELOG.md) ·
[Release process](https://daoyuanli2816.github.io/can-i-finetune-this/releasing/) ·
[License](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.1/LICENSE)

MIT. Maintainer: Daoyuan Li.
