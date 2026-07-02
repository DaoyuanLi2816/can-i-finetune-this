<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/main/docs/banner.svg" alt="canifinetune — Can I fine-tune this LLM on my GPU? Estimate up front, run local benchmarks, get ready-to-run LoRA/QLoRA recipes for 12–24 GB consumer GPUs." width="880">
</p>

<div align="center">

[![CI](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/canifinetune.svg)](https://pypi.org/project/canifinetune/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

**Estimate, benchmark, and generate fine-tuning recipes for LLMs on consumer GPUs.**

![can-i-finetune-this architecture](https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/main/docs/architecture.png)

You have one consumer-grade NVIDIA GPU. You want to fine-tune an open-weight LLM
with LoRA or QLoRA, but you do not want to download 14 GB of weights just to
discover that your 12 GB / 16 GB / 24 GB card OOMs on step 1.

`canifinetune` answers, before you spend the disk and the time:

1. Can I fine-tune this model?
2. About how much VRAM will it use?
3. What batch size / sequence length / LoRA rank / quantization should I use?
4. If I can't, how should I downsize?
5. Is there local benchmark evidence for that answer?
6. Can I get a ready-to-run Hugging Face + PEFT + TRL training script for that config?

It is a single Python package with a CLI:

```bash
canifinetune doctor
canifinetune estimate --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --gpu-vram-gb 16 --seq-len 2048 --micro-batch-size 1 --lora-rank 16
canifinetune recommend --model Qwen/Qwen2.5-1.5B-Instruct --gpu-vram-gb 16
canifinetune bench    --model sshleifer/tiny-gpt2 --method lora --steps 3
canifinetune calibrate --benchmarks benchmarks/results
canifinetune recipe   --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --output recipes/qwen2.5-1.5b-qlora-4080
canifinetune report   --benchmarks benchmarks/results --out report.md
canifinetune compare  --benchmarks benchmarks/results --out compare.md
```

What `canifinetune estimate` actually prints:

```text
+-------- Qwen/Qwen2.5-1.5B-Instruct  (qlora) ---------+
| feasible: YES    ratio = 0.53    confidence = medium |
+------------------------------------------------------+
      Memory breakdown (GB)
+--------------------------------+
| Component             |  Value |
|-----------------------+--------|
| static model          |  1.517 |
| quantization overhead |  0.072 |
| trainable params      | 4.4 MB |
| gradients             |  0.016 |
| optimizer states      |  0.010 |
| activations           |  0.689 |
| logits / loss         |  4.057 |
| CUDA / fragmentation  |  1.280 |
| safety margin         |  0.800 |
| total                 |  8.441 |
+--------------------------------+
```

On a real RTX 4080 this exact config peaks at 7.10 GB reserved — the
estimate lands ~1.3 GB above it, on the safe side, instead of promising
3 GB and OOM-ing. Two terms most static estimators miss do the work here:
the **logits / cross-entropy chain** (`seq × vocab × ~14 B`; 4.1 GB for
Qwen's 152k vocab at seq 2048, and gradient checkpointing does *not* remove
it) and the **fp32 upcast of embeddings/norms** that
`prepare_model_for_kbit_training` performs under QLoRA. Every coefficient
was fitted against measured `torch.cuda` peaks — see
`docs/rtx4080_baselines.md` — and `canifinetune bench` / `calibrate` can
still ground the estimate on *your* machine.

---

## Install

`canifinetune` runs in two layers:

| Layer | Install | What you get |
| --- | --- | --- |
| Core (estimate / recommend / recipe / report) | `pip install canifinetune` | All CLI commands. No PyTorch required. |
| Training (bench / real fine-tuning) | `pip install canifinetune[train]` | Adds `torch`, `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`. |
| Reporting extras | `pip install canifinetune[report]` | Pandas/tabulate for prettier tables. |
| Development | `pip install canifinetune[dev]` | pytest, ruff, mypy. |

If you use `uv`:

```bash
uv venv
uv pip install -e ".[dev,report]"
# Add training deps when you want to run benchmarks:
uv pip install -e ".[dev,train,report]"
```

PyTorch should generally be installed with the CUDA wheel that matches your driver,
e.g.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

See `docs/troubleshooting.md` for Windows / WSL / bitsandbytes specifics.

---

## Quickstart

```bash
# 1. See what your machine looks like
canifinetune doctor

# 2. Ask if a model fits on your card
canifinetune estimate \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --method qlora \
  --gpu-vram-gb 16 \
  --seq-len 2048 \
  --micro-batch-size 1 \
  --lora-rank 16

# 3. Have it search for a feasible config
canifinetune recommend --model Qwen/Qwen2.5-1.5B-Instruct --gpu-vram-gb 16

# 4. Run a tiny real benchmark (downloads sshleifer/tiny-gpt2, ~5 MB)
canifinetune bench --model sshleifer/tiny-gpt2 --method lora --steps 3

# 5. Generate a ready-to-run training recipe
canifinetune recipe \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --method qlora \
  --seq-len 2048 \
  --output recipes/qwen2.5-1.5b-qlora-4080
```

---

## What's different from `accelerate estimate-memory`?

`accelerate estimate-memory` tells you how much memory **loading** a model takes.
That is not enough to know whether you can **train** it.

This project tries to answer the harder question. It models:

- Model weights, in fp32 / fp16 / bf16 / int8 / NF4 + double-quant —
  including the fact that QLoRA only quantizes the Linear layers while
  embeddings / lm_head / norms are upcast to fp32 by
  `prepare_model_for_kbit_training` (4 GB on an untied 7B!)
- The logits / cross-entropy chain (`seq × batch × vocab × ~14 B`) — the
  single biggest training buffer for modern 128k–152k-vocab models, and
  one gradient checkpointing does not touch
- LoRA / QLoRA trainable parameter count for typical `target_modules`
- Gradients only for trainable parameters
- AdamW vs 8-bit / paged AdamW optimizer states
- Activations as a function of `seq_len`, `batch_size`, `hidden_size`,
  `intermediate_size`, `num_layers`, with and without gradient checkpointing,
  with coefficients fitted to measured peaks on real hardware
- A fragmentation / CUDA / buffer safety margin
- A feasibility decision against your actual GPU
- Concrete degradation suggestions when not feasible

Estimates are **always** marked with an `assumptions` block and a `confidence`
level, because activation memory in particular is hard to predict statically.
Run `canifinetune bench` and `canifinetune calibrate` to ground them in real
measurements on your machine.

---

## RTX 4080 baselines

`docs/rtx4080_baselines.md` contains real measurements collected on a single
RTX 4080 (16 GB). These are not synthetic. If a configuration was not run, the
table says "not run", not a guessed number. The same runs are pinned as
regression fixtures in `tests/test_estimator_accuracy.py`, so the estimator
cannot silently drift away from measured reality.

Highlights (more in the doc):

| model | method | seq_len | estimated | measured peak | tok/sec |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | qlora | 1024 | 5.01 GB | 3.30 GB | 3337 |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 6.07 GB | 4.36 GB | 2483 |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 2048 | 8.44 GB | 7.10 GB | 2327 |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 4096 | 13.19 GB | 13.56 GB | 1662 |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora (no ckpt) | 1024 | 10.77 GB | 9.55 GB | 3003 |
| `Qwen/Qwen2.5-3B-Instruct` | qlora | 1024 | 7.31 GB | 5.54 GB | 1303 |
| `Qwen/Qwen2.5-7B-Instruct` | qlora | 1024 | 12.54 GB | 11.23 GB | 923 |

---

## Repository layout

```
src/canifinetune/        # package code (estimator, bench, recipes, reports, cli)
benchmarks/              # configs/, results/ (JSON), calibration/
docs/                    # design, memory model, troubleshooting
examples/                # end-to-end recipe folders
tests/                   # pytest tests (CPU-only, no large downloads)
scripts/                 # helper scripts for collecting baselines
.github/workflows/       # CI (ruff + pytest on CPU)
```

---

## Roadmap

The current scope is "single consumer GPU, single node, LoRA / QLoRA, causal LM,
Hugging Face stack". Possible directions, none committed:

- DeepSpeed ZeRO and FSDP estimation for multi-GPU setups
- Heuristics for sequence-classification / encoder-decoder training
- Throughput modeling (tokens / sec), not just feasibility
- Auto-tuning of `gradient_accumulation_steps` for a target effective batch size
- A web UI on top of the CLI

Contributions welcome.

---

## Star History

<a href="https://star-history.com/#DaoyuanLi2816/can-i-finetune-this&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DaoyuanLi2816/can-i-finetune-this&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DaoyuanLi2816/can-i-finetune-this&type=Date" />
    <img alt="Star history chart" src="https://api.star-history.com/svg?repos=DaoyuanLi2816/can-i-finetune-this&type=Date" />
  </picture>
</a>

---

## License

MIT. See `LICENSE`.
