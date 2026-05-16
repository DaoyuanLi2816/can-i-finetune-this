# can-i-finetune-this

**Estimate, benchmark, and generate fine-tuning recipes for LLMs on consumer GPUs.**

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

- Model weights, in fp32 / fp16 / bf16 / int8 / NF4 + double-quant
- LoRA / QLoRA trainable parameter count for typical `target_modules`
- Gradients only for trainable parameters
- AdamW vs 8-bit / paged AdamW optimizer states
- Activations as a function of `seq_len`, `batch_size`, `hidden_size`, `num_layers`,
  with and without gradient checkpointing
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
table says "not run", not a guessed number.

---

## Repository layout

```
src/canifinetune/        # package code (estimator, bench, recipes, reports, cli)
benchmarks/              # configs/, results/ (JSON), calibration/
docs/                    # design, memory model, troubleshooting, IP safety
examples/                # end-to-end recipe folders
tests/                   # pytest tests (CPU-only, no large downloads)
scripts/                 # helper scripts for collecting baselines
.github/workflows/       # CI (ruff + pytest on CPU)
```

---

## IP / data safety

This project is **general LLM fine-tuning infrastructure**. It is intentionally
not a recommender system. See `docs/ip_safety.md` for details. All bundled
examples use public models and synthetic or public datasets.

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

## License

MIT. See `LICENSE`.
