# RTX 4080 baselines

These are **real benchmark results** collected on a single RTX 4080 (16 GB).
No numbers in this file are synthetic. If a configuration is missing, it was
not run — we do not interpolate.

## Hardware / software

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (16,376 MiB) |
| Compute capability | 8.9 (Ada Lovelace) |
| Driver / CUDA driver version | 595.97 / 13.2 |
| PyTorch | 2.6.0+cu124 |
| transformers / peft / bitsandbytes / trl | 5.8.1 / 0.19.1 / 0.49.2 / 1.4.0 |
| OS | Windows 11 |
| Notes | Display was attached to the same GPU; ~1 GB of VRAM was held by the desktop / browser at the start of each run. |

## Smoke + 0.5B / 1.5B / 3B QLoRA results

Each row is one `canifinetune bench` invocation. Both `estimated GB` and
`measured peak (reserved) GB` are reported so the gap is visible — that is
exactly the gap that `canifinetune calibrate` consumes.

| model | method | seq_len | batch | rank | quant | ckpt | optimizer | est. GB | **measured peak (reserved) GB** | tok/sec | step time (s) | OOM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sshleifer/tiny-gpt2` | lora | 128 | 1 | 8 | bf16 | off | adamw_torch | 2.08 | **0.12** | 1735 | 0.07 | no |
| `Qwen/Qwen2.5-0.5B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 2.41 | **3.30** | 1995 | 0.51 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 3.00 | **4.36** | 1352 | 0.76 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 2048 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 3.16 | **7.10** | 1470 | 1.39 | no |
| `Qwen/Qwen2.5-3B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 3.86 | **5.54** | 1158 | 0.88 | no |

The 1.5B model fits comfortably at seq=2048 with QLoRA on a 16 GB consumer
card — even with a desktop / browser eating ~1 GB at runtime. The 3B model
likewise fits at seq=1024.

## Calibration

Running `canifinetune calibrate --benchmarks benchmarks/results` on the
results above produces a single multiplicative correction. The mean
measured/estimated ratio across the five runs is **1.31×**. That is the
factor `canifinetune estimate --use-calibration` applies to the
weights / activations / CUDA-overhead components.

The biggest residual gap is at `seq_len=2048`: 7.10 GB measured vs.
~3.16 GB statically estimated (ratio 2.25×). This is in line with what the
[memory model docs](memory_model.md) warn about — the static activation
formula under-predicts when bitsandbytes 4-bit unpacking buffers
materialize bf16 copies of weight blocks during each forward step.

The `canifinetune` philosophy is to surface this gap, not paper over it:
the static estimate is *fast and conservative*, the benchmark is *slow and
real*, and calibration closes the loop on the operator's machine.

## Reproducing

```bash
# 1. Install with training extras and matching CUDA torch wheel:
uv pip install -e ".[train]"
uv pip install torch>=2.6 --index-url https://download.pytorch.org/whl/cu124

# 2. Run all baselines:
bash scripts/run_4080_baselines.sh

# 3. Re-fit calibration and rebuild this file:
canifinetune calibrate --benchmarks benchmarks/results --out benchmarks/calibration/local_4080.json
canifinetune compare  --benchmarks benchmarks/results --out benchmarks/results/_compare.md
canifinetune report   --benchmarks benchmarks/results --out benchmarks/results/_report.md
```

All result JSONs live in `benchmarks/results/`. Each is small (a few KB) and
is committed so contributors without a 4080 can still see what the table
above is built from.
