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

## Results: estimated vs measured

Each row is one `canifinetune bench` invocation (rank 16, `attention` scope,
`paged_adamw_8bit`, `nf4_double_quant` for QLoRA). `est. GB` is the *static*
estimate produced by the same code that ships in this repo — including its
CUDA-overhead and safety-margin headroom — and `measured` is the
`torch.cuda.max_memory_reserved` peak of the real run. The estimate is meant
to sit slightly *above* the measured peak.

| model | method | seq_len | batch | ckpt | est. GB | **measured peak (reserved) GB** | tok/sec | OOM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | qlora | 1024 | 1 | on | 5.01 | **3.30** | 3337 | no |
| `Qwen/Qwen2.5-0.5B-Instruct` | qlora | 2048 | 1 | on | 7.21 | **6.28** | 3445 | no |
| `Qwen/Qwen2.5-0.5B-Instruct` | lora (bf16) | 1024 | 1 | on | 5.19 | **3.79** | 3762 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 512 | 1 | on | 4.86 | **2.91** | 1498 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 1 | on | 6.05 | **4.36** | 2483 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 2048 | 1 | on | 8.42 | **7.10** | 2327 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 4096 | 1 | on | 13.16 | **13.56** | 1662 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 2 | on | 8.42 | **6.88** | 2662 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 1 | **off** | 10.75 | **9.55** | 3003 | no |
| `Qwen/Qwen2.5-3B-Instruct` | qlora | 1024 | 1 | on | 7.26 | **5.54** | 1303 | no |
| `Qwen/Qwen2.5-7B-Instruct` | qlora | 1024 | 1 | on | 12.43 | **11.23** | 923 | no |
| `sshleifer/tiny-gpt2` (smoke) | lora | 128 | 1 | on | 2.16 | **0.14** | 1470 | no |

Comparing the estimator's *real components* (total minus the headroom terms)
against `max_memory_allocated` instead, every non-smoke row lands within
**0.93–1.05×** of the measurement — that band is enforced by
`tests/test_estimator_accuracy.py`, which pins these exact runs as regression
fixtures. (The tiny-gpt2 smoke row is all fixed headroom by construction.)

Worth reading off the table directly:

- **seq_len slope is real and linear**: 2.91 → 4.36 → 7.10 → 13.56 GB for
  512 → 1024 → 2048 → 4096. Almost all of that increment is the
  logits/cross-entropy chain (vocab 151,936), which gradient checkpointing
  cannot remove.
- **batch 2 × seq 1024 ≈ batch 1 × seq 2048** (6.88 vs 7.10 GB) — dynamic
  memory scales with `batch × seq`.
- **Checkpointing off costs +5.2 GB** at 1.5B/seq 1024 (4.36 → 9.55 GB) and
  buys ~20% throughput. On a 16 GB card it is usually the wrong trade above
  seq 1024.
- **7B QLoRA fits a 16 GB card at seq 1024** (11.23 GB measured) but does
  *not* fit a 12 GB card — and the estimator now says exactly that (12.43
  estimated → `no` at 12 GB, `yes` at 16 GB).

## Calibration

Running `canifinetune calibrate --benchmarks benchmarks/results` on the
results above fits component-aware corrections from 11 samples (smoke-scale
runs are excluded automatically). The v2 calibration leaves static weights
and the policy safety margin untouched, fits dynamic memory against
`max_memory_allocated` (**1.04×** here), and separately fits the allocator gap
between allocated and reserved memory (**0.285×**). Calibration is only
applied to compatible model families, methods, and GPU-memory sizes.

The static estimator alone was 0.5–3.9× off (both directions) before the
logits-chain and fp32-upcast terms were added in 0.2.0; calibration existed
to paper over that. It no longer has to — it is now a per-machine fine-trim.

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
