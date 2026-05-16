#!/usr/bin/env bash
# canifinetune RTX 4080 baseline runner.
#
# This script runs every benchmark that we ship as a "RTX 4080 baseline".
# It is intended to be run on the maintainer's local 4080 box; CI does not
# run this (no GPU). Adjust BENCH_OUT to a path outside any sync'd folder.

set -euo pipefail
BENCH_OUT="${BENCH_OUT:-benchmarks/results}"
mkdir -p "$BENCH_OUT"

# A. Tiny model smoke (no quantization, no bitsandbytes required).
canifinetune bench \
  --model sshleifer/tiny-gpt2 \
  --method lora \
  --seq-len 128 \
  --micro-batch-size 1 \
  --lora-rank 8 \
  --steps 3 \
  --no-gradient-checkpointing \
  --base-dtype fp32 \
  --optimizer adamw_torch \
  --quantization bf16 \
  --attn eager \
  --out-dir "$BENCH_OUT"

# B. 0.5B QLoRA baseline.
canifinetune bench \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --method qlora \
  --seq-len 1024 \
  --micro-batch-size 1 \
  --lora-rank 16 \
  --steps 3 \
  --out-dir "$BENCH_OUT"

# C. 1.5B QLoRA baseline (the headline configuration).
canifinetune bench \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --method qlora \
  --seq-len 1024 \
  --micro-batch-size 1 \
  --lora-rank 16 \
  --steps 3 \
  --out-dir "$BENCH_OUT"

canifinetune bench \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --method qlora \
  --seq-len 2048 \
  --micro-batch-size 1 \
  --lora-rank 16 \
  --steps 3 \
  --out-dir "$BENCH_OUT"

# D. 3B QLoRA exploratory baseline (may OOM at long seq).
canifinetune bench \
  --model Qwen/Qwen2.5-3B-Instruct \
  --method qlora \
  --seq-len 1024 \
  --micro-batch-size 1 \
  --lora-rank 16 \
  --steps 2 \
  --out-dir "$BENCH_OUT" || echo "[warn] 3B run failed — check the JSON for the OOM stage."

# After all runs:
canifinetune calibrate --benchmarks "$BENCH_OUT"
canifinetune report --benchmarks "$BENCH_OUT" --out "$BENCH_OUT/_report.md"
canifinetune compare --benchmarks "$BENCH_OUT" --out "$BENCH_OUT/_compare.md"
