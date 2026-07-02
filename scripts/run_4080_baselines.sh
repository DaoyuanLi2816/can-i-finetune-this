#!/usr/bin/env bash
# canifinetune RTX 4080 baseline runner.
#
# This script runs every benchmark that we ship as a "RTX 4080 baseline"
# (see docs/rtx4080_baselines.md). It is intended to be run on the
# maintainer's local 4080 box; CI does not run this (no GPU). Adjust
# BENCH_OUT to a path outside any sync'd folder if needed.

set -euo pipefail
BENCH_OUT="${BENCH_OUT:-benchmarks/results}"
mkdir -p "$BENCH_OUT"

bench() {
  canifinetune bench --out-dir "$BENCH_OUT" "$@"
}

# A. Tiny model smoke (5 MB download; validates the plumbing).
bench --model sshleifer/tiny-gpt2 --method lora --seq-len 128 --lora-rank 8 --steps 3

# B. 0.5B: QLoRA at two sequence lengths + a bf16 LoRA point.
bench --model Qwen/Qwen2.5-0.5B-Instruct --method qlora --seq-len 1024 --lora-rank 16 --steps 3
bench --model Qwen/Qwen2.5-0.5B-Instruct --method qlora --seq-len 2048 --lora-rank 16 --steps 3
bench --model Qwen/Qwen2.5-0.5B-Instruct --method lora  --seq-len 1024 --lora-rank 16 --steps 3

# C. 1.5B QLoRA: the headline configuration, swept over seq_len,
#    plus batch-2 and checkpointing-off probes for the memory model.
bench --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --seq-len 512  --lora-rank 16 --steps 3
bench --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --seq-len 1024 --lora-rank 16 --steps 3
bench --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --seq-len 2048 --lora-rank 16 --steps 3
bench --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --seq-len 4096 --lora-rank 16 --steps 2
bench --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --seq-len 1024 --micro-batch-size 2 --lora-rank 16 --steps 2
bench --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --seq-len 1024 --lora-rank 16 --steps 3 --no-gradient-checkpointing

# D. 3B and 7B QLoRA (7B is the 16 GB boundary case: ~11.2 GB measured).
bench --model Qwen/Qwen2.5-3B-Instruct --method qlora --seq-len 1024 --lora-rank 16 --steps 2 \
  || echo "[warn] 3B run failed — check the JSON for the OOM stage."
bench --model Qwen/Qwen2.5-7B-Instruct --method qlora --seq-len 1024 --lora-rank 16 --steps 2 \
  || echo "[warn] 7B run failed — check the JSON for the OOM stage."

# After all runs:
canifinetune calibrate --benchmarks "$BENCH_OUT"
canifinetune report --benchmarks "$BENCH_OUT" --out "$BENCH_OUT/_report.md"
canifinetune report --benchmarks "$BENCH_OUT" --out "$BENCH_OUT/_report.html" --html
canifinetune compare --benchmarks "$BENCH_OUT" --out "$BENCH_OUT/_compare.md"
