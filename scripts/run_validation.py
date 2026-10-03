"""Execute one predeclared prospective validation case in an isolated process."""

import argparse
from pathlib import Path

from canifinetune.bench.runner import BenchConfig, run_bench
from canifinetune.estimator.memory import EstimateRequest, estimate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "qlora"], required=True)
    parser.add_argument("--seq-len", choices=[256, 512], type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    free, total = torch.cuda.mem_get_info()
    config = BenchConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        method=args.method,
        seq_len=args.seq_len,
        micro_batch_size=1,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0,
        optimizer="adamw_torch",
        base_dtype="bf16",
        gradient_checkpointing=True,
        attention_implementation="sdpa",
        steps=3,
    )
    prediction = estimate(
        EstimateRequest(
            **config.training_fields(), gpu_vram_gb=total / 2**30, available_vram_gb=free / 2**30
        )
    )
    if free / 2**30 < max(4, prediction.memory.total_estimated_gb):
        raise SystemExit("not run: insufficient free memory for the predeclared safe experiment")
    result = run_bench(config)
    result.provenance = {
        "source": "maintainer-local",
        "review_status": "verified" if result.success else "unreviewed",
        "cohort": "validation",
    }
    if args.out.exists():
        raise SystemExit("refusing to overwrite a historical observation")
    result.save(args.out)
    print(result.status, result.measured, result.notes)
    raise SystemExit(0 if result.success else 1)
