# Expected VRAM for this recipe

Computed by `canifinetune` at recipe-generation time using the static estimator.
Numbers are **predictions**, not measurements. Real VRAM may differ; run
`canifinetune bench` to calibrate on your card.

## Inputs

| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-1.5B-Instruct` (qwen2, ~1.54 B params, source: known) |
| method | qlora |
| base dtype | bf16 |
| quantization | nf4_double_quant |
| seq_len | 1024 |
| micro_batch_size | 1 |
| LoRA rank | 16 |
| target modules | q_proj, k_proj, v_proj, o_proj |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention impl | sdpa |
| target GPU | 16.0 GB |

## Breakdown

| Component | GB |
| --- | --- |
| static weights | 0.737 |
| quantization overhead | 0.018 |
| trainable parameters | 4.4 MB |
| gradients | 0.008 |
| optimizer states | 0.010 |
| activations | 0.164 |
| CUDA / fragmentation overhead | 1.280 |
| safety margin | 0.800 |
| **total estimated** | **2.999** |

Feasibility on a 16.0 GB GPU: **yes**
(confidence: medium).

## How to verify on your machine

```bash
canifinetune bench \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --method qlora \
  --seq-len 1024 \
  --micro-batch-size 1 \
  --lora-rank 16 \
  --steps 3
```
