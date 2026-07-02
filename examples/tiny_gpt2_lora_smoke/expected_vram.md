# Expected VRAM for this recipe

Computed by `canifinetune` at recipe-generation time using the static estimator.
Numbers are **predictions**, not measurements. Real VRAM may differ; run
`canifinetune bench` to calibrate on your card.

## Inputs

| Field | Value |
| --- | --- |
| model | `sshleifer/tiny-gpt2` (gpt2, ~0.00 B params, source: known) |
| method | lora |
| base dtype | fp32 |
| quantization | bf16 |
| seq_len | 128 |
| micro_batch_size | 1 |
| LoRA rank | 8 |
| target modules | c_attn, c_proj |
| optimizer | adamw_torch |
| gradient_checkpointing | False |
| attention impl | eager |
| target GPU | 16.0 GB |

## Breakdown

| Component | GB |
| --- | --- |
| static weights | 0.000 |
| quantization overhead | 0.000 |
| trainable parameters | 0.0 MB |
| gradients | 0.000 |
| optimizer states | 0.000 |
| activations | 0.001 |
| logits / loss chain | 0.084 |
| CUDA / fragmentation overhead | 1.280 |
| safety margin | 0.800 |
| **total estimated** | **2.165** |

Feasibility on a 16.0 GB GPU: **yes**
(confidence: medium).

## How to verify on your machine

```bash
canifinetune bench \
  --model "sshleifer/tiny-gpt2" \
  --method lora \
  --seq-len 128 \
  --micro-batch-size 1 \
  --lora-rank 8 \
  --steps 3
```
