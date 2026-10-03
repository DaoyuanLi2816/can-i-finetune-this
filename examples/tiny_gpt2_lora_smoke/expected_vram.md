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
| quantization | none |
| seq_len | 128 |
| micro_batch_size | 1 |
| LoRA rank | 8 |
| target modules | c_attn, c_proj |
| optimizer | adamw_torch |
| gradient_checkpointing | False |
| attention impl | eager |
| target GPU | 16.0 GiB |

## Breakdown

| Component | GiB |
| --- | --- |
| static weights | 0.000 |
| quantization overhead | 0.000 |
| trainable parameters | 0.0 million params |
| gradients | 0.000 |
| optimizer states | 0.000 |
| activations | 0.001 |
| logits / loss chain | 0.096 |
| CUDA / fragmentation overhead | 1.280 |
| safety margin | 0.800 |
| **total estimated** | **2.178** |

Feasibility on a 16.0 GiB GPU: **yes**
(confidence: medium).

## How to verify on your machine

Run the generated `train.py` with the same YAML configuration. Its `run.json`
records actual software/configuration and allocator peaks. For a synthetic bench,
copy **all** relevant fields into BenchConfig (including alpha/dropout, revision,
dtype, optimizer, quantization, targets, checkpointing and attention). Short sample
data and full-length synthetic tokens are different workloads.

Compare process reserved peak with total prediction **minus safety_margin_gb**;
keep the safety allowance separate for feasibility. This is a planning heuristic,
not a confidence interval. Liger uses a stock upper planning proxy, not a measured
fused-loss estimate. See the package validation report for its bounded scope.
