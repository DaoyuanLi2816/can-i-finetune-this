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
| target GPU | 16.0 GiB |

## Breakdown

| Component | GiB |
| --- | --- |
| static weights | 1.496 |
| quantization overhead | 0.072 |
| trainable parameters | 4.4 million params |
| gradients | 0.016 |
| optimizer states | 0.010 |
| activations | 0.344 |
| logits / loss chain | 2.029 |
| CUDA / fragmentation overhead | 1.280 |
| safety margin | 0.800 |
| **total estimated** | **6.047** |

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
