# 0.4.0 prospective local validation

The cohort was fixed before observing results: Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775`, LoRA / NF4 double-quant QLoRA,
sequence 256 / 512, batch 1, rank 8, alpha 16, dropout 0, attention targets,
native AdamW, checkpointing on, bf16, SDPA, stock all-token loss, 3 optimizer
updates. Each case ran in an isolated process. No coefficients were fitted to it.

Hardware: one native Windows RTX 4080, total 15.992 GiB, driver 596.49,
compute capability 8.9. Python 3.12.13, torch 2.6.0+cu124, Transformers 5.8.1,
PEFT 0.19.1, Accelerate 1.13.0, bitsandbytes 0.49.2. Free memory is recorded
per run from the CUDA-driver API; nvidia-smi was not captured for each of these
pre-release observations. Their feasibility counts use that original driver-view
budget. Final release discovery additionally records nvidia-smi and conservatively
uses the lower reading when there is an unambiguous single-GPU match. Historical
fields are not rewritten to apply that later policy. Budget: one process, at most 1.5 GB new weights (988,097,824-byte model),
three updates per case. The compute-precision fix was enabled and actual
configuration was checked after updates. Source package fingerprints identify
these pre-release observations; exact final-wheel qualification is a separate
release receipt. Subsequent UI/error-message changes are not new measurements.

## Same observations, old and new estimator

Published 0.3.0 was installed from official PyPI in an isolated torch-free
environment. Both versions use the same hardware total and architecture. Its
supported configuration fields were copied explicitly: precision, optimizer,
quantization, target scope, batch and sequence match the observations. It has no
alpha/dropout/loss-backend/effective-runtime fields; here dropout is zero and
alpha does not change static adapter size. This compares its **estimator** on
new-runtime observations, not an old-runtime performance benchmark.

Values below are GiB. Observation is the process's torch **reserved** peak,
covering load, first optimizer-state allocation and three updates. Process proxy
= total planning budget **minus safety allowance**, counted once. This remains
a planning proxy: its CUDA/context allowance can include allocations outside the
allocator, one reason for conservative error. Device free/total is a different
scope. Allocated peaks and stage snapshots remain in the raw JSON.

| Method | Sequence | Reserved observed | 0.3 proxy | 0.4 proxy | 0.3 error | 0.4 error |
| --- | --- | --- | --- | --- | --- | --- |
| lora | 256 | 1.6875 | 2.7616 | 2.7616 | +63.7% | +63.7% |
| lora | 512 | 2.3750 | 3.3036 | 3.3036 | +39.1% | +39.1% |
| qlora | 256 | 1.6660 | 2.5477 | 2.5477 | +52.9% | +52.9% |
| qlora | 512 | 2.3984 | 3.1001 | 3.1001 | +29.3% | +29.3% |

Predictions are unchanged for these configurations: **no accuracy improvement is
demonstrated**. All four estimates are conservative. MAPE is **46.2%**, median
signed error **+46.0%**, range **+29.3% to +63.7%**; 0 underestimates and 4
overestimates. Full-gradient/fused-optimizer corrections affect other settings;
this cohort does not independently establish their estimation accuracy.

Feasibility uses the same recorded free budget for both versions (0.3 classified
post hoc from its total estimate, retaining actual total for overhead). All four
predicted yes and succeeded: false-feasible **0/4** under this exact short workload.
There is no predicted-no sample: false-infeasible is **undefined**, not 0%.
Marginal/unknown, incomplete/error/forward-only, historical fitting and unreviewed
community observations are excluded from those binary rates. Four points on one
GPU/model are descriptive counts, not statistical OOM risk, cross-hardware
validation or a long-training guarantee. Synthetic labels do not measure quality.

## Execution evidence and negative results

Minimum and recommended stacks each passed 14 real offline CPU integration tests:
full/LoRA updates, native BPE multi-turn masks, shared EOS/PAD, save/reload, output
protection, malformed/truncated data and save failure. A separate real 0.5B QLoRA
recipe exercised paged AdamW8bit with two updates. Exact installed candidate/public
wheel qualification outside the checkout covers local random tiny CPU full/LoRA
and CUDA LoRA/QLoRA with save/reload. Release receipts retain configuration,
software, update counts and peaks. Tiny smoke proves a pipeline, not capability.

A tiny recipe at sequence 256 failed explicitly because a row encoded 264 tokens;
declared sequence 512 completed without dropping rows. At baseline, TRL import
failed with a GBK UnicodeDecodeError and entered the legacy fallback; that does
not prove every modern TRL installation fails. An extra repeat was interrupted
when device free memory fell to about 0.25 GiB. It has no successful observation
or exact peak and does not enter these four denominators. No unrelated GPU
process was stopped. Earlier local repeats remain local qualification evidence,
not additional independent samples. Final release and fresh official PyPI
qualification, including pinned 0.5B QLoRA save/reload, are documented separately
in [release verification](release-verification-0.4.0.md); they are not extra
observations in this four-point accuracy cohort.

No WSL GPU, other GPU, Flash Attention, Liger, int8/fp4 accuracy, long-training
stability or useful fine-tuned quality is qualified. Historical development raw
files remain unchanged and separate. Community measurements need review.

[Raw records and old/new summaries](https://github.com/DaoyuanLi2816/can-i-finetune-this/tree/v0.4.0/benchmarks/validation-0.4.0).
