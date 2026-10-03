# Troubleshooting

## Missing dependencies or CUDA

Core estimate/recipe/demo needs no torch. Actual training needs the qualified
[quickstart stack](quickstart.md). Use a fresh project environment if unrelated
packages cause import failures. CPU uses explicit `--device cpu --base-dtype fp32
--optimizer adamw_torch` and full/LoRA. QLoRA and 8-bit optimizers require CUDA.
`nvidia-smi` working does not establish that the installed torch wheel has CUDA;
check `canifinetune doctor` and `torch.cuda.is_available()`. Do not replace global
CUDA or other project environments to fix this package.

The verified combination is torch 2.6.0/cu124 and bitsandbytes 0.49.2. Use the
provided constraints instead of an unbounded upgrade. Windows console logs with
non-ASCII text should use `$env:PYTHONUTF8="1"` in PowerShell. Linux uses
`export PYTHONUTF8=1`. CI sets UTF-8 explicitly.

## Dataset or configuration rejected

Read the line number and generated `dataset_format.md`. Preserve system/multi-turn
records; unsupported roles or templates fail explicitly. Assistant-only chat needs
native generation masks. Empty replies and no shifted supervised labels fail.
Increase sequence length, shorten input deliberately, or explicitly opt into
`truncation: right`; right truncation still cannot remove all response labels.
Unknown config keys fail. Regenerate old recipes rather than copying legacy
`save_steps`/TRL settings into the strict schema. Output must be empty; choose a
fresh directory. `--force` overwrites generated recipe files only, never a run.

## OOM or a busy display GPU

Reduce micro-batch/sequence length, use attention-only targets, QLoRA or a smaller
model, and re-estimate the changed configuration. Accumulation preserves effective
batch after reducing micro-batch; it does not shrink a single micro-batch. A load
OOM will not be fixed by checkpointing. Inspect `run.json`/bench stage locally.
An OOM receipt has no exact peak and cannot be evaluated as successful.

Keep actual total capacity in `--gpu-vram-gb` and pass currently free capacity
separately with `--available-vram-gb`. Background allocations change; don't compare
device-level usage directly with process allocator peaks. Safety is an additional
planning allowance, not part of observed reserved memory. A short probe cannot
guarantee a long run will fit. Do not terminate unrelated Python processes.

## BF16, attention, Liger

Unsupported BF16 is an explicit error. Choose fp16 for a supported adapter path
or fp32, then re-estimate. Full fp16 weight training is rejected. Use SDPA/eager
explicitly if Flash Attention is unavailable; no loader retry removes attention.
Flash Attention and Linux/Triton Liger have not received 0.4.0 CUDA qualification.
Liger's estimate is a low-confidence stock upper planning proxy. Bench rejects it.

## Offline and gated models

`--offline` uses curated metadata, local directories or cached config; unknown
uncached metadata fails clearly. Training additionally needs cached weights and
tokenizers. Curated metadata does not establish access to gated weights. Obtain
access under the model's terms yourself and authenticate using the normal Hub
workflow. Never put credentials in datasets, YAML, issues or logs for sharing.
Remote model code is disabled; models requiring it are unsupported.

## Reload or save failure

Check the run is successful and contains its model/adapter artifact. Failed saves
remove `.saving`; they do not become successful checkpoints. Adapter evaluation
reloads the same pinned base and applies the saved adapter; it never silently
ignores adapter errors. These are inference artifacts, not a full resume snapshot.
