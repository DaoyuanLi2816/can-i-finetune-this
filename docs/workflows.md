# Choose a workflow

## Plan before downloading weights

Use the curated catalogue offline, or config/parameter metadata for another
supported HF causal model. Metadata lookup does not load weights.

```console
canifinetune doctor
canifinetune estimate --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --gpu-vram-gb 16 --seq-len 2048 --offline
canifinetune recommend --model Qwen/Qwen2.5-0.5B-Instruct --gpu-vram-gb 16 --offline --top-k 3
```

Keep actual total VRAM in `--gpu-vram-gb`. On a busy device, also pass the current
free budget with `--available-vram-gb`; overhead is still based on actual capacity.
Every `*_gb` JSON value uses GiB. Inspect assumptions before acting on YES/NO.

## Generate and execute a recipe

Install a qualified [training stack](quickstart.md), then create a fresh folder:

```console
canifinetune recipe --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --method qlora --seq-len 256 --max-steps 2 --grad-accum 1 --output my-recipe
python my-recipe/train.py --config my-recipe/config.yaml
python my-recipe/eval_smoke.py --output-dir my-recipe/output --max-new-tokens 8
```

Training downloads weights when they are not cached. Check `output/run.json`:
status must be success, update count must match, effective settings must match
your intended workload, and reload must also succeed. Full saves `model`;
LoRA/QLoRA saves `adapter`. Existing output directories are protected.
Read the [data contract](training.md) before replacing sample records.

## Measure a bounded workload

```console
canifinetune bench --model Qwen/Qwen2.5-0.5B-Instruct --method qlora --seq-len 256 --optimizer adamw_torch --steps 3 --out-dir measurements
canifinetune report --benchmarks measurements --out report.md
canifinetune compare --benchmarks measurements --out compare.md
```

Bench loads weights and uses full-length synthetic labels. It can use more memory
than the short recipe samples. Measurements cover loading through the declared
updates, including first optimizer-state allocation. Compare the process reserved
peak with `estimated_process_reserved_gb`, not the total planning budget including
safety. Three updates do not demonstrate arbitrary long-run stability.

## Fit or contribute evidence deliberately

`calibrate` fits eligible development observations; it is not independent
validation. Compatibility checks include estimator version, configuration profile,
model family and GPU capacity. Community/validation/failed observations are
excluded. See [evidence](evidence.md) before exporting or fitting.
