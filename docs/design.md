# Design notes

`canifinetune` is structured as four layers:

```
+-------------------+
|       cli         |  typer entry points, rich tables, JSON output
+-------------------+
|  estimator (math) |  pure formulas → memory breakdown → feasibility decision
+-------------------+
|   bench (torch)   |  loads tiny / small model, runs N steps, captures VRAM
+-------------------+
|  recipes / reports|  jinja2 templates + markdown/html rendering
+-------------------+
```

## Why a separate `estimator` layer

The estimator is intentionally pure: every function in
`src/canifinetune/estimator/formulas.py` is a function on plain numbers. That
makes the heuristics testable in CI without any GPU, without torch, and
without network access. The CLI exposes the estimator behind a stable
`EstimateRequest` / `Estimate` pair of pydantic models.

## Why bench is lazy

`canifinetune.bench.runner` imports `torch` at function-call time, not at
module-import time. That lets `import canifinetune.bench` succeed on a CPU-only
machine; only `run_bench()` requires the training stack. The CLI's `bench`
subcommand follows the same rule.

## Calibration

The static estimator is a *model*; the bench runner produces *measurements*.
Calibration is the function that closes the loop:

```
calibration = fit(measurements → estimates)
estimate(req, calibration=...) = adjust(base_estimate, calibration)
```

For now this is a single multiplicative correction on the activations term
(the noisiest one). With enough samples, this can be split per (GPU, model
family, seq_len) bucket.

## Recipes

Recipe generation is template-based (jinja2). The templates are intentionally
plain: every recipe is a small, single-file `train.py` that uses Hugging Face
Transformers + PEFT + (optionally) bitsandbytes + TRL. No accelerate config,
no DeepSpeed, no Trainer subclasses — easier to read, easier to debug.

## Extension points

- **New model families**: add an entry to `KNOWN_MODELS` in `estimator/model_metadata.py`,
  and a `target_modules` block in `estimator/formulas.py` if PEFT's defaults
  don't match.
- **New optimizers**: add a row to `OPTIMIZER_BYTES_PER_PARAM` in `estimator/formulas.py`.
- **New quantization schemes**: add to `QUANT_OVERHEAD_BYTES_PER_PARAM` in `utils/units.py`,
  then handle the storage-dtype mapping in `_quant_storage_dtype` (`estimator/formulas.py`).
- **New report formats**: add a renderer next to `reports/markdown.py` and expose
  it via `reports/__init__.py`.
