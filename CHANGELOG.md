# Changelog

## 0.4.0

- Replace divergent TRL/Trainer paths with one package-backed runtime and strict
  requested/effective configuration; preserve explicit dtype and attention.
- Preserve native system/multi-turn chat, keep all-token loss by default, add
  verified assistant masks, reject silent truncation and check causal labels.
- Execute real updates, stage model/adapter saves, protect output directories and
  verify reload; report dependency/configuration/access/OOM failures explicitly.
- Add a torch-free loopback interactive estimator and opt-in redacted evidence
  export/validation with manual community review.
- Separate GiB/process peaks/safety/free-device budgets, fix native optimizer and
  full-gradient accounting, constrain calibration to compatible development data
  and report fresh same-cohort old/new validation including negative results.
- Qualify minimum/recommended CPU stacks and native Windows RTX 4080 CUDA paths;
  CI checks actual training, installed wheels and rendered Python.
- Publish the same hashed, qualified wheel/sdist through the existing Trusted
  Publisher; verify official PyPI hashes and fresh public installs.

Migration: regenerate recipes, install their versioned package requirements and
remove unknown legacy YAML keys. Training no longer needs TRL/datasets. Default
truncation now errors; opt into right truncation deliberately. Regenerate old
calibration caches. Confidence remains heuristic; Liger remains experimental.

## 0.3.0

- Use Hugging Face safetensors metadata when available for exact parameter counts.
- Account for all experts in mixture-of-experts model weights and active experts in
  activation estimates.
- Keep quantization metadata separate from packed weight memory.
- Calibrate dynamic memory and allocator overhead without rescaling static weights.
- Fail clearly when requested QLoRA or 8-bit optimizer dependencies are unavailable.
- Refuse to overwrite non-empty recipe directories unless `--force` is passed.
- Add optional Liger kernels to generated recipes with `--liger`.
- Add broader tests, static type checks, formatting checks, and release-tag validation.

## 0.2.0

- Add calibrated memory estimates, benchmark reporting, and generated training recipes.
- Improve activation, attention, and logits memory accounting.
- Add RTX 4080 benchmark baselines.
