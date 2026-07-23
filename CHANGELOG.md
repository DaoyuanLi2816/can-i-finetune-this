# Changelog

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
