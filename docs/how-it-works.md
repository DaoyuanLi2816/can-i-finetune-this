# From plan to measured execution

<picture>
  <source media="(max-width: 760px)" srcset="../assets/architecture-mobile.svg">
  <img src="../assets/architecture.svg" alt="Shared configuration connects torch-free metadata planning with installed recipe and benchmark execution">
</picture>

## A common configuration contract

`TrainingConfig` gives dtype, attention, quantization, LoRA targets, optimizer,
checkpointing, backend and supervision the same meaning in estimation,
recommendation, benchmarking and recipes. Unknown fields fail validation.
CPU requires explicit fp32. Unsupported BF16 does not silently become FP16;
loader TypeErrors do not discard settings. Declared pre-quantized checkpoints
are rejected before weights because their config could override requested loading.

## Plan with metadata

Model information comes from the catalogue, local/cached config or HF metadata.
The core does not load weights or import torch. The estimator separates weights,
gradients, optimizer state, activations, logits/loss and device overhead from
safety. Catalogue entries describe architecture; a training receipt pins the
actual resolved revision. The local demo calls this same Python core.

## Execute the installed package

Recipes contain small entry scripts and a readable YAML configuration, with
an exact package requirement. One Transformers/PEFT runtime handles native
tokenization, labels, updates, staged saving and evaluation reload. System and
alternating multi-turn history remain intact. Default loss remains all-token;
assistant-only chat requires a tokenizer template with verified generation masks.

The benchmark uses the same loader and configuration meanings. It covers load,
first state allocation and all requested optimizer updates. Records distinguish
success, OOM and other errors; missing peaks are absent rather than zero.

## Make the evidence useful

Execution records keep requested/effective settings, software/source identities,
hardware/memory readings, measurement scope and outcome. Reports compare matched
workloads. Calibration fits compatible development data; independent observations
and community uploads stay separate. Export is local, opt-in and redacted.

[Memory formulas](memory_model.md) · [Training contract](training.md) ·
[Evidence rules](evidence.md) · [Maintainer design notes](design.md)
