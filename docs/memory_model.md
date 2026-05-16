# Memory model

`canifinetune` decomposes the GPU memory footprint of a training step into the
following components. All numbers are in bytes; the CLI rounds to GB on output.

```
Total estimated VRAM
  =  weights (base model, possibly quantized)
   + gradients          (trainable params only)
   + optimizer states   (trainable params only)
   + activations        (depends on seq_len, batch, hidden, layers, checkpointing)
   + CUDA / fragmentation overhead
   + safety margin
```

## 1. Weights

```
weights_bytes = num_params * bytes_per_param
```

`bytes_per_param` depends on the storage dtype:

| dtype          | bytes/param |
| -------------- | ----------- |
| fp32           | 4.0         |
| fp16 / bf16    | 2.0         |
| int8           | 1.0         |
| NF4            | 0.5         |

For 4-bit / 8-bit schemes, we also add a *quantization overhead* per
parameter for absmax scalars and lookup tables:

| scheme               | overhead bytes/param | source              |
| -------------------- | -------------------- | ------------------- |
| int8 (`Linear8bitLt`)| ~0.15                | bitsandbytes        |
| NF4 (blocksize 64)   | ~0.0625              | QLoRA paper, blocksize 64, fp32 absmax |
| NF4 + double-quant   | ~0.0125              | bitsandbytes        |

So **QLoRA's effective bytes/param is ~0.51**, not exactly 0.5.

## 2. Trainable parameters (LoRA / QLoRA only)

A LoRA adapter on a `Linear[in_dim, out_dim]` layer adds:

```
adapter_params = rank * (in_dim + out_dim)
```

`canifinetune` walks all selected `target_modules` per transformer layer. For
GQA models, K/V projections are sized using `num_key_value_heads`, not
`num_attention_heads`.

The default `target_modules` per family mirror PEFT's defaults:

| family        | attention scope                          | all_linear scope                              |
| ------------- | ---------------------------------------- | --------------------------------------------- |
| llama / qwen2 | q_proj, k_proj, v_proj, o_proj           | + gate_proj, up_proj, down_proj                |
| mistral       | same as llama                            | same as llama                                  |
| gemma         | same as llama                            | same as llama                                  |
| phi           | q_proj, k_proj, v_proj, dense            | + fc1, fc2                                     |
| gpt2          | c_attn, c_proj                           | + c_fc                                         |

## 3. Gradients

For LoRA / QLoRA, gradients exist *only* for adapter parameters (the base
model is frozen). For full fine-tuning, gradients cover all parameters.

```
gradients_bytes = trainable_params * dtype_bytes(grad_dtype)
```

We default `grad_dtype = bf16` (HF + accelerate default with bf16 mixed
precision).

## 4. Optimizer states

Per-parameter bytes for the most common optimizers:

| optimizer                  | bytes/param |
| -------------------------- | ----------- |
| adamw_torch (fp32 m+v+master)| 12.0      |
| paged_adamw_32bit          | 12.0        |
| adamw_8bit / paged_adamw_8bit | 2.5      |
| sgd                        | 4.0         |
| sgd + momentum             | 8.0         |
| lion_8bit                  | 2.0         |
| adafactor                  | 4.0         |

This is multiplied by `trainable_params`, *not* total parameters. For LoRA
/ QLoRA, the optimizer footprint is consequently tiny (a few MB).

## 5. Activations

This is the noisiest term. We use the per-layer formula from
*"Reducing Activation Recomputation in Large Transformer Models"*
(Korthikanti et al., 2022):

```
per_layer_activations = s * b * h * (34 + 5 * a * s / h)
total = num_layers * per_layer_activations * dtype_bytes
```

Where `s = seq_len`, `b = batch`, `h = hidden`, `a = num_heads`.

Two adjustments:

- **Fused attention** (SDPA / flash-attn) does not materialize the
  `(b, a, s, s)` softmax matrix, so we drop the `5*a*s/h` term.
- **Gradient checkpointing** approximates full activation recomputation: only
  the block inputs are kept, so per-layer activations collapse to ~`2*s*b*h`.

Empirically these formulas tend to over-estimate slightly. That's by design:
the feasibility verdict is conservative. Use `canifinetune calibrate` to
correct downward (or upward) using your real measurements.

## 6. CUDA / fragmentation overhead

PyTorch's caching allocator, CUDA context, cuBLAS / cuDNN workspaces, and
fragmentation eat a non-trivial fraction of VRAM. We model this as a flat
fraction (default 8%) of the GPU's total VRAM. Calibration can tune this.

## 7. Safety margin

A small fraction (default 5%) of the GPU's total VRAM is held back so the
estimator never recommends running at the absolute brink. Display compositors
and Chrome / Edge processes routinely take 0.5–2 GB on consumer cards.

## Feasibility classification

```
ratio = total_estimated / available_vram
feasible == "yes"      if ratio <= 0.85
feasible == "marginal" if 0.85 < ratio <= 0.97
feasible == "no"       otherwise
```

The 0.85 threshold is empirical: with that headroom, every 1B / 1.5B / 3B
QLoRA configuration we benchmark on an RTX 4080 finishes the smoke run
without OOM (see `docs/rtx4080_baselines.md`).

## When the estimator is wrong

Common reasons for the static estimate diverging from reality:

- **Very long seq_len (≥ 8192)**: small differences in the activation formula
  compound, and many models use position-dependent KV caches that aren't part
  of the standard activation accounting.
- **No flash-attn**: if the model falls back to eager attention, the `s²`
  attention matrix appears in activations and the estimate is too low.
- **Optimizer fusion**: some optimizers (e.g. `adamw_torch_fused`) hold
  workspace buffers we don't model precisely.
- **Loaded display GPU**: the OS / desktop / browser take VRAM at runtime.
  Use `canifinetune doctor` to see free VRAM, and pass that as `--gpu-vram-gb`
  if you want a tighter feasibility decision.

When in doubt: run `canifinetune bench` and `canifinetune calibrate`.
