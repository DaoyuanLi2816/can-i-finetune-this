# Memory model

`canifinetune` decomposes the GPU memory footprint of a training step into the
following components. All numbers are in bytes; the CLI rounds to GB on output.

```
Total estimated VRAM
  =  weights (base model, possibly quantized; + fp32 LoRA adapters)
   + quantization overhead   (absmax metadata + dequant workspace, QLoRA only)
   + gradients               (trainable params only)
   + optimizer states        (trainable params only)
   + activations             (seq_len, batch, hidden, ffn, layers, checkpointing)
   + logits / loss chain     (seq_len, batch, vocab — NOT reduced by checkpointing)
   + CUDA / fragmentation overhead
   + safety margin
```

The coefficients below were calibrated against `torch.cuda.max_memory_allocated`
traces on an RTX 4080 (torch 2.6, transformers 5.8, peft 0.19, bitsandbytes
0.49) across Qwen2.5-0.5B/1.5B/3B/7B, seq 512–4096, batch 1–2, LoRA + QLoRA,
checkpointing on/off. On those runs the sum of the "real" components lands
within about ±10% of the measured peak (see `tests/test_estimator_accuracy.py`
and `docs/rtx4080_baselines.md`).

## 1. Weights

For fp32 / fp16 / bf16 runs:

```
weights_bytes = num_params * bytes_per_param     # 4.0 / 2.0 / 2.0
```

For QLoRA, **only the transformer Linear layers are quantized** by
bitsandbytes. The input embedding, the `lm_head` (a second full matrix when
`tie_word_embeddings=false`), and the norms stay in full precision — and
PEFT's `prepare_model_for_kbit_training` **upcasts them to fp32**:

```
linear_params  = num_params - embeddings - norms
weights_bytes  = linear_params * (0.5 + quant_overhead)
               + (embeddings + norms) * 4.0        # fp32 after kbit-prepare
```

Measured: Qwen2.5-1.5B (tied, vocab 151936) loads at **1.51 GiB**
(0.61 GiB packed 4-bit + 0.87 GiB fp32 embedding), not the 0.79 GiB an
"all params × 0.5 B" model predicts. Qwen2.5-7B (untied) loads at
**7.2 GiB** — the fp32 embedding + lm_head alone are 4.06 GiB.

Quantization metadata (absmax scalars, lookup tables):

| scheme               | overhead bytes/param | source              |
| -------------------- | -------------------- | ------------------- |
| int8 (`Linear8bitLt`)| ~0.15                | bitsandbytes        |
| NF4 (blocksize 64)   | ~0.0625              | fp32 absmax / 64    |
| NF4 + double-quant   | ~0.017               | int8 absmax / 64 + fp32 second level (measured via `quant_state`) |

The quantization-overhead component also includes a transient **dequant
workspace** (`2 * hidden * intermediate * 2 B`): each 4-bit matmul
materializes a bf16 copy of the weight tile.

LoRA adapter weights themselves are charged at 4 B/param (PEFT keeps adapters
in fp32 on quantized bases).

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
model is frozen), and the adapters live in fp32:

```
gradients_bytes = trainable_params * 4.0     # fp32 adapters
```

For full fine-tuning we charge bf16 gradients (the fp32 master copy is
accounted for in the optimizer term).

## 4. Optimizer states

Per-parameter bytes for the most common optimizers:

| optimizer                      | bytes/param |
| ------------------------------ | ----------- |
| adamw_torch (fp32 m+v+master)  | 12.0        |
| paged_adamw_32bit              | 12.0        |
| adamw_8bit / paged_adamw_8bit  | 2.5         |
| sgd                            | 4.0         |
| sgd + momentum                 | 8.0         |
| lion_8bit                      | 2.0         |
| adafactor                      | 4.0         |

This is multiplied by `trainable_params`, *not* total parameters. For LoRA
/ QLoRA, the optimizer footprint is consequently tiny (a few MB).

## 5. Activations

Shaped after *"Reducing Activation Recomputation in Large Transformer
Models"* (Korthikanti et al., 2022), with coefficients re-fitted on the
modern HF stack (SDPA attention, SwiGLU MLPs, bitsandbytes 4-bit):

```
per_layer = s * b * (9 * h * act_bytes  +  mlp_tensors * ffn * mlp_bytes)

mlp_tensors = 4.5   for SwiGLU families (llama, qwen2, mistral, gemma, ...)
              2.8   for classic 2-matmul MLPs (gpt2, phi, opt, ...)
mlp_bytes   = 4.0   under QLoRA (kbit intermediates are held in fp32)
              act_bytes (2.0 for bf16) otherwise
```

- **Fused attention** (SDPA / flash-attn) does not materialize the
  `(b, a, s, s)` softmax matrix; with `--attn eager` we add the classic
  `5 * a * s²` term per layer.
- **Gradient checkpointing** keeps only each block's input
  (`2 * s * b * h * act_bytes` per layer) plus **one** full layer's
  activations for the recomputation peak during backward.

Note what checkpointing does *not* remove: the logits chain below. That is
why real-world peaks at seq 2048 stay several GiB even with checkpointing on.

## 6. Logits / loss chain

The dominant training buffer for modern large-vocab models, and the term most
older estimators miss:

```
logits_bytes = s * b * vocab * (act_bytes + 12)
             ≈ s * b * vocab * 14         # bf16 logits
```

That is: bf16 logits (2 B) + the fp32 upcast the HF loss performs (4 B) +
log-softmax workspace (4 B) + the fp32 logits gradient allocated in backward
(4 B). For Qwen2.5 (vocab 151 936) at seq 2048 this is **~4.1 GiB** — more
than the entire 4-bit weight footprint of the 1.5B model. It scales linearly
with `seq_len * batch * vocab` and is unaffected by gradient checkpointing.

Fused cross-entropy kernels (e.g. Liger) collapse most of this term; the
estimator models the stock HF Trainer path and says so in `assumptions`.

## 7. CUDA / fragmentation overhead

PyTorch's caching allocator, CUDA context, cuBLAS / cuDNN workspaces, and
fragmentation eat a non-trivial fraction of VRAM. We model this as a flat
fraction (default 8%) of the GPU's total VRAM. Calibration can tune this.

## 8. Safety margin

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

The 0.85 threshold is empirical: with that headroom, every QLoRA
configuration we benchmark on an RTX 4080 finishes the smoke run without OOM
(see `docs/rtx4080_baselines.md`), including a seq-4096 run measured at 83%
of VRAM.

## When the estimator is wrong

Common reasons for the static estimate diverging from reality:

- **A fused-CE kernel is active** (Liger, cut-cross-entropy): the logits
  component largely disappears and the estimate is several GiB too high.
  This is the safe direction, but worth knowing.
- **Very long seq_len (≥ 8192)**: allocator fragmentation grows with the
  largest single tensors; the flat 8% overhead can be too optimistic.
- **No flash-attn**: if the model silently falls back to eager attention,
  pass `--attn eager` so the `s²` term is included.
- **Different PEFT versions**: the fp32 upcast of embeddings/norms is
  `prepare_model_for_kbit_training` behavior; skipping that call (or using
  `bnb_4bit_quant_storage` tricks) changes the static term.
- **Loaded display GPU**: the OS / desktop / browser take VRAM at runtime.
  Use `canifinetune doctor` to see free VRAM, and pass that as `--gpu-vram-gb`
  if you want a tighter feasibility decision.

When in doubt: run `canifinetune bench` and `canifinetune calibrate`.
