# Memory model

`canifinetune` decomposes the GPU memory footprint of a training step into the
following components. All numbers are in bytes; the CLI rounds to GiB on output.

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

Historical RTX 4080 development measurements informed these coefficients.
Historical regression tolerances describe fitted-set consistency, not independent
accuracy. See [0.4.0 prospective validation](validation-0.4.0.md) for the frozen
new cohort, substantial conservative errors and old/new same-observation comparison.
No coefficients were fitted on that cohort. Confidence is a qualitative evidence
grade, never a statistical probability. All `*_gb` fields mean GiB (2**30 bytes).

Compare observed process `peak_reserved_gb` with the process planning proxy
`total_estimated_gb - safety_margin_gb`. Allocated and reserved differ; device
free/total readings are a different scope. For feasibility compare the total
planning budget (including safety once) with currently free capacity if supplied.
Keep overhead allowances based on actual total hardware capacity. OOM has no exact
peak; missing values are not zero. A three-update probe is bounded evidence.

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
weights_bytes  = linear_params * 0.5
               + (embeddings + norms) * 4.0        # fp32 after kbit-prepare
quant_overhead = linear_params * metadata_bytes_per_param
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

For mixture-of-experts models, adapter weights on expert MLP projections are
multiplied by `num_local_experts`; activation memory is multiplied only by
`num_experts_per_tok`. The base-model parameter count comes from Hub
safetensors metadata when available, with a config-derived MoE formula as the
offline fallback.

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

For full training, gradients use the explicitly loaded base parameter dtype.
FP32 weights therefore require FP32 gradients; BF16 uses BF16 gradients.

## 4. Optimizer states

Native torch AdamW stores two state tensors in parameter dtype. There is no
implicit FP32 master-weight copy. Non-fused CUDA foreach can add a tensor-sized
workspace: the planning charge is 3 times parameter bytes, versus 2 times for
fused AdamW. FP32 adapter AdamW thus charges 12 B/param (8 B states + 4 B workspace),
and fused charges 8 B. Plain SGD in this runtime has no momentum state (0 B).
8-bit/paged AdamW charges 2.5 B/param as a heuristic; small tensors may use FP32
state, so this is not an exact optimizer allocation prediction. Real receipts
record optimizer class and state dtypes. The runtime supports only optimizers in
TrainingConfig; historical formula entries are not a training support promise.

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

These are heuristic planning thresholds, not calibrated OOM probabilities.
Historical fitting runs do not establish a success rate on new hardware or
workloads. Prospective reports state the matched yes/no denominators explicitly;
`marginal` and `unknown` are excluded from those binary rates.

## When the estimator is wrong

Common reasons for the static estimate diverging from reality:

- **A fused-CE kernel is active** (Liger, cut-cross-entropy): the logits
  component largely disappears and the estimate is several GiB too high.
  This is the safe direction, but worth knowing.
- **Very long seq_len (≥ 8192)**: allocator fragmentation grows with the
  largest single tensors; the flat 8% overhead can be too optimistic.
- **Different attention**: explicitly select `--attn eager` if needed so the
  quadratic term is included. This runtime rejects silent attention changes.
- **Different PEFT versions**: the fp32 upcast of embeddings/norms is
  `prepare_model_for_kbit_training` behavior; skipping that call (or using
  `bnb_4bit_quant_storage` tricks) changes the static term.
- **Loaded display GPU**: the OS / desktop / browser take VRAM at runtime.
  Use `canifinetune doctor` to see free VRAM, and pass it as `--available-vram-gb`
  while retaining the hardware total in `--gpu-vram-gb`.

Run a bounded bench before relying on a budget. Calibration fits a development
cohort; it does not independently validate the resulting estimates.
