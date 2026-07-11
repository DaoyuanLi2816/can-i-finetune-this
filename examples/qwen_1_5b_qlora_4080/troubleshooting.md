# Troubleshooting

Generated for `Qwen/Qwen2.5-1.5B-Instruct` (qlora, seq_len=1024, micro_batch=1, rank=16).

## CUDA OOM during training

In order of "least disruptive" first:

1. `--max-steps 1` — confirm whether the OOM is at warmup or after several steps.
2. Set `micro_batch_size` to 1 in `config.yaml` (it likely already is).
3. Increase `gradient_accumulation_steps` to keep the effective batch size the same.
4. Halve `seq_len` (e.g. 1024 → 512).
5. Halve `lora_rank` (e.g. 16 → 8).
6. Restrict `target_modules` to attention only: `["q_proj", "k_proj", "v_proj", "o_proj"]`.
7. If method == lora, switch to `qlora` with `quantization: "nf4_double_quant"` and
   `optimizer: "paged_adamw_8bit"`.
8. As a last resort: use a smaller base model. `canifinetune recommend --gpu-vram-gb <your-gb>`
   will list options that fit.

If the OOM happens at *model load* (before training starts), you need to
quantize the weights (move to QLoRA) or pick a smaller model — gradient
checkpointing won't help with weight storage.

## bitsandbytes import or `CUDA_SETUP` errors

- On Windows, `pip install --upgrade bitsandbytes>=0.43.1` ships official Windows wheels.
- Verify the install with: `python -c "import bitsandbytes as bnb; print(bnb.__version__)"`.
- `bnb.optim.PagedAdamW8bit` requires a recent enough version (≥ 0.43).

## CUDA / driver mismatch

`torch.cuda.is_available()` is False even though `nvidia-smi` works:

- Your torch wheel was built for a CUDA toolkit version your driver doesn't support.
- Reinstall torch with the right index, e.g.:
  `pip install --index-url https://download.pytorch.org/whl/cu124 torch`.
- For an older driver, try `cu121`; for an even older one, `cu118`.

## bf16 not supported

If `torch.cuda.is_bf16_supported()` is False, change `base_dtype` in
`config.yaml` to `fp16`. The training script auto-falls-back, but explicit is
better.

## Hugging Face gated model

Some models (e.g. `meta-llama/*`) require accepting the license and an HF
token:

```bash
huggingface-cli login   # paste a token created at https://huggingface.co/settings/tokens
```

The token lives in `~/.cache/huggingface/token`; do **not** check it into git.

## flash-attn install failure

Flash-attention 2 requires a recent CUDA + a matching wheel. If
`pip install flash-attn` fails, set `attention_implementation: "sdpa"` in
`config.yaml`. PyTorch ≥ 2.2 has a built-in scaled-dot-product attention that
is almost as fast and avoids the build.

## display GPU is busy

Your desktop, Chrome, Edge, and tools like Ollama can occupy 1–3 GB on your
GPU. `nvidia-smi` shows the breakdown. The estimator subtracts ~5–8% as a
safety margin, but a heavily loaded display GPU may still OOM. Close other
GPU consumers, or set `gpu_vram_gb` in `canifinetune` to your *measured free*
VRAM, not your card's total.
