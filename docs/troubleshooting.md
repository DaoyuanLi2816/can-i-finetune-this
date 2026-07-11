# Troubleshooting

## CUDA OOM during training

In order of "least disruptive" first:

1. Re-run with `--steps 1` to isolate whether OOM is at warmup or after several
   steps.
2. `micro_batch_size = 1`.
3. Increase `gradient_accumulation_steps` to preserve effective batch size.
4. Halve `seq_len` (e.g. 2048 → 1024).
5. Halve `lora_rank` (e.g. 16 → 8).
6. Restrict `target_modules` to attention-only (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
7. If `method == "lora"`, switch to QLoRA: `--method qlora --quantization nf4_double_quant --optimizer paged_adamw_8bit`.
8. Use a smaller base model. `canifinetune recommend --gpu-vram-gb <your-gb>`
   will list options that fit.

If OOM happens at *model load* (before training starts), gradient checkpointing
will not help — you need 4-bit quantization or a smaller model.

## bitsandbytes import or `CUDA_SETUP` errors

- Update: `pip install --upgrade bitsandbytes>=0.43.1`. Recent versions ship
  official **Windows wheels** — no manual build needed.
- Verify: `python -c "import bitsandbytes as bnb; print(bnb.__version__)"`.
- `bnb.optim.PagedAdamW8bit` requires `bitsandbytes >= 0.43`.

## CUDA / driver mismatch

Symptom: `torch.cuda.is_available()` is `False` even though `nvidia-smi`
works.

Fix: install a torch wheel that matches your driver's CUDA support level:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch
# or, for older drivers:
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install --index-url https://download.pytorch.org/whl/cu118 torch
```

The matrix is published at https://pytorch.org/get-started/locally/. Driver
13.2 (the version behind the RTX 4080 baselines in this repo) supports
`cu121` and `cu124` wheels; `cu124` is what was actually used.

## Windows / WSL caveats

- `bitsandbytes` ≥ 0.43 supports native Windows. Older versions need WSL2.
- File I/O inside OneDrive can sync `*.safetensors` files mid-write. If you
  see strange checkpoint corruption, either set `HF_HOME` to a path outside
  OneDrive or pause OneDrive sync during training.
- `signal` handling differs from POSIX; if a training run is interrupted
  with Ctrl+C and torch processes are left behind, kill them via
  `Get-Process python | Stop-Process`.

## flash-attn install failure

flash-attention 2 builds against a specific CUDA toolkit and needs a matching
wheel. If `pip install flash-attn` fails:

- Use `attention_implementation: "sdpa"` in the recipe (PyTorch ≥ 2.2 has a
  built-in scaled-dot-product attention that is nearly as fast).
- Or use `"eager"` to fall back to the textbook implementation (slower,
  larger memory).

## Hugging Face gated models

Some models (e.g. `meta-llama/*`) require accepting the license and using an
HF token:

```bash
huggingface-cli login   # paste a token from https://huggingface.co/settings/tokens
```

The token is cached at `~/.cache/huggingface/token`. **Do not** commit it.

`canifinetune estimate` does **not** require download access — the curated
metadata table covers the most popular gated models. But `canifinetune bench`
must actually load the weights, so the token has to be set.

## "No NVIDIA GPU detected"

`canifinetune doctor` says no GPU even though you have one:

- WSL guests don't always see the GPU; install NVIDIA's CUDA-on-WSL2 layer.
- Headless VMs without a passed-through GPU obviously can't.
- Some Windows users have nvidia-smi at `C:\Windows\System32\nvidia-smi.exe`
  and it's not in PATH. Add `C:\Windows\System32` or use the full path.

## Display GPU is busy

Browsers (Chrome, Edge, Firefox), Discord, Windows compositor, NVIDIA overlay,
and tools like Ollama can each take 0.5–2 GB on your display GPU. `nvidia-smi`
shows the breakdown.

Strategies:

- Pass `--gpu-vram-gb <free-gb>` instead of total VRAM to the estimator.
- Close GPU consumers before training.
- On Windows, if you have a second GPU, set the display to that GPU in
  Settings → System → Display → Graphics so the training GPU is unloaded.

## PyTorch memory fragmentation

Symptom: training fits at step 1 but OOMs at step N with "tried to allocate
… free memory but reserved more".

Fix: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching.
This is opt-in upstream because it interacts poorly with some custom CUDA
kernels, but it solves most fragmentation issues on consumer GPUs.

## bf16 not supported

Symptom: model loads in fp32 even though you asked for bf16.

bf16 requires Ampere or newer (sm_80+). The RTX 4080 (Ada Lovelace, sm_89)
supports it. On older cards (Volta / Turing / older Ampere consumer cards),
fall back to `fp16`. `train.py` already auto-falls-back if
`torch.cuda.is_bf16_supported()` is False; you can also set `base_dtype: fp16`
in `config.yaml` explicitly.

## "no nvcc" but torch CUDA works

That's normal. `nvcc` is the CUDA toolkit compiler, used to build kernels from
source. PyTorch ships pre-compiled CUDA kernels in its wheel, so you don't
need `nvcc` to *use* torch + CUDA. You only need it to *compile* custom
extensions like flash-attn from source.

## NCCL warnings on single-GPU

Some logs mention NCCL even on a single GPU. NCCL is harmless on a single
GPU — `canifinetune` does not use it. You can ignore these messages.
