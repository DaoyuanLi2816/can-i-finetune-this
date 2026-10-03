# Troubleshooting this recipe

- Missing dependencies: install this recipe's requirements in the same environment.
- CPU: use method full/lora, base_dtype fp32, optimizer adamw_torch, device cpu.
- BF16 unsupported: select fp16 explicitly for adapters and rerun the estimate.
  There is no silent fallback. Full FP16 training is rejected.
- OOM: inspect run.json, reduce micro batch/sequence/rank or use checkpointing.
  A short probe is not a long-training guarantee. Retry into a new output directory.
- Empty/long/malformed data: inspect the reported line and dataset_format.md.
  Increasing seq_len changes the memory budget; estimate again.
- Unknown YAML keys: correct the key; ignored configuration is not permitted.
- Save failure: status is error, no completed artifact is advertised. Fix disk
  access/space and run again in an empty output directory. Resume is unsupported.
- Model access/offline errors: request access yourself through Hugging Face, use
  the official hf CLI for authentication, or select cached weights with
  local_files_only: true. Never put tokens in config, logs or shared evidence.
- Attention errors: choose a supported implementation explicitly and re-estimate.
  Flash Attention and Liger remain optional/unqualified here; no fallback removes them.
- Busy display GPU: keep actual total VRAM and separately supply currently free
  memory with estimate --available-vram-gb. Do not replace total capacity with free.
- Evaluation fails if run.json does not say success, the artifact is absent, or
  an adapter cannot load. It never silently generates from the bare base model.
