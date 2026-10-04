# Compatibility and migration

| Component | Qualified scope |
| --- | --- |
| Core | Python 3.10–3.14, no torch; Linux/Windows CI |
| Training | Python 3.12, Torch 2.6.0; minimum/recommended combinations below |
| CPU | Real full/LoRA update, save/reload and data/failure tests on Linux and Windows |
| CUDA | Native Windows RTX 4080, CUDA 12.4 wheel, LoRA/QLoRA update/save/reload |
| Demo | Loopback stdlib server, desktop and 375px mobile layout |

## Reproduce a supported stack

| Stack | Transformers | PEFT | Accelerate | bitsandbytes |
| --- | --- | --- | --- | --- |
| Minimum | 4.57.6 | 0.18.1 | 1.12.0 | 0.49.2 |
| Recommended | 5.8.1 | 0.19.1 | 1.13.0 | 0.49.2 |

Choose the CPU or CUDA Torch 2.6.0 wheel separately. Use the
[constraints](https://github.com/DaoyuanLi2816/can-i-finetune-this/tree/v0.4.1/constraints)
and [platform quickstart](quickstart.md). Core and train dependencies are separate;
TRL/datasets are not required. Dependency ranges accept these qualified stacks,
not arbitrary future major versions.

WSL GPU, other GPU hardware, macOS training, Flash Attention and Liger have not
been qualified. Liger recipes remain experimental with a conservative stock-loss
planning proxy and low confidence; benchmark rejects it until a matching path
is qualified. Int8/fp4 accuracy and long training are not established here.

## Upgrade an existing recipe

1. Regenerate it with the installed package version; old exported scripts do not
   automatically acquire the new runtime contract.
2. Remove unknown legacy YAML keys. Keep all-token loss unless you deliberately
   select assistant-only and meet its tokenizer/data requirements.
3. Default truncation is `error`. Explicit `right` truncation must retain valid
   response supervision after the causal shift; records are not silently dropped.
4. Use an unquantized base and request quantization explicitly. Remote code,
   distributed execution and automatic gated-model approval are unsupported.
5. Recreate incompatible old calibration caches. The 0.4.1 presentation patch
   keeps **estimator version 0.4.0**: it changes package/documentation version,
   not the formulas or the identity of historical measurement versions.

Full training in FP16 is rejected; CPU needs fp32. Saved models/adapters are
inference artifacts, not complete optimizer/data-position/RNG resume checkpoints.
For detailed behavior see [training](training.md) and [troubleshooting](troubleshooting.md).
