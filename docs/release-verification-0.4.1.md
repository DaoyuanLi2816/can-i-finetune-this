# 0.4.1 presentation and public-release verification

This record describes the immutable 0.4.1 artifacts. Current main's README and
documentation use English entry points without the custom logo.

The earlier README rewrite removed the display entries for the existing banner
and architecture image. The source assets survived. Version 0.4.1 restores the
original banner style and adds a new project logo, editable current SVG diagrams,
English/Chinese introductions and searchable Material documentation.

The runtime source differs from 0.4.0 only in the package version. Estimator
version remains 0.4.0; formulas and original experiment records remain unchanged.
See the [implementation record](delivery-0.4.0.md) for the underlying training,
data, configuration and evidence fixes.

## Source and exact distributions

- Release commit: `ec4465e87ea4e9c9f067c1ff5c37b275999c3667`.
- Tag and [GitHub Release](https://github.com/DaoyuanLi2816/can-i-finetune-this/releases/tag/v0.4.1): `v0.4.1`.
- [Official PyPI package](https://pypi.org/project/canifinetune/0.4.1/): `canifinetune==0.4.1`.
- Runtime source fingerprint: `bb0f7bdd357c291163f76b41679ba759348d76e158b0abc0ebd247f6e4ac73cf`.

| File | SHA-256 |
| --- | --- |
| `canifinetune-0.4.1-py3-none-any.whl` | `ced68e8c39bda14cca8526227ada205e5d30f4416c1adcae5021bdffe26534db` |
| `canifinetune-0.4.1.tar.gz` | `bb214c4477422d7b8fabc8018f86c03f309e7b2da445c8a9602d80a19c823fa7` |
| `qualification.json` | `f9914aacd7a4a0e5a6ad65e6c05d836313a00ac721aba32dd695cd33eb59e8e2` |
| `public-qualification.json` | `cd0422b4568f06eea7c09cb1c7a157215430287dac2cbba27e766e9d30a2c478` |

The release assets contain the candidate manifest, both redacted receipts and
SHA256SUMS. Wheel/sdist hashes match the PyPI JSON and the actual downloaded files
from `files.pythonhosted.org`. The wheel was built from the sdist, checked against
all package source/resources and uploaded unchanged. Wheel metadata contains the
README, documentation URL, MIT license and Daoyuan Li author identity. The sdist
also includes the Chinese README.

## Execution checks

Local qualification used native Windows, Python 3.12.13, Torch 2.6.0+cu124,
RTX 4080 (15.992 GiB), driver 596.49. Each installed candidate/public run executed
outside the checkout. The core environments contained no torch.

| Layer | Result |
| --- | --- |
| Core tests | 102 passed; 14 training cases explicitly deselected; 71.44% coverage |
| Static/generated Python | Ruff lint/format, mypy and all generated method checks passed |
| Candidate minimum stack | Transformers 4.57.6 / PEFT 0.18.1 / Accelerate 1.12.0; CPU full/LoRA passed |
| Candidate recommended stack | Transformers 5.8.1 / PEFT 0.19.1 / Accelerate 1.13.0; CPU full/LoRA and CUDA LoRA/QLoRA passed |
| Public recommended stack | Fresh official-index install; the same four training paths passed |
| Public quickstart | Pinned Qwen2.5-0.5B-Instruct QLoRA; two updates, 96 changed adapter tensors, reload/generation passed |

Training checks required actual saved-tensor changes and two optimizer updates.
QLoRA exercised 14 quantized layers and uint8 optimizer state. The public 0.5B
quickstart used existing cached weights, with a 1.419 GiB allocated / 1.521 GiB
reserved process peak measured from loading through the final update.

Validation commands included `pytest -q -m "not training" --cov=canifinetune
--cov-fail-under=70`, `python scripts/check_generated.py`, `python -m build`,
`twine check`, `scripts/qualify_candidate.py`, `scripts/verify_candidate.py`
and `scripts/verify_public.py`. CI additionally performed real offline data,
failure-path, CPU update/save/reload and installed-wheel checks on Linux/Windows.

## Public presentation and automation

- [PR #16](https://github.com/DaoyuanLi2816/can-i-finetune-this/pull/16) merged after
  nine successful build/test checks. Its deploy job was intentionally skipped.
- [Release-commit CI](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/runs/37177032789)
  completed all eight jobs successfully.
- [Pages deployment](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/runs/37177032807)
  built and deployed successfully. The initial public site had 17 content pages;
  all returned HTTP 200. Six visual/CSS assets matched the source bytes.
- The strict build checked 18 HTML files (including 404) and 809 internal
  page/asset/fragment links. Browser checks covered desktop/mobile layouts,
  dark mode, English/Chinese search and the public mobile architecture image.
- PyPI's README matches the tagged source; all five versioned visual URLs and
  both READMEs' documentation links/fragments were checked. GitHub displayed the
  restored banner, logo, screenshot and architecture image.
- [Trusted Publishing workflow](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/runs/37177255950)
  finished with verify, publish and public-install all successful. Its first
  public-install attempt failed because the versioned PyPI JSON exposed 0.4.1
  before the Simple installation index. Only that job was rerun after propagation;
  verify/publish retained their original timestamps and no second upload occurred.
- The dynamic Shields badge initially remained at 0.4.0, while GitHub's image
  proxy still displayed 0.3.0. Main's READMEs explicitly identify the verified
  0.4.1 release to avoid promoting a stale cached badge. Tagged package metadata
  remains immutable.

All new commit/tag/package author identities are Daoyuan Li. GitHub's merge
committer and normal Actions/attestation records are retained as platform records.

## Evidence limits

These bounded execution checks prove the package pipeline, not useful language
quality, long-run OOM safety or cross-GPU estimator accuracy. The four original
prospective observations remain unchanged: one GPU/model, 46.2% MAPE and all
overestimates. WSL GPU, other GPUs, Flash Attention and Liger are unqualified;
see [compatibility](compatibility.md).
