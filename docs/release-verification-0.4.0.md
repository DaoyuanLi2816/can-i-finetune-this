# 0.4.0 release verification

Release source: `2edf707ac2040a2dc7a55dfe26dfa9f427b8fc45`, annotated tag
`v0.4.0`. Baseline was `053ce9e115b213e0ded3634e00d564ef97561f9b` and
published 0.3.0. Implementation and confirmed delivery fixes were integrated
through PRs [11](https://github.com/DaoyuanLi2816/can-i-finetune-this/pull/11),
[12](https://github.com/DaoyuanLi2816/can-i-finetune-this/pull/12),
[13](https://github.com/DaoyuanLi2816/can-i-finetune-this/pull/13) and
[14](https://github.com/DaoyuanLi2816/can-i-finetune-this/pull/14).
The [eight-job source CI](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/runs/37164526351)
passed on the exact release commit.

## Confirmed problems and implementation

The old recipe had divergent TRL/raw-data and pretokenized paths, first-turn
formatting, a BF16-to-FP16 load fallback without matching training precision,
and broad TypeError retries that removed requested options. One package-backed
Transformers runtime, shared validated configuration and dataset processing now
serve generated recipes and benchmarks. The default remains all-token loss.
System and multi-turn history are preserved; assistant-only supervision requires
verified native masks or completion offsets. Causal-shift supervision, EOS/PAD,
malformed rows, truncation and staged saves have behavioral regressions.

The actual old tiny recipe completed one update after a Windows GBK
UnicodeDecodeError prevented TRL import and selected the legacy fallback. This
does not establish that all modern TRL versions fail. The new runtime removes
the confirmed semantic split independently of that environment-specific result.

Actual Transformers 5.8.1 configuration merging also retained a checkpoint's
4-bit configuration despite a requested int8 configuration. Declared native
pre-quantized bases are now rejected before weight loading, including metadata
discovery. This config-only probe is not a quantization accuracy experiment.
The shared guard is covered without downloading weights. Doctor installation
advice now matches qualified constraints, and the PyPI README's main links use
absolute versioned repository URLs after a published relative link returned 404.

Key implementation: `src/canifinetune/configuration.py`, `training/data.py`,
`training/runtime.py`, `bench/runner.py`, `estimator/memory.py`,
`utils/gpu.py`, `evidence.py`, `demo.py`, packaged recipe/web resources,
and `.github/workflows/{ci,release}.yml`. See the
[implementation record](delivery-0.4.0.md) and [training contract](training.md).

## Source tests and installed candidate

Commands used in isolated project environments:

```console
python -m pytest -q -m "not training" --cov=canifinetune --cov-fail-under=70
python -m pytest -q -m training
ruff check .
ruff format --check .
mypy src
python scripts/check_generated.py
python -m build --outdir .cache/release-0.4.0
python -m twine check .cache/release-0.4.0/*.whl .cache/release-0.4.0/*.tar.gz
python scripts/verify_candidate.py --dist .cache/release-0.4.0 --tag v0.4.0 --commit 2edf707ac2040a2dc7a55dfe26dfa9f427b8fc45
```

Core: 102 passed, 71.44% coverage; 14 training tests selected separately.
Each minimum/recommended local stack passed all 14 real CPU integration tests.
CI covers core Python 3.10–3.14, Linux minimum/recommended training and Windows
recommended training on Python 3.12. The original 70% coverage gate is retained.
Optional torch runtime execution is verified separately rather than represented
as torch-free core coverage. Ruff, format, mypy and generated Python checks passed.

The isolated build produced an sdist first and built the wheel from that sdist.
Twine, metadata, README, license and packaged-resource checks passed. The exact
two files below were installed non-editably, qualified outside the checkout and
uploaded as release assets. Private earlier candidates were excluded. No rebuild
or file substitution occurred after this qualification.

| Distribution | SHA-256 |
| --- | --- |
| `canifinetune-0.4.0-py3-none-any.whl` | `080b1bb450c8abb8cf8213f15717131fefe90544e79b06cdd8416a1e5e07cd58` |
| `canifinetune-0.4.0.tar.gz` | `6294418f881271483e7aeaba8bcd9094c323d0a8c4ce635b68a54c1bf6c6fcdb` |

Package source fingerprint:
`b0cdb9ac5cc747e284e584d5de7c3312442ac1d1e905fffffc991a44c969b10c`.
The [candidate manifest](https://github.com/DaoyuanLi2816/can-i-finetune-this/releases/download/v0.4.0/candidate.json)
binds the commit, files and
[qualification receipt](https://github.com/DaoyuanLi2816/can-i-finetune-this/releases/download/v0.4.0/qualification.json).
The receipt's SHA-256 is
`1cc0924f3bebc046a0a2777fd66fdf916d76343d5e24954abe762c3656749478`.

Core qualification ran without torch: installed import/version, CLI,
offline estimate/recommend, recipe generation/AST and demo/core equality.
The README estimate is 8.420 GiB, `yes` against 16 GiB, heuristic `medium`.
Missing-torch bench exits 1 and records a dependency failure with absent peaks;
recipe input/dependency errors exit 2, and genuine recipe OOM exits 3.

Minimum CPU stack: Python 3.12.13, torch 2.6.0+cu124 (CPU execution),
Transformers 4.57.6, PEFT 0.18.1, Accelerate 1.12.0, bitsandbytes 0.49.2.
Recommended stack: Transformers 5.8.1, PEFT 0.19.1, Accelerate 1.13.0;
the same Python/torch/bitsandbytes. CPU full/LoRA each made two actual updates,
changed 26/8 saved parameter tensors, saved and reloaded for generation.

Candidate CUDA qualification used one native Windows RTX 4080, total 15.992 GiB,
driver 596.49, CUDA 12.4, BF16 and SDPA. LoRA and NF4 double-quant QLoRA each
completed two updates, changed eight saved adapter tensors, saved and reloaded.
QLoRA used paged AdamW8bit, rank 128 to exercise real uint8 optimizer state,
14 quantized linear layers and verified BF16 compute after updates.
Allocated/reserved peaks were 0.017112/0.021484 GiB for tiny LoRA and
0.017678/0.021484 GiB for tiny QLoRA. These cover load through final update.
Free budgets used the lower CUDA-driver/nvidia-smi view, approximately 14 GiB.
Waiting under earlier unrelated GPU pressure did not alter gates, stop another
process or count an unrun attempt as a pass.

## Prospective memory comparison

Four predeclared Qwen2.5-0.5B-Instruct observations at pinned revision
`7ae557604adf67be50417f59c2c2f167def9a775` cover LoRA/QLoRA and sequence
256/512, three updates each, one GPU. New weights were 988,097,824 bytes,
within the 1.5 GB budget. Their original source fingerprints and driver-view
free-memory fields remain unchanged; they are distinct from final wheel receipts.

Published 0.3.0 and 0.4.0 predictions on the same observations are identical:
MAPE 46.2%, median signed error +46.0%, range +29.3% to +63.7%, four
overestimates and zero underestimates. No accuracy improvement is demonstrated.
The process planning proxy excludes the separate safety allowance and is compared
with reserved allocator peaks; device free memory is a different measurement scope.
False-feasible is 0/4 for the recorded short workload/budget. False-infeasible
is undefined because there is no predicted-no sample. These are descriptive
counts, not calibrated OOM probabilities. No new cohort observations were fitted.

See [raw records and paired analysis](../benchmarks/validation-0.4.0) and
[full validation report](validation-0.4.0.md). Historical raw measurements are
preserved. A resource-pressure repeat was interrupted and excluded; it is neither
a successful observation nor an OOM with a fabricated peak. Tiny/random and short
real-model runs demonstrate execution, not useful fine-tuned language quality.

## Public release and verification

GitHub [v0.4.0](https://github.com/DaoyuanLi2816/can-i-finetune-this/releases/tag/v0.4.0)
was published at 2026-10-04 00:41:55 UTC. The
[release workflow](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/runs/37165739098)
completed successfully: candidate verification, Trusted Publishing and public
installation all passed. [Official PyPI 0.4.0](https://pypi.org/project/canifinetune/0.4.0/)
exposes exactly the wheel and sdist above. Both files were downloaded from
`files.pythonhosted.org` and hashed again; both match the qualified release assets.
PyPI displays publishing attestations for both files, bound to the release commit
and the existing `.github/workflows/release.yml` publisher. The no-environment
OIDC binding is preserved; only the upload job has `id-token: write`.

Post-release commands used fresh isolated Python 3.12 environments:

```console
uv pip install --python PUBLIC_CORE_PYTHON --index-url https://pypi.org/simple --no-cache canifinetune==0.4.0
uv pip install --python PUBLIC_TRAIN_PYTHON --index-url https://pypi.org/simple --no-cache -c constraints/train-recommended.txt "canifinetune[train]==0.4.0"
python scripts/verify_public.py --version 0.4.0 --candidate .cache/release-0.4.0
PUBLIC_CORE_PYTHON /absolute/path/scripts/qualify_candidate.py --receipt public-core-qualification.json
PUBLIC_TRAIN_PYTHON /absolute/path/scripts/qualify_candidate.py --training --cuda --receipt public-training-qualification.json
```

`PUBLIC_*_PYTHON` denotes each environment's interpreter, not a shell-independent
literal command. Torch 2.6.0+cu124 and recommended train dependencies were installed
from their official indexes before the train-extra command. Qualification commands
ran with the working directory outside the checkout. Imports resolve to installed
site-packages, `direct_url.json` is absent for the index installs, author/version
metadata matches, and both public packages have the candidate source fingerprint.
The core environment has no torch. Public CPU full/LoRA and CUDA LoRA/QLoRA each
completed two updates, changed saved tensors and reloaded for generation; QLoRA
again allocated actual uint8 optimizer state. The README CLI estimate returned
8.420 GiB/YES/medium. All ten versioned README documentation/license/constraint
links returned HTTP 200, and the live PyPI badge displayed 0.4.0.

A fresh public-package pinned 0.5B QLoRA recipe at sequence 256 also completed
two updates, changed 96 saved adapter tensors and reloaded for eight-token
generation. It used rank 16, paged AdamW8bit with float32/uint8 state, BF16
compute, 168 quantized linear layers, four instruction rows and no dropped or
truncated rows. Allocated/reserved peaks were 1.419052/1.521484 GiB; cached
weights were reused, with no additional weight download. This is a short
quickstart execution check, not a new accuracy-cohort observation or quality test.

The [public qualification receipt](https://github.com/DaoyuanLi2816/can-i-finetune-this/releases/download/v0.4.0/public-qualification.json)
contains the official distribution hashes, installation checks and redacted run
records. The local demo was restarted from the public core installation; its
visible 8.42 GiB result and generated commands were checked again. No external
release blocker remains. Documentation completion commits do not move the tag
or replace the published package files.

## Product entry, compatibility and attribution

The [PyPI-only quickstart](quickstart.md) separates native Windows and Linux/WSL
environment commands. `canifinetune demo` serves the same estimator at
`http://127.0.0.1:8765` using stdlib and packaged resources. Desktop and 375px
mobile layouts, free-budget inputs, errors, stale-result clearing and copied
commands were checked. Repository Pages is unconfigured; this is a shipped local
application, with no claim of a hosted service. Community measurement export is
manual, explicit opt-in, redacted by allowlist and unreviewed by default; it does
not automatically enter calibration or authoritative accuracy statistics.

Core supports Python 3.10–3.14; training qualification is Python 3.12 with the
minimum/recommended constraints above. Regenerate recipes after upgrading, remove
unknown YAML keys, and deliberately opt into right truncation if needed. CPU
requires fp32; unsupported BF16/full-FP16 requests fail without silent fallback.
Use raw unquantized bases and request quantization explicitly. Old calibration
caches require regeneration. Saved inference artifacts are not full training resume.
WSL GPU, other GPUs, Flash Attention, Liger, int8/fp4 accuracy, long training and
language quality remain unqualified; no new cross-hardware claim is made.

All new Git commit authors and package author metadata are Daoyuan Li. No agent/bot
authors, coauthors or hand-maintained contributors were added. GitHub's merge
committer, workflow actors and package attestations remain platform provenance.
Legitimate historical third-party attribution and licenses were preserved.
