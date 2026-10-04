# Contributing

Start with a reproducible behavior change or a measurement on hardware we have
not qualified. Core stays torch-free; no account, telemetry or automatic uploads.

## Development

```console
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest -q -m "not training" --cov=canifinetune --cov-fail-under=70
python scripts/check_generated.py
```

For real offline CPU training, install torch 2.6.0 CPU, then `.[dev,train]` with
`constraints/train-min.txt` or `train-recommended.txt`, and run
`pytest -q tests/integration`. CI separates no-torch core tests from actual
update/save/reload tests and installed-wheel checks. CUDA evidence stays separate.
Generated Python needs syntax/lint and execution checks. Do not lower gates or
replace behavior checks with template-string assertions.

## Documentation and visuals

```console
python -m pip install -r requirements-docs.txt
mkdocs build --strict
python scripts/verify_docs.py --site site
mkdocs serve
```

Keep the English/Chinese entry points, README banner and architecture diagrams
aligned with actual behavior. Edit the SVG sources in `docs/assets`; retain
the original visuals and historical measurement records. Check desktop/mobile
layout and both search languages. PRs build the site without publishing; main
deploys it to GitHub Pages. Documentation dependencies are separate from core.

## Share a measurement manually

```console
canifinetune bench --model PUBLIC_OR_LOCAL_MODEL --method qlora --seq-len 256 --steps 3 --out-dir measurements
canifinetune evidence-export --input measurements/YOUR_RESULT.json --out share.json
canifinetune evidence-validate --input share.json
```

Preview `share.json`. Default export removes model identity, personal paths, raw
errors, notes, GPU UUIDs, hostnames and unknown fields. No samples or environment
variables are exported. `--include-public-model` is a separate explicit opt-in:
only use it for an identifier you are allowed to disclose. Public syntax does
not prove a repository is public. Review every retained field for sensitive
content before sharing. The tool performs no upload.

Open the measurement issue form or a PR adding the reviewed JSON. Describe the
actual workload, outcome and differences from the generated recipe. Submit JSON
as data, never commands/scripts or private weights. Schema 2 uses GiB. Missing
peaks must be absent; OOM is a boundary, not a zero or exact peak. Allocator peaks
and device total/free are different scopes. Local logs may contain private paths;
do not attach them without review.

Community exports remain `source: community`, `review_status: unreviewed` and
`cohort: unspecified`. A maintainer must reproduce/inspect configuration and
provenance before considering calibration. Uploads never automatically enter
calibration or authoritative validation summaries. Do not relabel existing
historical fits as independent validation. Keep originals and estimator versions.

Useful contributions: real measurements on another NVIDIA GPU; a native chat-mask
edge case with a local tokenizer; or a documented metadata/target mapping plus
an actual bounded recipe on a new model family. Preserve third-party attribution
and licenses. Maintainer: Daoyuan Li.
