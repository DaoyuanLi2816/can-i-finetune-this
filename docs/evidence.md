# Read and contribute measurement evidence

## Know which question a record answers

| Evidence | Use | Boundary |
| --- | --- | --- |
| Historical development measurements | Explain fitted coefficients and regression behavior | Not independent validation |
| Frozen prospective cohort | Compare predictions on new matched observations | One GPU/model, four short cases |
| Installed-package qualification | Prove update/save/reload and effective configuration | Not language quality or estimator accuracy |
| Community export | Supply a reviewable observation from another machine | Unreviewed by default; excluded from fits |

The [prospective 0.4.0 report](validation-0.4.0.md) preserves the four original
observations and the old/new paired comparison. Predictions are unchanged:
46.2% MAPE, four overestimates. False-feasible is 0/4 for the exact short workload
and recorded free budget; false-infeasible is undefined without predicted-no
samples. These counts do not estimate a calibrated general OOM probability.

## Compare matching measurements

All `*_gb` fields mean **GiB**, including historical JSON. Torch allocated and
reserved peaks describe the process allocator. Device total/free memory has a
different scope. A planning process proxy excludes the separate safety allowance;
total planning feasibility includes safety once. Do not mix these quantities.

Bench covers load through declared optimizer updates and initial state allocation.
OOM is an observed boundary with no fabricated exact peak. Missing/error/forward-only
records are not successful zero-memory runs. Reports exclude marginal/unknown,
unmatched workloads, failed/incomplete runs and unreviewed submissions from binary
feasibility rates. Confidence is an evidence grade, not a probability interval.

## Export only what you choose to share

```console
canifinetune evidence-export --input measurements/YOUR_RESULT.json
canifinetune evidence-validate --input evidence.json
```

Export is explicit opt-in and local. Inspect the preview, then share manually
through the [measurement issue form](https://github.com/DaoyuanLi2816/can-i-finetune-this/issues/new?template=measurement.yml)
or a PR. The allowlist removes private paths, samples, detailed exceptions,
model identifiers by default, GPU UUIDs and unknown fields. A separate
`--include-public-model` opt-in permits a public Hub identifier/revision.
Never include private data or credentials. There are no automatic uploads.

Community records are validated as data, not executed as instructions. Format
validation does not mean maintainer review and does not make a measurement an
authoritative accuracy sample. See
[contributing](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/main/CONTRIBUTING.md).
