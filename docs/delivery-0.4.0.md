# 0.4.0 implementation and qualification record

Baseline: `053ce9e115b213e0ded3634e00d564ef97561f9b` (clean main). GitHub
and the live PyPI JSON API both report 0.3.0; the previous release workflow
succeeded. Main has no branch protection or repository rulesets; the maintainer
has ADMIN permission. Work uses a branch and CI before integration.

## Confirmed findings

| Review lead | Finding at the baseline |
| --- | --- |
| Two tokenization paths | Confirmed: prepared labels are discarded when TRL imports; SFTTrainer reprocesses raw rows. |
| First user/assistant | Confirmed in both formatters; system and later turns disappear in the legacy formatter, while modern TRL can prioritize messages. Actual behavior depends on TRL version. |
| BF16 fallback | Confirmed: load falls back to fp16 but TrainingArguments does not enable fp16 for a bf16 request. |
| Broad TypeError retry | Confirmed in bench, training and evaluation; drops attention and eventually dtype. |
| Training CI | Confirmed: the 80 passing baseline tests do not execute a training recipe. |
| Calibration evidence | Historical regression data informed the estimator; it is not independent validation. |
| Configuration divergence | Confirmed: QLoRA bench overrides compute dtype; fused optimizer name does not actually enable fused AdamW. |
| Trusted publishing | Already present and working; retain release.yml and its current no-environment OIDC binding. |

## Design decisions

Use one package-backed Transformers Trainer runtime, one validated configuration
and one dataset processor. Default remains all-token loss. Assistant-only chat
requires native generation masks; unsupported templates fail explicitly. No
string searches or prompt-prefix guesses. Reject truncation by default; opt-in
right truncation must retain shifted causal supervision and response tokens.
Keep historical measurements unchanged and avoid fitting the new validation data.

## Predeclared local validation cohort

Before observing new results, freeze a bounded cohort on the detected RTX 4080
(16 GiB): local tiny Qwen2 smoke (CPU full/LoRA, CUDA LoRA/QLoRA), then
Qwen2.5-0.5B-Instruct at sequence 256 and 512, LoRA and NF4 double-quant QLoRA,
batch 1, rank 8, attention targets, AdamW, checkpointing on, bf16, SDPA, stock
loss, 3 updates. Maximum new weight download 1.5 GB; maximum one larger process
at a time; stop on an unexpected OOM. Compare published 0.3.0 and candidate on
the same observations. This is one GPU and one additional real model, not a
cross-hardware or long-training guarantee. Recheck free memory before each run.

Baseline verification: `python -m pytest -q`: 80 passed. Windows Python 3.12,
torch 2.6.0+cu124, Transformers 5.8.1, PEFT 0.19.1, Accelerate 1.13.0,
bitsandbytes 0.49.2. nvidia-smi and torch report changing free memory; no other
processes will be stopped. Disk free space exceeds 500 GB.

## Implemented and verified locally

One Transformers runtime replaces the data-path split; no silent dtype/attention
retry. Native masks and offset boundaries preserve system/multi-turn/all-token
semantics. Strict configuration, staged saving and real reload are shared by
installed recipes. A loopback stdlib demo and explicit redacted export are shipped.
Calibration schema 3 separates compatible fits from prospective/community evidence.
The original 70% core coverage gate is retained; optional torch runtime has actual
CPU integration and candidate CUDA qualification rather than fabricated core coverage.

No-torch Python 3.12 core: 100 tests passed,
ruff/format/mypy/generated Python checks passed. Minimum and recommended CPU
stacks each passed 13 real integration tests. Four bounded prospective GPU
observations succeeded; predictions are unchanged from 0.3.0 and conservatively
high (MAPE 46.2%). No claim of accuracy improvement or cross-hardware validation.
See validation-0.4.0.md and its raw files. A resource-pressure repeat was interrupted
and excluded, without stopping unrelated processes. Historical raw files are intact.

The actual 0.3.0 tiny recipe completed one update after a GBK UnicodeDecodeError
prevented TRL import and selected its legacy fallback. This is version/environment
specific; it does not establish that modern TRL always fails. The statically
confirmed divergent processing paths were removed regardless of that import result.

## Delivery status

Implementation, candidate qualification, release and public-install verification
are in progress. No new version has been released yet.
