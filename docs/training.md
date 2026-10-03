# Training contract and migration

0.4.0 recipes import a versioned package runtime. Regenerate older recipes and
install their `requirements.txt`; changing a legacy formatter does not change a
0.4.0 run. TRL and datasets are no longer training dependencies. Config paths are
relative to the YAML file; command-line path overrides are relative to the shell.
Unknown YAML keys fail (remove legacy `save_steps`). Default all-token loss remains.

One JSONL line is either instruction/input/output or native messages. Blank lines
are counted. Malformed rows stop with a line number, and no rows are dropped.
Instruction strings are formatted once, tokenized without extra special tokens,
and end in one EOS. A fast tokenizer's complete-text offsets define completion
boundaries; overlapping prompt/answer tokens are masked in assistant-only mode.

Messages preserve every turn and optional initial system text. User/assistant
must alternate and end with a nonempty assistant. Native chat templates insert
their own special tokens. Assistant-only loss requires generation spans supplied
by that template; we verify its tokens match the ordinary native rendering. Tools,
multimodal input, extra text fields, empty responses and unsupported role order
fail explicitly. A template without generation masks cannot use assistant loss.

`truncation: error` is default. Opt-in `right` truncation requires both a retained
response token after position zero and valid labels after the causal shift.
An answer removed by a long prompt fails. Fixed right padding masks positions,
not token IDs, preserving EOS when EOS and PAD share an ID. Statistics record
rows, blanks, truncations, supervised tokens and zero dropped rows. Mixed schemas
may coexist across lines if both are individually valid for the selected mode.

`TrainingConfig` defines dtype, attention, quantization, LoRA targets/rank/alpha/
dropout, optimizer, checkpointing and loss backend across product entry points.
The run records requested and effective fields, resolved revision, actual weight
and optimizer-state dtypes, optimizer class and quantized linear layer count.
Attention, base weights, checkpointing and 4-bit compute precision are verified
before and after updates. Qualified bitsandbytes 0.49's first-input precision
heuristic is disabled to preserve an explicit 4-bit compute request. Adapter
weights and PEFT-prepared unquantized parameters can be fp32 with bf16 compute.

CPU supports explicit fp32 full/LoRA with native AdamW or SGD. CUDA BF16 requires
hardware support; there is no dtype/attention retry. Full fp16 weight training is
rejected. Single default-device training only (`cpu` or `cuda`); explicit secondary CUDA
device selection is rejected until it is qualified. Actual parameter devices are
checked before and after updates; remote code is disabled and gated access
must already be granted. Flash Attention and Linux/Triton Liger are experimental,
not release-qualified. Liger uses a separate low-confidence stock planning proxy,
never stock calibration or a stock-loss accuracy claim.

Trainer executes genuine optimizer updates and requires finite loss. A successful
save is staged then renamed to model/adapter. Failed/OOM/save-error runs record
an error, remove partial staged artifacts and cannot pass evaluation. Successful
artifacts support inference reload, not complete optimizer/RNG resume. Nonempty
run directories are never overwritten. Missing dependencies, configuration,
access and runtime failures are distinguished; OOM has no fabricated exact peak.

Bench shares model and adapter loading but intentionally uses synthetic all-token
stock loss. It rejects assistant-only and Liger requests. Its `steps` are optimizer
updates, each containing the requested accumulation microsteps. Forward-only
probes are labelled and excluded from independent training accuracy statistics.

Legacy measurement imports remain readable as legacy fits and are incompatible
with the 0.4 estimator. New fits require complete successful current measurements
and matching configuration profiles. Mixed estimator versions cannot apply.
