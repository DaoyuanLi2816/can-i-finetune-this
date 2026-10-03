# Design notes

A torch-free `TrainingConfig` defines shared meanings. CLI, loopback demo and
recommendation call the same metadata/formula estimator. Metadata comes from
manual overrides, curated catalogue entries, local config files or cached/remote
Hub config (never weights). Catalogue entries describe architecture, not a live
check of every requested revision; the training receipt records the resolved
revision. For a changed architecture provide an explicit metadata override.

Bench and generated recipes lazily import torch/Transformers/PEFT/bitsandbytes.
One strict runtime loads models and adapters with explicit dtype/attention and
no broad fallback. One dataset processor applies native templates, truncation,
padding and shifted causal-label checks. Recipes contain readable config and
small entry scripts backed by their exact package version. No TRL data path,
remote code, distributed backend or implicit gated approval.

Calibration schema 3 fits development measurements while leaving deterministic
static memory untouched. It requires compatible configuration profiles, estimator
version, family/method and GPU capacity. Validation/community/failed observations
are excluded from fitting. Reports keep historical fits separate from prospective
validation and state denominator rules. See [memory model](memory_model.md).

The stdlib HTTP demo binds loopback, serves packaged assets and only estimates
catalogue models. No arbitrary URLs, execution, login, telemetry or uploads.
Community sharing is an explicit allowlist export followed by manual submission
and maintainer verification. See [contributing](../CONTRIBUTING.md).

Extension work should add a model/target mapping with behavioral and bounded
training checks, or improve measured scope and configuration matching. A new
optimizer requires both real execution support and a corresponding memory model.
