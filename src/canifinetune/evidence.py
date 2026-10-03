"""Validated opt-in evidence export and explicitly defined error denominators."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SharedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: int = 2
    units: str = "GiB"
    status: str
    config: dict
    env: dict
    gpu: dict
    measured: dict
    estimated_total_gb: float | None = Field(None, gt=0)
    estimated_process_reserved_gb: float | None = Field(None, gt=0)
    estimated_feasible: str = "unknown"
    prediction_budget_gb: float | None = Field(None, gt=0)
    estimator_version: str
    completed_steps: int = Field(0, ge=0)
    workload: str
    provenance: dict
    effective_configuration: dict = Field(default_factory=dict)
    success: bool
    oom: dict
    model_family: str = "undisclosed"

    @model_validator(mode="after")
    def check(self):
        if self.schema_version != 2 or self.units != "GiB":
            raise ValueError("supported evidence format is schema 2 with GiB units")
        if self.status not in {
            "success",
            "oom",
            "configuration_error",
            "dependency_error",
            "runtime_error",
            "access_error",
            "not_run",
        }:
            raise ValueError("unknown outcome status")
        if self.success != (self.status == "success"):
            raise ValueError("status and success disagree")
        if bool(self.oom.get("happened")) != (self.status == "oom"):
            raise ValueError("OOM status must match oom.happened")
        allowed_metrics = {"peak_reserved_gb", "peak_allocated_gb"}
        if set(self.measured) - allowed_metrics:
            raise ValueError("only process allocator peaks are accepted")
        for value in self.measured.values():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("missing peaks must be omitted, never zero/NaN/negative")
        if self.status != "success" and self.measured:
            raise ValueError("failed/OOM observations cannot claim an exact peak")
        if (
            len(self.measured) == 2
            and self.measured["peak_reserved_gb"] < self.measured["peak_allocated_gb"]
        ):
            raise ValueError("reserved peak cannot be below allocated peak")
        return self


_CONFIG_KEYS = {
    "method",
    "seq_len",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "steps",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_scope",
    "optimizer",
    "quantization",
    "base_dtype",
    "gradient_checkpointing",
    "attention_implementation",
    "forward_only",
    "loss_backend",
    "loss_mode",
    "training_backend",
    "truncation",
}


def export_evidence(data, *, include_public_model=False):
    """No network, secrets, paths, raw messages, hostname, usernames or GPU UUIDs."""
    if data.get("schema_version") != 2:
        raise ValueError("legacy evidence lacks verified configuration; rerun bench before sharing")
    config = {
        key: data.get("config", {}).get(key)
        for key in _CONFIG_KEYS
        if key in data.get("config", {})
    }
    effective = {
        key: data.get("effective_configuration", {}).get(key)
        for key in _CONFIG_KEYS
        if key in data.get("effective_configuration", {})
    }
    family = "undisclosed"
    if include_public_model:
        model = data.get("config", {}).get("model_id", "")
        # Sharing the identifier requires an explicit opt-in; no local paths.
        import re

        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model):
            raise ValueError("only an explicitly opted-in public Hub identifier can be shared")
        config["model_id"] = model
        config["revision"] = data.get("effective_configuration", {}).get("resolved_revision")
        family = data.get("model_family", "unknown")
    env = {
        key: data.get("env", {}).get(key)
        for key in (
            "python",
            "platform",
            "canifinetune",
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "bitsandbytes",
            "cuda_version",
            "source_fingerprint_sha256",
        )
        if data.get("env", {}).get(key) is not None
    }
    gpu = {
        key: data.get("gpu", {}).get(key)
        for key in ("name", "total_vram_gb", "free_vram_gb", "driver_version", "compute_capability")
        if data.get("gpu", {}).get(key) is not None
    }
    success = data.get("status") == "success"
    measured = {
        key: value
        for key, value in data.get("measured", {}).items()
        if key in {"peak_reserved_gb", "peak_allocated_gb"}
        and success
        and value is not None
        and value > 0
    }
    result = SharedEvidence(
        status=data.get("status", "not_run"),
        config=config,
        env=env,
        gpu=gpu,
        measured=measured,
        estimated_total_gb=data.get("estimated_total_gb"),
        estimated_process_reserved_gb=data.get("estimated_process_reserved_gb"),
        estimated_feasible=data.get("estimated_feasible", "unknown"),
        prediction_budget_gb=data.get("prediction_budget_gb"),
        estimator_version=data.get("estimator_version", "unknown"),
        completed_steps=data.get("completed_steps", 0),
        workload=data.get("workload", "unknown"),
        provenance={"source": "community", "review_status": "unreviewed", "cohort": "unspecified"},
        effective_configuration=effective,
        success=success,
        oom={"happened": data.get("status") == "oom"},
        model_family=family,
    )
    return result.model_dump()


def error_summary(results):
    errors, underestimated, overestimated = [], 0, 0
    yes_runs = no_runs = false_yes = false_no = 0
    excluded = 0
    for data in results:
        status = data.get("status") or (
            "oom"
            if data.get("oom", {}).get("happened")
            else "success"
            if data.get("success")
            else "unknown"
        )
        provenance = data.get("provenance", {})
        # Historical fits and community uploads are never independent evidence.
        eligible = (
            data.get("schema_version") == 2
            and provenance.get("cohort") == "validation"
            and provenance.get("source") == "maintainer-local"
            and provenance.get("review_status") == "verified"
            and data.get("effective_configuration")
            and not data.get("config", {}).get("forward_only")
            and data.get("completed_steps", 0) == data.get("config", {}).get("steps")
        )
        # OOM cannot finish all steps; it can enter feasibility denominator only
        # when metadata explicitly identifies the same attempted workload.
        if status == "oom" and data.get("workload_match_verified"):
            eligible = (
                data.get("schema_version") == 2
                and provenance.get("cohort") == "validation"
                and provenance.get("source") == "maintainer-local"
                and provenance.get("review_status") == "verified"
            )
        if not eligible:
            excluded += 1
            continue
        predicted = data.get("estimated_feasible", "unknown")
        if predicted == "yes" and status in {"success", "oom"}:
            yes_runs += 1
            false_yes += int(status == "oom")
        if predicted == "no" and status in {"success", "oom"}:
            no_runs += 1
            false_no += int(status == "success")
        observed = data.get("measured", {}).get("peak_reserved_gb")
        predicted_peak = data.get("estimated_process_reserved_gb")
        if status == "success" and observed and predicted_peak:
            error = (predicted_peak - observed) / observed
            errors.append(error)
            underestimated += int(error < 0)
            overestimated += int(error > 0)
    return {
        "paired_successes": len(errors),
        "excluded": excluded,
        "mean_absolute_percentage_error": statistics.mean(abs(e) * 100 for e in errors)
        if errors
        else None,
        "median_signed_percentage_error": statistics.median(errors) * 100 if errors else None,
        "signed_percentage_error_range": [min(errors) * 100, max(errors) * 100] if errors else None,
        "underestimates": underestimated,
        "overestimates": overestimated,
        "predicted_yes_actual_oom": {
            "numerator": false_yes,
            "denominator": yes_runs,
            "rate": false_yes / yes_runs if yes_runs else None,
        },
        "predicted_no_actual_success": {
            "numerator": false_no,
            "denominator": no_runs,
            "rate": false_no / no_runs if no_runs else None,
        },
        "scope": "verified new validation cohort only; process reserved proxy excludes safety; marginal/unknown predictions excluded from feasibility denominators; short probe only",
    }


def read_results(paths):
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
