import json

import pytest

from canifinetune.estimator.calibration import _result_to_sample
from canifinetune.evidence import SharedEvidence, error_summary, export_evidence


def record(status="success", predicted="yes"):
    return {
        "schema_version": 2,
        "status": status,
        "success": status == "success",
        "units": "GiB",
        "config": {
            "model_id": "private/secret",
            "steps": 3,
            "seq_len": 128,
            "device": "cuda",
            "local_path": "C:/private",
        },
        "completed_steps": 3,
        "effective_configuration": {"base_dtype": "bf16", "model_id": "private/secret"},
        "gpu": {"name": "RTX 4080", "uuid": "GPU-secret", "total_vram_gb": 16},
        "env": {"torch": "2.6", "hostname": "private"},
        "measured": {"peak_reserved_gb": 2, "peak_allocated_gb": 1.8}
        if status == "success"
        else {},
        "estimated_total_gb": 2.4,
        "estimated_process_reserved_gb": 2.2,
        "estimated_feasible": predicted,
        "prediction_budget_gb": 16,
        "estimator_version": "0.4.0",
        "model_family": "qwen2",
        "workload": "3 synthetic steps",
        "provenance": {
            "source": "maintainer-local",
            "review_status": "verified",
            "cohort": "validation",
        },
        "oom": {"happened": status == "oom", "message": "private path / token"},
        "notes": ["private details"],
    }


def test_export_is_allowlisted_and_opt_in():
    shared = export_evidence(record())
    text = json.dumps(shared)
    for private in ("private", "secret", "uuid", "hostname", "local_path", "notes", "model_id"):
        assert private not in text
    assert shared["provenance"]["review_status"] == "unreviewed"
    assert SharedEvidence.model_validate(shared)
    assert (
        export_evidence(record(), include_public_model=True)["config"]["model_id"]
        == "private/secret"
    )
    invalid = record()
    invalid["config"]["model_id"] = "C:/local/model"
    with pytest.raises(ValueError):
        export_evidence(invalid, include_public_model=True)


def test_outcome_and_missing_measurement_validation():
    good = export_evidence(record("oom"))
    assert good["measured"] == {}
    for changes in (
        {"units": "GB"},
        {"measured": {"peak_reserved_gb": 0}},
        {"success": True},
        {"measured": {"peak_reserved_gb": 1}},
        {"schema_version": 99},
    ):
        with pytest.raises(ValueError):
            SharedEvidence.model_validate({**good, **changes})


def test_report_denominators_exclude_marginal_unknown_community_and_fits():
    success = record()
    oom = record("oom")
    oom["workload_match_verified"] = True
    false_no = record(predicted="no")
    historical = record()
    historical["provenance"]["cohort"] = "historical-fit"
    summary = error_summary(
        [
            success,
            oom,
            false_no,
            record(predicted="marginal"),
            export_evidence(record()),
            historical,
        ]
    )
    assert summary["paired_successes"] == 3
    assert summary["predicted_yes_actual_oom"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert summary["predicted_no_actual_success"] == {"numerator": 1, "denominator": 1, "rate": 1}
    assert error_summary([])["mean_absolute_percentage_error"] is None


def test_failed_or_community_runs_cannot_fit_calibration():
    failed = record("oom")
    failed["measured"] = {"peak_reserved_gb": 3.5}
    assert _result_to_sample(failed) is None
    assert _result_to_sample(export_evidence(record())) is None


def test_cli_preview_validate_and_overwrite_protection(tmp_path):
    from typer.testing import CliRunner

    from canifinetune.cli import app

    source = tmp_path / "raw.json"
    source.write_text(json.dumps(record()), encoding="utf-8")
    target = tmp_path / "share.json"
    runner = CliRunner()
    preview = runner.invoke(app, ["evidence-export", "--input", str(source)])
    assert preview.exit_code == 0
    assert json.loads(preview.stdout)["provenance"]["review_status"] == "unreviewed"
    exported = runner.invoke(app, ["evidence-export", "--input", str(source), "--out", str(target)])
    assert exported.exit_code == 0
    before = target.read_bytes()
    validated = runner.invoke(app, ["evidence-validate", "--input", str(target)])
    assert validated.exit_code == 0
    duplicate = runner.invoke(
        app, ["evidence-export", "--input", str(source), "--out", str(target)]
    )
    assert duplicate.exit_code != 0 and target.read_bytes() == before
    target.write_text("{}", encoding="utf-8")
    assert runner.invoke(app, ["evidence-validate", "--input", str(target)]).exit_code != 0
