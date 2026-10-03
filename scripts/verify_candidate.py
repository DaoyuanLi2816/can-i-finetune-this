"""Verify candidate hashes, source correspondence and explicit qualification receipts."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(dist, tag, commit):
    from canifinetune import __version__

    dist = Path(dist)
    manifest = json.loads((dist / "candidate.json").read_text(encoding="utf-8"))
    assert manifest["version"] == __version__ == tag.removeprefix("v")
    assert manifest["commit"] == commit
    assert manifest["qualification"]["core"] == "passed"
    assert manifest["qualification"]["cpu"] == "passed"
    assert manifest["qualification"]["cuda_lora"] == "passed"
    assert manifest["qualification"]["cuda_qlora"] == "passed"
    files = manifest["sha256"]
    assert set(files) == {
        f"canifinetune-{__version__}-py3-none-any.whl",
        f"canifinetune-{__version__}.tar.gz",
    }
    for name, sha in files.items():
        assert digest(dist / name) == sha, name
    receipt_spec = manifest["qualification_receipt"]
    assert receipt_spec["file"] == "qualification.json"
    assert digest(dist / receipt_spec["file"]) == receipt_spec["sha256"]
    receipts = json.loads((dist / receipt_spec["file"]).read_text(encoding="utf-8"))
    from canifinetune.training.runtime import software_stack

    for key in ("core", "training_recommended", "training_min"):
        assert receipts[key]["qualified"] == __version__
        assert (
            receipts[key]["software"]["source_fingerprint_sha256"]
            == software_stack()["source_fingerprint_sha256"]
        )
    required = {("full", "cpu"), ("lora", "cpu"), ("lora", "cuda"), ("qlora", "cuda")}
    runs = receipts["training_recommended"]["runs"]
    assert {
        (run["effective_configuration"]["method"], run["effective_configuration"]["device"])
        for run in runs
    } == required
    assert all(run["status"] == "success" and run["completed_updates"] == 2 for run in runs)
    assert receipts["sdist_rebuild"] == "passed"
    wheel = dist / f"canifinetune-{__version__}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        source = Path("src/canifinetune")
        expected = {
            "canifinetune/" + str(path.relative_to(source)).replace("\\", "/"): path
            for path in source.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        packaged = {
            name
            for name in archive.namelist()
            if name.startswith("canifinetune/") and not name.endswith("/")
        }
        assert packaged == set(expected), packaged.symmetric_difference(expected)
        for name, path in expected.items():
            # Git normalizes text line endings; the wheel is built from source
            # with LF in every tracked package resource.
            assert archive.read(name) == path.read_bytes(), name
    print(
        json.dumps({"candidate_verified": __version__, "commit": commit, "sha256": files}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    verify(args.dist, args.tag, args.commit)
