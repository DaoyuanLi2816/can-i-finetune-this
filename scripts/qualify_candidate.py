"""Run against an installed candidate, outside a checkout; no model downloads."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def qualify(training=False, cuda=False, receipt=None):
    import canifinetune
    from canifinetune.demo import demo_estimate
    from canifinetune.estimator.memory import EstimateRequest, estimate
    from canifinetune.recipes import RecipeRequest, generate_recipe
    from canifinetune.training.runtime import software_stack

    runs = []

    assert "site-packages" in str(Path(canifinetune.__file__).resolve()), canifinetune.__file__
    metadata = importlib.metadata.metadata("canifinetune")
    assert "Daoyuan Li" in metadata["Author-email"]
    assert metadata["Version"] == canifinetune.__version__
    result = estimate(
        EstimateRequest(model_id="Qwen/Qwen2.5-1.5B-Instruct", gpu_vram_gb=16, use_network=False)
    )
    assert result.memory.total_estimated_gb > 0
    assert (
        demo_estimate({"model_id": "Qwen/Qwen2.5-1.5B-Instruct"})["memory"]
        == result.memory.model_dump()
    )
    subprocess.run([sys.executable, "-m", "canifinetune.cli", "--version"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "canifinetune.cli",
            "recommend",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--gpu-vram-gb",
            "16",
            "--offline",
            "--top-k",
            "1",
            "--json",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with tempfile.TemporaryDirectory(prefix="canifinetune-qualification-") as temporary:
        root = Path(temporary)
        generate_recipe(
            RecipeRequest(model_id="Qwen/Qwen2.5-0.5B-Instruct", output_dir=root / "core-recipe")
        )
        for path in (root / "core-recipe").glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"))
        if training:
            from canifinetune.training.smoke import create_smoke_model

            model = create_smoke_model(root / "tiny-model")
            cases = [("full", "cpu"), ("lora", "cpu")]
            if cuda:
                cases += [("lora", "cuda"), ("qlora", "cuda")]
            for method, device in cases:
                if device == "cuda":
                    from canifinetune.utils.gpu import probe_gpus_via_nvidia_smi

                    gpus = probe_gpus_via_nvidia_smi()
                    if len(gpus) != 1 or gpus[0].free_vram_gb < 1:
                        raise SystemExit(
                            "not run: tiny CUDA qualification requires at least 1 GiB free on one GPU"
                        )
                recipe = root / f"{method}-{device}"
                generate_recipe(
                    RecipeRequest(
                        model_id=str(model),
                        method=method,
                        device=device,
                        base_dtype="fp32" if device == "cpu" else "bf16",
                        optimizer="paged_adamw_8bit" if method == "qlora" else "adamw_torch",
                        # Tiny dimensions need >=4096-element adapter tensors
                        # to exercise actual 8-bit optimizer state allocation.
                        lora_rank=128 if method == "qlora" else 16,
                        seq_len=128,
                        max_steps=2,
                        gradient_accumulation_steps=1,
                        local_files_only=True,
                        output_dir=recipe,
                    )
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(recipe / "train.py"),
                        "--config",
                        str(recipe / "config.yaml"),
                    ],
                    cwd=root,
                    check=True,
                )
                run = json.loads((recipe / "output/run.json").read_text(encoding="utf-8"))
                assert run["status"] == "success" and run["completed_updates"] == 2
                assert run["effective_configuration"]["method"] == method
                if method == "qlora":
                    assert "uint8" in run["effective_configuration"]["optimizer_state_dtypes"]
                import torch

                artifact = recipe / "output" / run["saved_artifact"]
                if method == "full":
                    from transformers import AutoModelForCausalLM

                    original = AutoModelForCausalLM.from_pretrained(model, local_files_only=True)
                    updated = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True)
                    changed = sum(
                        not torch.equal(a, b)
                        for a, b in zip(original.parameters(), updated.parameters(), strict=True)
                    )
                    del original, updated
                else:
                    from safetensors.torch import load_file

                    weights = load_file(str(artifact / "adapter_model.safetensors"))
                    changed = sum(
                        int(torch.count_nonzero(value)) > 0
                        for name, value in weights.items()
                        if "lora_B" in name
                    )
                assert changed > 0, "optimizer steps did not change saved parameters"
                runs.append(
                    {
                        "status": run["status"],
                        "completed_updates": run["completed_updates"],
                        "changed_parameter_tensors": changed,
                        "effective_configuration": {
                            key: value
                            for key, value in run["effective_configuration"].items()
                            if key != "model_id"
                        },
                        "measured": run.get("measured", {}),
                        "hardware": run.get("hardware", {}),
                    }
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(recipe / "eval_smoke.py"),
                        "--output-dir",
                        str(recipe / "output"),
                        "--max-new-tokens",
                        "2",
                    ],
                    cwd=root,
                    check=True,
                )
    result = {
        "qualified": canifinetune.__version__,
        "training": training,
        "cuda": cuda,
        "import_source": canifinetune.__file__,
        "software": software_stack(),
        "runs": runs,
    }
    if receipt:
        # Public receipt omits local import paths and dataset/model directories.
        public = {key: value for key, value in result.items() if key != "import_source"}
        with Path(receipt).open("x", encoding="utf-8") as handle:
            json.dump(public, handle, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--receipt", type=Path)
    options = parser.parse_args()
    qualify(options.training, options.cuda, options.receipt)
