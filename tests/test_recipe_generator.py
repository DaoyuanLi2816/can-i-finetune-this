from __future__ import annotations

from pathlib import Path

import pytest

from canifinetune.recipes import RecipeRequest, generate_recipe


def test_recipe_files_are_created(tmp_path: Path):
    out = tmp_path / "recipe"
    req = RecipeRequest(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        method="qlora",
        seq_len=1024,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
        lora_rank=16,
        max_steps=3,
        output_dir=out,
    )
    res = generate_recipe(req)
    assert res.output_dir == out
    names = {p.name for p in res.files}
    expected = {
        "train.py",
        "config.yaml",
        "run.sh",
        "eval_smoke.py",
        "requirements.txt",
        "README.md",
        "expected_vram.md",
        "dataset_format.md",
        "troubleshooting.md",
        "sample.jsonl",
    }
    assert expected.issubset(names)


def test_recipe_yaml_round_trips(tmp_path: Path):
    out = tmp_path / "recipe"
    req = RecipeRequest(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        method="qlora",
        seq_len=512,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        lora_rank=8,
        max_steps=2,
        output_dir=out,
    )
    generate_recipe(req)
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["method"] == "qlora"
    assert cfg["seq_len"] == 512
    assert cfg["lora_rank"] == 8
    assert "q_proj" in cfg["target_modules"]


def test_recipe_train_script_compiles(tmp_path: Path):
    """The generated train.py should at least be syntactically valid Python."""
    out = tmp_path / "recipe"
    req = RecipeRequest(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        method="qlora",
        seq_len=128,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        lora_rank=8,
        max_steps=1,
        output_dir=out,
    )
    generate_recipe(req)
    src = (out / "train.py").read_text(encoding="utf-8")
    compile(src, str(out / "train.py"), "exec")


def test_recipe_includes_target_modules_for_family(tmp_path: Path):
    out = tmp_path / "recipe"
    req = RecipeRequest(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        method="qlora",
        seq_len=1024,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
        lora_rank=16,
        max_steps=1,
        output_dir=out,
    )
    generate_recipe(req)
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    # Llama family should expose q/k/v/o linear projections.
    assert {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(set(cfg["target_modules"]))
