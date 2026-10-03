"""Real, offline training: selected explicitly in CPU/CUDA CI qualification."""

import json
import subprocess
import sys

import pytest

from canifinetune.recipes import RecipeRequest, generate_recipe
from canifinetune.training.data import encode_record, load_records
from canifinetune.training.runtime import RunConfig, evaluate, train

pytestmark = pytest.mark.training


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import AutoTokenizer

    from canifinetune.training.smoke import create_smoke_model

    model = create_smoke_model(tmp_path_factory.mktemp("local-model"))
    return model, AutoTokenizer.from_pretrained(model, local_files_only=True)


def test_multiturn_masks_eos_padding_and_native_tokens(tiny):
    _, tokenizer = tiny
    messages = [
        {"role": "system", "content": "Briefly."},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": "Again."},
        {"role": "assistant", "content": "Two."},
    ]
    native = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(native, "keys"):
        native = native["input_ids"]
    row, info = encode_record({"messages": messages}, tokenizer, 128, loss_mode="assistant")
    assert row["input_ids"][: len(native)] == native
    decoded = tokenizer.decode(row["input_ids"][: len(native)])
    for preserved in ("Briefly.", "Hello.", "One.", "Again.", "Two."):
        assert preserved in decoded
    assistant_ids = [
        i for i, label in zip(row["input_ids"], row["labels"], strict=False) if label != -100
    ]
    assert assistant_ids.count(tokenizer.eos_token_id) == 2
    assert "One." in tokenizer.decode(assistant_ids) and "Two." in tokenizer.decode(assistant_ids)
    assert row["labels"][len(native) :] == [-100] * (128 - len(native))
    assert info["supervised_tokens"] == sum(x != -100 for x in row["labels"][1:])


@pytest.mark.parametrize(
    "record",
    [
        {"messages": [{"role": "assistant", "content": "No user."}]},
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}]},
        {"messages": [{"role": "user", "content": "Hi"}], "text": "ambiguous"},
        {"instruction": "hi", "output": ""},
        {"instruction": "hi", "output": 1},
        {"text": "unsupported"},
    ],
)
def test_malformed_records_are_errors(tiny, record):
    with pytest.raises(ValueError):
        encode_record(record, tiny[1], 128)


def test_truncation_and_causal_shift_reject_lost_answer(tiny):
    record = {"instruction": "Hello " * 100, "output": "One."}
    with pytest.raises(ValueError, match="increase"):
        encode_record(record, tiny[1], 32)
    with pytest.raises(ValueError, match="no response token"):
        encode_record(record, tiny[1], 32, truncation="right")
    with pytest.raises(ValueError, match="causal"):
        encode_record({"instruction": "Hi", "output": "One"}, tiny[1], 1, truncation="right")


def test_template_without_generation_mask_is_rejected(tiny):
    _, tokenizer = tiny
    original = tokenizer.chat_template
    try:
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + eos_token }}{% endfor %}"
        )
        data = {
            "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "One"}]
        }
        encode_record(data, tokenizer, 128)
        with pytest.raises(ValueError, match="generation"):
            encode_record(data, tokenizer, 128, loss_mode="assistant")
    finally:
        tokenizer.chat_template = original


@pytest.mark.parametrize("method", ["full", "lora"])
def test_real_cpu_update_save_reload_and_generated_entrypoint(tmp_path, tiny, method):
    import torch
    from transformers import AutoModelForCausalLM

    model_path, _ = tiny
    recipe = tmp_path / "recipe"
    generate_recipe(
        RecipeRequest(
            model_id=str(model_path),
            method=method,
            seq_len=128,
            base_dtype="fp32",
            optimizer="adamw_torch",
            device="cpu",
            max_steps=2,
            gradient_accumulation_steps=1,
            local_files_only=True,
            output_dir=recipe,
        )
    )
    process = subprocess.run(
        [sys.executable, str(recipe / "train.py"), "--config", str(recipe / "config.yaml")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads((recipe / "output/run.json").read_text())
    assert record["status"] == "success" and record["completed_updates"] == 2
    assert record["effective_configuration"]["mixed_precision"] == "none"
    assert evaluate(recipe / "output", max_new_tokens=2)
    artifact = recipe / "output" / ("model" if method == "full" else "adapter")
    if method == "full":
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        updated = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True)
        assert any(
            not torch.equal(a, b)
            for a, b in zip(base.parameters(), updated.parameters(), strict=False)
        )
    else:
        from safetensors.torch import load_file

        parameters = load_file(str(artifact / "adapter_model.safetensors"))
        assert any(
            "lora_B" in name and torch.count_nonzero(weight) for name, weight in parameters.items()
        )
    retry = subprocess.run(
        [sys.executable, str(recipe / "train.py"), "--config", str(recipe / "config.yaml")],
        capture_output=True,
    )
    assert retry.returncode != 0
    assert (recipe / "output/run.json").read_text() == json.dumps(record, indent=2)


def test_failed_dataset_and_save_never_mark_success(tmp_path, tiny, monkeypatch):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"instruction":"Hi", "output":""}\n')
    cfg = RunConfig(
        model_id=str(tiny[0]),
        dataset_path=str(path),
        output_dir=str(tmp_path / "bad"),
        method="lora",
        base_dtype="fp32",
        optimizer="adamw_torch",
        device="cpu",
        max_steps=1,
        seq_len=128,
        local_files_only=True,
    )
    with pytest.raises(ValueError):
        train(cfg)
    assert json.loads((tmp_path / "bad/run.json").read_text())["status"] == "error"
    assert not (tmp_path / "bad/adapter").exists()
    path.write_text('{"instruction":"Hi", "output":"One."}\n')
    from peft import PeftModel

    def fail(*args, **kwargs):
        raise OSError("injected disk save failure")

    monkeypatch.setattr(PeftModel, "save_pretrained", fail)
    cfg.output_dir = str(tmp_path / "save-failed")
    with pytest.raises(OSError, match="save failure"):
        train(cfg)
    assert json.loads((tmp_path / "save-failed/run.json").read_text())["status"] == "error"
    assert not (tmp_path / "save-failed/adapter").exists()


def test_dataset_reports_line_and_statistics(tmp_path, tiny):
    path = tmp_path / "data.jsonl"
    path.write_text('\n{"instruction":"Hi", "output":"One."}\n')
    rows, stats = load_records(path, tiny[1], 128)
    assert len(rows) == 1 and stats["blank_lines"] == 1 and stats["dropped_rows"] == 0
    path.write_text('\n{"instruction":"Hi", "output":""}\n')
    with pytest.raises(ValueError, match="line 2"):
        load_records(path, tiny[1], 128)
