"""Shared optional training runtime. No implicit dtype/backend fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from .. import __version__
from ..configuration import TrainingConfig, resolve_targets
from ..utils.logging import utc_now_iso
from .data import load_records


class RunConfig(TrainingConfig):
    output_dir: str = "output"
    dataset_path: str = "data/sample.jsonl"
    learning_rate: float = Field(2e-4, gt=0)
    max_steps: int = Field(50, gt=0)
    logging_steps: int = Field(1, gt=0)
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path):
        path = Path(path).resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config must be a YAML mapping")
        config = cls.model_validate(data)
        for name in ("dataset_path", "output_dir"):
            value = Path(getattr(config, name))
            if not value.is_absolute():
                setattr(config, name, str(path.parent / value))
        if Path(config.model_id).is_dir() is False and (path.parent / config.model_id).is_dir():
            config.model_id = str((path.parent / config.model_id).resolve())
        return config


def software_stack():
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "canifinetune": __version__,
    }
    fingerprint = hashlib.sha256()
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob("*.py")):
        fingerprint.update(str(path.relative_to(package)).replace("\\", "/").encode())
        fingerprint.update(path.read_bytes().replace(b"\r\n", b"\n"))
    result["source_fingerprint_sha256"] = fingerprint.hexdigest()
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "liger-kernel"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def failure_kind(exc):
    from ..bench.oom import is_oom

    if is_oom(exc):
        return "oom"
    if isinstance(exc, ImportError):
        return "dependency_error"
    if type(exc).__name__ in {
        "GatedRepoError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "HfHubHTTPError",
        "LocalEntryNotFoundError",
    }:
        return "access_error"
    if isinstance(exc, (ValueError, TypeError)):
        return "configuration_error"
    if exc.__cause__ is not None and exc.__cause__ is not exc:
        return failure_kind(exc.__cause__)
    return "runtime_error"


def validate_device(config, device):
    import torch

    if device == "cpu":
        if (
            config.method == "qlora"
            or "8bit" in config.optimizer
            or config.optimizer.startswith("paged")
        ):
            raise ValueError(
                "CPU qualification supports full/LoRA with adamw_torch or sgd; QLoRA/8-bit require CUDA"
            )
        if config.base_dtype != "fp32":
            raise ValueError(
                "CPU training requires explicit base_dtype=fp32; no automatic dtype fallback"
            )
        if config.use_liger_kernel or config.attention_implementation == "flash_attention_2":
            raise ValueError("Liger/Flash Attention require a qualified CUDA environment")
    elif device != "cuda":
        raise ValueError(
            "device must be cpu or cuda; secondary CUDA device selection is not qualified"
        )
    else:
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA unavailable; select device=cpu and fp32 for a tiny full/LoRA run"
            )
        torch.cuda.set_device(0)
        if config.base_dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError(
                "requested bf16 is unsupported; explicitly choose fp16 and re-estimate"
            )


def load_model(config: TrainingConfig, device="cuda"):
    import torch
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    validate_device(config, device)
    if config.base_dtype == "fp16" and config.method == "full":
        raise ValueError(
            "full fp16 weight training is unsupported by this runtime; choose bf16 or fp32"
        )
    kwargs = {
        "trust_remote_code": False,
        "revision": config.revision,
        "local_files_only": config.local_files_only,
        "attn_implementation": config.attention_implementation,
    }
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        config.base_dtype
    ]
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs[dtype_key] = dtype
    if config.method == "qlora":
        import bitsandbytes  # noqa: F401

        if config.quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="fp4" if config.quantization in {"fp4", "int4"} else "nf4",
                bnb_4bit_use_double_quant=config.quantization == "nf4_double_quant",
            )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    # A loader error is an error. Retrying after removing kwargs would change
    # the memory configuration and hide unrelated TypeErrors.
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    if config.method != "qlora":
        model = model.to(device)
    if config.method == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if config.quantization != "int8":
            from bitsandbytes.nn import Linear4bit

            for module in model.modules():
                if isinstance(module, Linear4bit):
                    # Qualified bitsandbytes 0.49 chooses precision again from
                    # the first input (often fp32 after PEFT preparation).
                    # Preserve the explicitly requested compute dtype instead.
                    if not hasattr(module, "compute_type_is_set"):
                        raise ValueError("bitsandbytes cannot preserve explicit compute precision")
                    module.compute_dtype = dtype
                    module.compute_type_is_set = True
    model.config.use_cache = False
    return model


def attach_adapter(model, config: TrainingConfig):
    if config.method != "full":
        from peft import LoraConfig, get_peft_model

        targets = resolve_targets(config, model.config.model_type)
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def effective_configuration(model, config, device):
    import torch

    effective = config.training_fields()
    expected_device = torch.device("cpu" if device == "cpu" else "cuda:0")
    if any(p.device != expected_device for p in model.parameters()):
        raise ValueError("model parameters differ from the requested execution device")
    effective["target_modules"] = resolve_targets(config, model.config.model_type)
    effective["device"] = device
    effective["resolved_revision"] = getattr(model.config, "_commit_hash", None)
    effective["model_family"] = model.config.model_type
    effective["parameter_dtypes"] = sorted(
        {str(p.dtype).replace("torch.", "") for p in model.parameters()}
    )
    effective["trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    actual_attention = getattr(model.config, "_attn_implementation", None)
    if actual_attention != config.attention_implementation:
        raise ValueError(
            f"attention mismatch: requested {config.attention_implementation}, loaded {actual_attention}"
        )
    effective["attention_implementation"] = actual_attention
    if bool(getattr(model, "is_gradient_checkpointing", False)) != config.gradient_checkpointing:
        raise ValueError("gradient checkpointing differs from requested configuration")
    effective["mixed_precision"] = config.base_dtype if device != "cpu" else "none"
    effective["quantization_compute_dtype"] = (
        config.base_dtype if config.method == "qlora" and config.quantization != "int8" else None
    )
    if config.method == "qlora":
        import bitsandbytes as bnb

        quantized = [
            module
            for module in model.modules()
            if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
        ]
        if not quantized:
            raise ValueError("QLoRA request did not load quantized linear layers")
        effective["quantized_linear_layers"] = len(quantized)
        if config.quantization != "int8":
            actual_compute = {
                str(module.compute_dtype).replace("torch.", "") for module in quantized
            }
            expected_compute = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[
                config.base_dtype
            ]
            if actual_compute != {expected_compute}:
                raise ValueError(f"quantization compute dtype mismatch: {actual_compute}")
    else:
        expected_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            config.base_dtype
        ]
        base_parameters = [p for name, p in model.named_parameters() if "lora_" not in name]
        if any(p.dtype != expected_dtype for p in base_parameters):
            raise ValueError("loaded base weights differ from requested dtype")
    effective["cuda_version"] = torch.version.cuda if device != "cpu" else None
    return effective


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def train(config: RunConfig):
    import torch
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )

    output = Path(config.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            "output_dir must be empty; choose a new directory (training never overwrites a run)"
        )
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 2,
        "status": "running",
        "timestamp": utc_now_iso(),
        "requested_configuration": config.model_dump(),
        "software": software_stack(),
        "estimate_version": "0.4.0",
        "units": "GiB",
    }
    _write_json(output / "run.json", record)
    try:
        set_seed(config.seed)
        validate_device(config, config.device)
        cuda = config.device != "cpu"
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            record["hardware"] = {
                "name": torch.cuda.get_device_name(),
                "total_gib": total / 2**30,
                "free_before_load_gib": free / 2**30,
                "free_memory_source": "torch.cuda.mem_get_info",
            }
            from ..utils.gpu import probe_cuda

            info = probe_cuda()
            if len(info.gpus) == 1:
                record["hardware"].update(info.gpus[0].to_dict())
                record["hardware"]["free_before_load_gib"] = info.gpus[0].free_vram_gb
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            local_files_only=config.local_files_only,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs PAD or EOS")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        rows, statistics = load_records(
            config.dataset_path,
            tokenizer,
            config.seq_len,
            loss_mode=config.loss_mode,
            truncation=config.truncation,
        )
        record["dataset_statistics"] = statistics
        model = attach_adapter(load_model(config, config.device), config)
        record["effective_configuration"] = effective_configuration(model, config, config.device)
        _write_json(output / "run.json", record)
        print(
            json.dumps(
                {
                    "effective_configuration": record["effective_configuration"],
                    "dataset_statistics": statistics,
                },
                indent=2,
            )
        )
        optimizer_name = "adamw_torch" if config.optimizer == "adamw" else config.optimizer
        arguments = TrainingArguments(
            output_dir=str(output / "trainer"),
            per_device_train_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            logging_steps=config.logging_steps,
            save_strategy="no",
            report_to=[],
            bf16=config.base_dtype == "bf16",
            fp16=config.base_dtype == "fp16",
            use_cpu=not cuda,
            optim=optimizer_name,
            seed=config.seed,
            gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            dataloader_num_workers=0,
            use_liger_kernel=config.use_liger_kernel,
        )
        trainer = Trainer(
            model=model, args=arguments, train_dataset=rows, data_collator=default_data_collator
        )
        outcome = trainer.train()
        record["effective_configuration"] = effective_configuration(model, config, config.device)
        actual_optimizer: Any = trainer.optimizer
        if actual_optimizer is None:
            raise RuntimeError("training completed without an optimizer")
        while hasattr(actual_optimizer, "optimizer"):
            actual_optimizer = actual_optimizer.optimizer
        record["effective_configuration"]["optimizer_class"] = (
            f"{type(actual_optimizer).__module__}.{type(actual_optimizer).__name__}"
        )
        record["effective_configuration"]["optimizer_state_dtypes"] = sorted(
            {
                str(value.dtype).replace("torch.", "")
                for state in actual_optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            }
        )
        if trainer.state.global_step != config.max_steps or not torch.isfinite(
            torch.tensor(outcome.training_loss)
        ):
            raise RuntimeError("training did not complete the requested updates with finite loss")
        if cuda:
            torch.cuda.synchronize()
            record["measured"] = {
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                "scope": "process torch allocator: load through final optimizer update",
            }
        record["completed_updates"] = trainer.state.global_step
        record["training_loss"] = outcome.training_loss
        destination = "model" if config.method == "full" else "adapter"
        partial = output / ".saving"
        model.save_pretrained(partial)
        tokenizer.save_pretrained(partial)
        os.replace(partial, output / destination)
        record["saved_artifact"] = destination
        record["status"] = "success"
        record["resume_supported"] = False
        _write_json(output / "run.json", record)
        print(f"training finished: {config.max_steps} updates; saved {destination}")
        return record
    except Exception as exc:
        from ..bench.oom import is_oom

        record["status"] = "oom" if is_oom(exc) else "error"
        record["failure_kind"] = failure_kind(exc)
        record["error_type"] = type(exc).__name__
        # Keep details local; community export uses a separate allowlist.
        record["error"] = str(exc)
        if (output / ".saving").exists():
            shutil.rmtree(output / ".saving")
        _write_json(output / "run.json", record)
        raise


def recipe_main():
    parser = argparse.ArgumentParser(
        description="Train the generated recipe using canifinetune's shared runtime"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    try:
        config = RunConfig.from_yaml(args.config)
        updates = {
            name: value
            for name, value in {
                "max_steps": args.max_steps,
                "dataset_path": args.dataset,
                "output_dir": args.output_dir,
                "device": args.device,
            }.items()
            if value is not None
        }
        config = RunConfig.model_validate({**config.model_dump(), **updates})
        train(config)
    except ImportError as exc:
        print(
            f"Missing training dependency: {exc}. Install canifinetune[train] in this environment."
        )
        return 2
    except Exception as exc:
        from ..bench.oom import is_oom

        print(f"Training failed ({type(exc).__name__}): {exc}")
        return 3 if is_oom(exc) else 2
    return 0


def evaluate(output_dir, prompt="Hello", max_new_tokens=8):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(output_dir)
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if record["status"] != "success":
        raise ValueError("evaluation requires a successfully saved run")
    config = RunConfig.model_validate(record["requested_configuration"])
    artifact = output / record["saved_artifact"]
    tokenizer = AutoTokenizer.from_pretrained(
        artifact, trust_remote_code=False, local_files_only=True
    )
    if config.method == "full":
        model = AutoModelForCausalLM.from_pretrained(
            artifact, local_files_only=True, trust_remote_code=False
        ).to(config.device)
    else:
        from peft import PeftModel

        # Reload the same quantized base and revision, never silently ignore an adapter.
        revision = record["effective_configuration"].get("resolved_revision")
        if revision:
            config.revision = revision
        model = PeftModel.from_pretrained(load_model(config, config.device), artifact)
    model.eval()
    model.config.use_cache = True
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def eval_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    try:
        print(evaluate(args.output_dir, args.prompt, args.max_new_tokens))
    except Exception as exc:
        print(f"Reload failed ({type(exc).__name__}): {exc}")
        return 2
    return 0
