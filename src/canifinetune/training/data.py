"""One explicit JSONL/tokenization/causal-supervision contract, without torch."""

from __future__ import annotations

import json
from pathlib import Path


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    expected = "user"
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("each message must have exactly role and content")
        role, content = message["role"], message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")
        if i == 0 and role == "system":
            continue
        if role != expected:
            raise ValueError("expected optional initial system, then alternating user/assistant")
        expected = "assistant" if expected == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError("training conversations must end with an assistant answer")
    return messages


def encode_record(row, tokenizer, seq_len, *, loss_mode="all", truncation="error"):
    if not isinstance(row, dict):
        raise ValueError("record must be a JSON object")
    if loss_mode not in {"all", "assistant"} or truncation not in {"error", "right"}:
        raise ValueError("invalid loss_mode or truncation")
    response_mask = None
    if "messages" in row:
        if set(row) != {"messages"}:
            raise ValueError("messages cannot be mixed with text/instruction fields")
        messages = validate_messages(row["messages"])
        if not tokenizer.chat_template:
            raise ValueError("messages require the model tokenizer's native chat_template")
        needs_mask = loss_mode == "assistant" or truncation == "right"
        # The tokenizer maps generation spans to tokens in the complete text.
        # Never infer boundaries by separately tokenizing a prompt prefix.
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=needs_mask,
        )
        ids = list(encoded["input_ids"])
        if needs_mask:
            response_mask = list(encoded.get("assistant_masks", []))
            if len(response_mask) != len(ids) or not any(response_mask):
                raise ValueError(
                    "assistant loss/right chat truncation requires native {% generation %} "
                    "spans and a tokenizer with working assistant masks; use all loss with "
                    "truncation=error or an explicitly reviewed model template"
                )
            reference = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            if hasattr(reference, "keys"):
                reference = reference["input_ids"]
            if list(reference) != ids or any(m not in (0, 1) for m in response_mask):
                raise ValueError("chat generation masks do not match native tokenization")
    else:
        if not {"instruction", "output"}.issubset(row) or set(row) - {
            "instruction",
            "input",
            "output",
        }:
            raise ValueError("instruction records require instruction/output and optional input")
        instruction, answer, input_text = row["instruction"], row["output"], row.get("input", "")
        if any(not isinstance(x, str) for x in (instruction, answer, input_text)):
            raise ValueError("instruction/input/output must be strings")
        if not instruction.strip() or not answer.strip():
            raise ValueError("instruction and answer must not be empty")
        if not tokenizer.is_fast:
            raise ValueError("instruction format requires a fast tokenizer for response offsets")
        prompt = f"### Instruction:\n{instruction}\n\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n\n"
        prompt += "### Response:\n"
        encoded = tokenizer(prompt + answer, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        # Tokens crossing the boundary are masked in assistant-only mode.
        response_mask = [
            int(start >= len(prompt) and end > start) for start, end in encoded["offset_mapping"]
        ]
        if tokenizer.eos_token_id is None:
            raise ValueError("instruction format requires eos_token_id")
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
            response_mask.append(1)
    original_length = len(ids)
    if original_length > seq_len:
        if truncation == "error":
            raise ValueError(
                f"record has {original_length} tokens > seq_len={seq_len}; increase it or opt into right truncation"
            )
        ids = ids[:seq_len]
        response_mask = response_mask[:seq_len] if response_mask is not None else None
    if response_mask is not None and not any(response_mask[1 : len(ids)]):
        raise ValueError("no response token remains after the causal shift/truncation")
    labels = [
        token if loss_mode == "all" or response_mask[i] else -100 for i, token in enumerate(ids)
    ]
    # Causal LM predicts label[t] using logits[t-1]; label[0] cannot train.
    supervised = sum(label != -100 for label in labels[1:])
    if not supervised:
        raise ValueError("record has no effective supervised token after causal shift")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs a PAD or EOS id")
    pad = seq_len - len(ids)
    return {
        "input_ids": ids + [pad_id] * pad,
        "attention_mask": [1] * len(ids) + [0] * pad,
        "labels": labels + [-100] * pad,
    }, {
        "original_tokens": original_length,
        "truncated": int(original_length > seq_len),
        "supervised_tokens": supervised,
    }


def load_records(path: str | Path, tokenizer, seq_len, *, loss_mode="all", truncation="error"):
    rows = []
    stats = {
        "rows": 0,
        "blank_lines": 0,
        "truncated_rows": 0,
        "supervised_tokens": 0,
        "dropped_rows": 0,
    }
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            try:
                encoded, info = encode_record(
                    json.loads(line), tokenizer, seq_len, loss_mode=loss_mode, truncation=truncation
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"dataset line {line_number}: {exc}") from exc
            rows.append(encoded)
            stats["rows"] += 1
            stats["truncated_rows"] += info["truncated"]
            stats["supervised_tokens"] += info["supervised_tokens"]
    if not rows:
        raise ValueError("dataset contains no training records")
    return rows, stats
