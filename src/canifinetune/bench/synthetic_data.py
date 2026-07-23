"""Generate a tiny synthetic causal-LM batch for benchmark runs.

The runner only needs shape-realistic tokens to measure VRAM, so this module
fabricates them in memory instead of pulling a dataset off the Hub.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SyntheticBatch:
    input_ids: object  # torch.LongTensor
    attention_mask: object  # torch.LongTensor
    labels: object  # torch.LongTensor


def make_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
    seed: int = 0,
) -> SyntheticBatch:
    """Produce one synthetic batch on ``device``.

    The tokens are uniform random in ``[0, vocab_size)``. Labels equal inputs
    so the loss computation has something to chew on; for QLoRA we mask the
    pad token via the attention mask (no pad tokens here, all-ones mask).
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=max(1, vocab_size),
        size=(batch_size, seq_len),
        dtype=torch.long,
        generator=g,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    if device.startswith("cuda"):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    return SyntheticBatch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def make_text_dataset(num_rows: int, seq_len_chars: int = 256) -> list[dict[str, str]]:
    """A trivial instruction dataset for recipe smoke tests.

    Every row is the same template, varying only by an index.
    """
    rows = []
    for i in range(num_rows):
        instruction = "Summarize the following sentence in one short sentence."
        text = (
            f"Sample number {i}: open-source language models can be fine-tuned "
            "with parameter-efficient methods like LoRA and QLoRA on consumer "
            "GPUs when memory is managed carefully."
        )
        response = f"Sample {i}: open LLMs can be fine-tuned with LoRA/QLoRA on consumer GPUs."
        rows.append(
            {
                "instruction": instruction[:seq_len_chars],
                "input": text[:seq_len_chars],
                "output": response[:seq_len_chars],
            }
        )
    return rows
