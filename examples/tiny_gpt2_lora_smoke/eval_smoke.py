"""Tiny eval: load the saved adapter and generate a single short completion.

Use this after `python train.py` finishes; it confirms the adapter loads and
the model can still produce text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--adapter", default="output_smoke/adapter")
    parser.add_argument(
        "--prompt",
        default="### Instruction:\nWrite one short sentence about open-source LLMs.\n\n### Response:\n",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # transformers 5.0 renamed torch_dtype -> dtype; try the new name first.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=dtype, device_map="auto"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype, device_map="auto"
        )

    adapter = Path(args.adapter)
    if adapter.is_dir():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter))
        except Exception as e:
            print(f"[warn] could not attach adapter at {adapter}: {e}")

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    print("--- prompt ---")
    print(args.prompt)
    print("--- generation ---")
    print(text[len(args.prompt):])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
