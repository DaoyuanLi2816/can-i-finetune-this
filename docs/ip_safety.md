# IP / data safety

`canifinetune` is intentionally a **general LLM fine-tuning infrastructure**
project. It is not a recommender system, not a retrieval-for-recommendation
toolkit, not an ads-ranking toolkit, and not specific to any single company's
domain.

## What is *not* in this repository

- Any recommender system / ranking / CTR / CVR / candidate-generation /
  user-profiling / feed / ads logic.
- Any proprietary models, datasets, prompts, training corpora, or evaluation
  sets.
- Any company-internal terminology, code, or pipeline artifacts.
- Any credentials, tokens, internal URLs, or service identifiers.

If you are a contributor coming from a company that fine-tunes LLMs in
production, **please do not paste internal recipes, configs, prompts, dataset
samples, model checkpoints, or evaluation harness code into this repository**.
Use public analogues from the Hugging Face Hub, public papers, or your own
side projects.

## Public sources used

- **Models referenced in the curated metadata table**: each entry's
  architecture numbers (`hidden_size`, `num_hidden_layers`, etc.) come from
  the model's published `config.json` on the Hugging Face Hub at the time of
  writing. Citations are inline in `src/canifinetune/estimator/model_metadata.py`.
- **Memory-model formulas**: based on the published Megatron-LM activation
  derivations (Korthikanti et al., 2022) and the bitsandbytes / QLoRA
  documentation. See `docs/memory_model.md` for citations.
- **Example datasets**: only synthetic, human-written sentences generated for
  this project ship in `data/sample.jsonl`. No third-party corpus is bundled.

## License caveats

- This repo is MIT licensed. That covers our code only.
- Each model referenced in the curated table has its **own license**. Notably:
  - `meta-llama/*` requires accepting Meta's license and is gated on Hugging Face.
  - `Qwen/*` and `mistralai/*` have their own terms; check before commercial use.
  - `microsoft/phi-*` are MIT-licensed.
  - `sshleifer/tiny-gpt2` is Apache-2 and intended for smoke testing only.
- Dataset licenses: if you bring your own JSONL into a recipe, the license of
  that dataset is yours to track. `canifinetune` does not assume anything.

## Personal-side-project safety checklist

Before contributing:

1. The change is in public LLM infra territory (memory accounting, training
   tooling, recipe ergonomics, docs).
2. No file you add or modify contains content derived from your employer.
3. Any new model added to the curated table has only public information —
   typically just what's in the model's `config.json` on the Hub.
4. Any benchmark result you add was produced on your *personal* hardware and
   uses *only* public models and datasets.

If you're unsure, open a draft PR with a brief note in the description, and
the maintainer will help you scope the change.
