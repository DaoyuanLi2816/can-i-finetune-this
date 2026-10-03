"""Create a tiny random causal model/tokenizer locally; never download weights."""

from pathlib import Path


def create_smoke_model(output: str | Path):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("smoke model output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [
            "Hello world. A tiny local training example. ### Instruction: Write a sentence. ### Response: Open source models.",
            "system user assistant: Answer briefly. One two three. input output abcdefghijklmnopqrstuvwxyz",
        ],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<eos>", "<bos>"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' }}{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] + eos_token }}{% endgeneration %}{% else %}{{ m['content'] + '\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(output)
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(output)
    return output
