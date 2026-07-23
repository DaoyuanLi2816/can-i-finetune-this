"""Pure-math helpers for the memory estimator.

Everything here is a function on numbers — no torch, no HTTP, no I/O. Tests
exercise this module directly so the heuristics are reproducible.

References:
  * Megatron-LM activation-memory equations, "Reducing Activation Recomputation
    in Large Transformer Models" (Korthikanti et al., 2022). Used as the
    skeleton of the per-layer activation formula, with coefficients re-fitted
    against real ``torch.cuda.max_memory_allocated()`` traces (see below).
  * Hugging Face transformers + PEFT default LoRA target modules per
    architecture family.
  * bitsandbytes documentation for NF4 block size and double-quantization
    overhead.
  * PEFT ``prepare_model_for_kbit_training``, which upcasts every
    non-quantized parameter (embeddings, lm_head, norms) to fp32. Measured:
    Qwen2.5-1.5B loads at 1.51 GiB after prepare (0.61 GiB packed 4-bit +
    0.87 GiB fp32 tied embedding), Qwen2.5-7B at 7.2 GiB (3.04 GiB packed +
    2 x 2.03 GiB fp32 untied embedding/lm_head).

Coefficients were calibrated against measured peaks on an RTX 4080 (16 GB,
torch 2.6 / transformers 5.8 / peft 0.19 / bitsandbytes 0.49) across
Qwen2.5-0.5B/1.5B/3B/7B, seq_len 512-4096, batch 1-2, ckpt on/off, LoRA and
QLoRA. The static component of each estimate lands within about +-10% of
``max_memory_allocated`` on those runs; the formulas intentionally err
slightly toward over-estimation so the "feasible" verdict is conservative
on consumer cards.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.units import dtype_bytes


@dataclass(frozen=True)
class ArchHints:
    """Shape numbers needed for the activation formula."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    # Whether lm_head shares storage with the input embedding. Untied models
    # (Llama 3, Mistral, Qwen2.5-7B, ...) pay for the embedding matrix twice.
    tie_word_embeddings: bool = True
    # Mixture-of-experts metadata. Dense models use one expert, so these
    # defaults preserve the pre-0.3 behavior.
    num_local_experts: int = 1
    num_experts_per_tok: int = 1


# Families whose MLP is a classic 2-matmul block (fc1 -> act -> fc2) rather
# than the 3-matmul SwiGLU (gate/up/down) used by llama-likes.
CLASSIC_MLP_FAMILIES = {"gpt2", "phi", "opt", "bloom", "gpt_neox", "falcon"}


def _is_swiglu(family: str) -> bool:
    return (family or "").lower() not in CLASSIC_MLP_FAMILIES


# ---------------------------------------------------------------------------
# Weight memory
# ---------------------------------------------------------------------------


def embedding_params(arch: ArchHints) -> int:
    """Parameters stored at full precision in a k-bit quantized model: the
    input embedding, the lm_head when untied, and the (tiny) norm weights."""
    copies = 1 if arch.tie_word_embeddings else 2
    embeds = arch.vocab_size * arch.hidden_size * copies
    norms = arch.num_hidden_layers * 2 * arch.hidden_size + arch.hidden_size
    return embeds + norms


def weights_bytes(
    *,
    num_params: int,
    base_dtype: str,
    quantization: str | None,
    arch: ArchHints | None = None,
    kbit_upcast_fp32: bool = True,
) -> float:
    """Bytes used to store the *base* model weights at rest.

    ``quantization`` overrides ``base_dtype`` when set. For 4/8-bit schemes,
    only the transformer Linear layers are actually quantized by bitsandbytes;
    embeddings / lm_head / norms stay in full precision — and PEFT's
    ``prepare_model_for_kbit_training`` upcasts them to fp32
    (``kbit_upcast_fp32``). When ``arch`` is unknown we fall back to the old
    all-params-quantized lower bound.
    """
    if quantization and quantization.lower() not in {"none", "fp16", "bf16", "fp32"}:
        q = quantization.lower()
        bytes_per_param = dtype_bytes(_quant_storage_dtype(q))
        if arch is None:
            return num_params * bytes_per_param
        full_precision = min(num_params, embedding_params(arch))
        linear = max(0, num_params - full_precision)
        fp_bytes = 4.0 if kbit_upcast_fp32 else dtype_bytes(base_dtype)
        return linear * bytes_per_param + full_precision * fp_bytes
    return num_params * dtype_bytes(base_dtype)


def _quant_storage_dtype(quantization: str) -> str:
    if quantization in {"int8", "fp8"}:
        return "int8"
    if quantization in {"nf4", "fp4", "int4", "nf4_double_quant"}:
        return "nf4"
    raise ValueError(f"Unknown quantization {quantization!r}")


def dequant_workspace_bytes(*, arch: ArchHints, quantization: str | None) -> float:
    """Transient buffers bitsandbytes uses to dequantize one layer's weights.

    Each 4-bit matmul materializes a bf16 copy of the weight tile; the largest
    resident copy is the MLP projection (hidden x intermediate), and forward +
    backward can hold two of them briefly. Zero for non-quantized runs.
    """
    if not quantization or quantization.lower() in {"none", "fp16", "bf16", "fp32"}:
        return 0.0
    return 2.0 * arch.hidden_size * arch.intermediate_size * 2.0


# ---------------------------------------------------------------------------
# LoRA / QLoRA trainable parameter counting
# ---------------------------------------------------------------------------


def _lora_params_for_linear(in_dim: int, out_dim: int, rank: int) -> int:
    """LoRA parameters added by ``Linear[in,out]`` with rank ``rank``.

    A LoRA adapter on a Linear layer adds ``A: [rank, in]`` and ``B: [out, rank]``.
    Some configurations also keep a separate scalar ``alpha`` / ``dropout`` which
    contribute negligibly.
    """
    return rank * (in_dim + out_dim)


# Default target modules per HF architecture, mirroring PEFT defaults.
# "attention" subset is the common conservative choice.
TARGET_MODULES_BY_FAMILY: dict[str, dict[str, list[str]]] = {
    "llama": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "conservative": ["q_proj", "v_proj"],
    },
    "qwen2": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "conservative": ["q_proj", "v_proj"],
    },
    "mistral": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "conservative": ["q_proj", "v_proj"],
    },
    "phi": {
        "attention": ["q_proj", "k_proj", "v_proj", "dense"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        "conservative": ["q_proj", "v_proj"],
    },
    "phi3": {
        "attention": ["qkv_proj", "o_proj"],
        "all_linear": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "conservative": ["qkv_proj"],
    },
    "gpt2": {
        "attention": ["c_attn", "c_proj"],
        "all_linear": ["c_attn", "c_proj", "c_fc"],
        "conservative": ["c_attn"],
    },
    "gemma": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "conservative": ["q_proj", "v_proj"],
    },
    "mixtral": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
        "conservative": ["q_proj", "v_proj"],
    },
}


def _canonical_target_family(family: str) -> str:
    fam = (family or "").lower()
    if fam in TARGET_MODULES_BY_FAMILY:
        return fam
    if fam.startswith("qwen"):
        return "qwen2"
    if fam.startswith("llama"):
        return "llama"
    if fam.startswith("gemma"):
        return "gemma"
    if fam.startswith("phi3"):
        return "phi3"
    if fam.startswith("phi"):
        return "phi"
    if "mixtral" in fam:
        return "mixtral"
    return fam


def default_target_modules(
    family: str,
    scope: str = "attention",
    *,
    strict: bool = False,
) -> list[str]:
    """Get default target_modules for a model family ('attention'/'all_linear'/'conservative')."""
    fam = _canonical_target_family(family)
    table = TARGET_MODULES_BY_FAMILY.get(fam)
    if table is None:
        if strict:
            raise ValueError(
                f"No verified LoRA target-module mapping for model family {family!r}. "
                "Use a supported family or provide target modules explicitly."
            )
        table = TARGET_MODULES_BY_FAMILY["llama"]
    if scope not in table:
        scope = "attention"
    return list(table[scope])


def lora_trainable_params(
    *,
    arch: ArchHints,
    family: str,
    rank: int,
    scope: str = "attention",
    include_lm_head: bool = False,
) -> int:
    """Estimate LoRA adapter parameter count.

    The accounting walks each layer once per target-module name. For GQA models
    we use ``num_key_value_heads`` to size the K/V projections.
    """
    h = arch.hidden_size
    ff = arch.intermediate_size
    kv_dim = max(
        1,
        (arch.hidden_size // max(1, arch.num_attention_heads)) * arch.num_key_value_heads,
    )

    targets = default_target_modules(family, scope=scope)
    per_layer = 0
    for name in targets:
        if name == "c_proj" and family.lower() == "gpt2":
            # PEFT matches by module suffix. GPT-2 has both attn.c_proj
            # (h -> h) and mlp.c_proj (ff -> h), so one target name attaches
            # adapters to both modules.
            per_layer += _lora_params_for_linear(h, h, rank)
            per_layer += _lora_params_for_linear(ff, h, rank)
        elif name in {"q_proj", "o_proj", "c_proj", "dense"}:
            per_layer += _lora_params_for_linear(h, h, rank)
        elif name == "qkv_proj":
            per_layer += _lora_params_for_linear(h, h + 2 * kv_dim, rank)
        elif name == "gate_up_proj":
            per_layer += arch.num_local_experts * _lora_params_for_linear(h, 2 * ff, rank)
        elif name in {"k_proj", "v_proj"}:
            per_layer += _lora_params_for_linear(h, kv_dim, rank)
        elif name in {"gate_proj", "up_proj", "fc1", "w1", "w3"}:
            copies = arch.num_local_experts
            per_layer += copies * _lora_params_for_linear(h, ff, rank)
        elif name in {"down_proj", "fc2", "w2"}:
            copies = arch.num_local_experts
            per_layer += copies * _lora_params_for_linear(ff, h, rank)
        elif name == "c_attn":
            # GPT-2 fuses q,k,v into one Conv1D of shape (h, 3h).
            per_layer += _lora_params_for_linear(h, 3 * h, rank)
        elif name == "c_fc":
            per_layer += _lora_params_for_linear(h, ff, rank)
        else:
            # Unknown linear name; charge a square h x h adapter to stay safe.
            per_layer += _lora_params_for_linear(h, h, rank)

    total = per_layer * arch.num_hidden_layers
    if include_lm_head:
        total += _lora_params_for_linear(h, arch.vocab_size, rank)
    return int(total)


def adapter_weights_bytes(*, trainable_params: int) -> float:
    """Bytes for the LoRA adapter weights themselves.

    PEFT keeps adapters in fp32 on quantized bases (``prepare_model_for_
    kbit_training``) and they are tiny either way, so charge 4 B/param.
    """
    return trainable_params * 4.0


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def gradients_bytes(*, trainable_params: int, grad_dtype: str = "fp32") -> float:
    """Gradient buffer bytes. With LoRA, gradients exist only for adapters
    (which PEFT keeps in fp32, hence the fp32 default)."""
    return trainable_params * dtype_bytes(grad_dtype)


# ---------------------------------------------------------------------------
# Optimizer states
# ---------------------------------------------------------------------------

# Bytes per *trainable* parameter for the most common optimizers.
OPTIMIZER_BYTES_PER_PARAM: dict[str, float] = {
    # AdamW master weights (fp32) + m + v in fp32 = 12 B/param. PyTorch's default
    # AdamW in mixed precision keeps fp32 m/v plus fp32 master weights of the
    # *trainable* params. With LoRA the trainable set is tiny so this is fine.
    "adamw_torch": 12.0,
    "adamw_torch_fused": 12.0,
    "adamw": 12.0,
    # 8-bit AdamW from bitsandbytes: m + v in int8 + a small absmax block.
    "adamw_8bit": 2.5,
    "paged_adamw_8bit": 2.5,
    "paged_adamw_32bit": 12.0,
    "sgd": 4.0,
    "sgd_momentum": 8.0,
    "lion_8bit": 2.0,
    "adafactor": 4.0,
}


def optimizer_bytes(*, trainable_params: int, optimizer: str) -> float:
    """Bytes used by optimizer states for ``trainable_params``."""
    key = optimizer.lower()
    per_param = OPTIMIZER_BYTES_PER_PARAM.get(key, 12.0)
    return trainable_params * per_param


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

# Fitted per-layer activation coefficients (see module docstring):
#   * ATTN_SCALARS x hidden scalars for the attention block, norms, and
#     residual stream (q/k/v/sdpa-out/o-out + fp32 RMSNorm intermediates).
#   * MLP tensors proportional to intermediate_size: gate/up/act/mul for
#     SwiGLU, fc1-out/act for classic MLPs. Under QLoRA these are held in
#     fp32 (bitsandbytes/kbit-prepare upcasting), measured at ~4 B/scalar.
ATTN_SCALARS = 9.0
MLP_TENSORS_SWIGLU = 4.5
MLP_TENSORS_CLASSIC = 2.8


def per_layer_activation_bytes(
    *,
    seq_len: int,
    batch_size: int,
    arch: ArchHints,
    activation_dtype: str = "bf16",
    use_gradient_checkpointing: bool = False,
    attention_implementation: str = "sdpa",
    family: str = "llama",
    quantized_base: bool = False,
) -> float:
    """Per-transformer-layer activation bytes (training, mixed precision).

    Shaped after the accounting in "Reducing Activation Recomputation in Large
    Transformer Models" (Korthikanti et al., 2022), with two changes grounded
    in measurement on the modern HF stack:

      * Fused flash / SDPA attention does not materialize the (b, a, s, s)
        softmax matrix, so that term only applies to ``eager`` attention.
      * SwiGLU MLPs keep ~4.5 intermediate tensors of width
        ``intermediate_size`` alive, and under QLoRA those intermediates are
        stored in fp32 (~4 B/scalar) rather than bf16.

    With gradient checkpointing only each block's input is kept; the
    recomputation peak of a single layer is added once by
    :func:`activations_bytes`, not per layer.
    """
    s = seq_len
    b = batch_size
    h = arch.hidden_size
    ff = arch.intermediate_size
    a = max(1, arch.num_attention_heads)

    act_bytes = dtype_bytes(activation_dtype)

    if use_gradient_checkpointing:
        # Activation recomputation: only the block input is kept.
        return s * b * 2.0 * h * act_bytes

    mlp_tensors = MLP_TENSORS_SWIGLU if _is_swiglu(family) else MLP_TENSORS_CLASSIC
    mlp_tensors *= max(1, arch.num_experts_per_tok)
    mlp_scalar_bytes = 4.0 if quantized_base else act_bytes

    layer = s * b * (ATTN_SCALARS * h * act_bytes + mlp_tensors * ff * mlp_scalar_bytes)
    if attention_implementation.lower() not in {"flash", "flash_attention_2", "sdpa"}:
        # Eager attention materializes the (b, a, s, s) probability matrix
        # for forward and backward.
        layer += 5.0 * a * s * s * b * act_bytes
    return layer


def activations_bytes(
    *,
    seq_len: int,
    batch_size: int,
    arch: ArchHints,
    activation_dtype: str = "bf16",
    use_gradient_checkpointing: bool = False,
    attention_implementation: str = "sdpa",
    family: str = "llama",
    quantized_base: bool = False,
) -> float:
    """Sum activations across the whole transformer stack.

    With gradient checkpointing the total is ``layers x saved-input`` plus the
    recomputation peak of one full layer (recomputation happens one layer at a
    time during backward).
    """
    per_layer = per_layer_activation_bytes(
        seq_len=seq_len,
        batch_size=batch_size,
        arch=arch,
        activation_dtype=activation_dtype,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_implementation=attention_implementation,
        family=family,
        quantized_base=quantized_base,
    )
    total = per_layer * arch.num_hidden_layers
    if use_gradient_checkpointing:
        total += per_layer_activation_bytes(
            seq_len=seq_len,
            batch_size=batch_size,
            arch=arch,
            activation_dtype=activation_dtype,
            use_gradient_checkpointing=False,
            attention_implementation=attention_implementation,
            family=family,
            quantized_base=quantized_base,
        )
    return total


# ---------------------------------------------------------------------------
# Logits / loss chain
# ---------------------------------------------------------------------------


def logits_loss_bytes(
    *,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    logits_dtype: str = "bf16",
) -> float:
    """Peak bytes of the lm_head output and the cross-entropy loss chain.

    transformers materializes the full ``(b, s, vocab)`` logits tensor during
    training and upcasts it to fp32 for the loss; the backward pass allocates
    an fp32 gradient of the same shape. Measured on Qwen2.5 (vocab 151936)
    this chain costs ~14 B per token per vocab entry with bf16 logits:
    logits (2) + fp32 upcast (4) + log-softmax workspace (4) + grad (4).

    For large-vocab models this is the single biggest training buffer —
    e.g. seq 2048, vocab 152k => ~4.1 GiB — and it is unaffected by gradient
    checkpointing. (Fused-CE kernels such as Liger remove most of it; we model
    the stock HF Trainer path.)
    """
    return seq_len * batch_size * vocab_size * (dtype_bytes(logits_dtype) + 12.0)


# ---------------------------------------------------------------------------
# Misc buffers + safety margin
# ---------------------------------------------------------------------------


def cuda_overhead_bytes(*, available_vram_gb: float, overhead_fraction: float = 0.08) -> float:
    """Headroom for CUDA context, allocator fragmentation, and workspaces.

    Modeled as a fraction of the GPU's *available* VRAM, since fragmentation
    grows with allocator size. Default 8% mirrors what we routinely measure
    on 16 GB / 24 GB consumer cards.
    """
    return overhead_fraction * available_vram_gb * (1024.0**3)


def safety_margin_bytes(*, available_vram_gb: float, margin_fraction: float = 0.05) -> float:
    """A small unused-VRAM buffer so we don't recommend the absolute brink."""
    return margin_fraction * available_vram_gb * (1024.0**3)
