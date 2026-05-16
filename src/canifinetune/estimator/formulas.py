"""Pure-math helpers for the memory estimator.

Everything here is a function on numbers — no torch, no HTTP, no I/O. Tests
exercise this module directly so the heuristics are reproducible.

References:
  * Megatron-LM activation-memory equations, "Reducing Activation Recomputation
    in Large Transformer Models" (Korthikanti et al., 2022). Used as the
    skeleton of the per-layer activation formula.
  * Hugging Face transformers + PEFT default LoRA target modules per
    architecture family.
  * bitsandbytes documentation for NF4 block size and double-quantization
    overhead.

The formulas below intentionally err slightly toward over-estimation so the
"feasible" verdict is conservative on consumer cards.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.units import dtype_bytes, quant_overhead


@dataclass(frozen=True)
class ArchHints:
    """Shape numbers needed for the activation formula."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int


# ---------------------------------------------------------------------------
# Weight memory
# ---------------------------------------------------------------------------

def weights_bytes(
    *,
    num_params: int,
    base_dtype: str,
    quantization: str | None,
) -> float:
    """Bytes used to store the *base* model weights at rest.

    ``quantization`` overrides ``base_dtype`` when set. Adds a small overhead
    for absmax / lookup tables for 4/8-bit schemes.
    """
    if quantization and quantization.lower() not in {"none", "fp16", "bf16", "fp32"}:
        q = quantization.lower()
        bytes_per_param = dtype_bytes(_quant_storage_dtype(q))
        overhead = quant_overhead(q)
        return num_params * (bytes_per_param + overhead)
    return num_params * dtype_bytes(base_dtype)


def _quant_storage_dtype(quantization: str) -> str:
    if quantization in {"int8", "fp8"}:
        return "int8"
    if quantization in {"nf4", "fp4", "int4", "nf4_double_quant"}:
        return "nf4"
    raise ValueError(f"Unknown quantization {quantization!r}")


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
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "conservative": ["q_proj", "v_proj"],
    },
    "qwen2": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "conservative": ["q_proj", "v_proj"],
    },
    "mistral": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "conservative": ["q_proj", "v_proj"],
    },
    "phi": {
        "attention": ["q_proj", "k_proj", "v_proj", "dense"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        "conservative": ["q_proj", "v_proj"],
    },
    "gpt2": {
        "attention": ["c_attn", "c_proj"],
        "all_linear": ["c_attn", "c_proj", "c_fc"],
        "conservative": ["c_attn"],
    },
    "gemma": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "conservative": ["q_proj", "v_proj"],
    },
}


def default_target_modules(family: str, scope: str = "attention") -> list[str]:
    """Get default target_modules for a model family ('attention'/'all_linear'/'conservative')."""
    fam = (family or "").lower()
    table = TARGET_MODULES_BY_FAMILY.get(fam) or TARGET_MODULES_BY_FAMILY["llama"]
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
        if name in {"q_proj", "o_proj", "c_proj", "dense"}:
            per_layer += _lora_params_for_linear(h, h, rank)
        elif name in {"k_proj", "v_proj"}:
            per_layer += _lora_params_for_linear(h, kv_dim, rank)
        elif name in {"gate_proj", "up_proj", "fc1"}:
            per_layer += _lora_params_for_linear(h, ff, rank)
        elif name in {"down_proj", "fc2"}:
            per_layer += _lora_params_for_linear(ff, h, rank)
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


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def gradients_bytes(*, trainable_params: int, grad_dtype: str = "bf16") -> float:
    """Gradient buffer bytes. With LoRA, gradients exist only for adapters."""
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

def per_layer_activation_bytes(
    *,
    seq_len: int,
    batch_size: int,
    arch: ArchHints,
    activation_dtype: str = "bf16",
    use_gradient_checkpointing: bool = False,
    attention_implementation: str = "sdpa",
) -> float:
    """Per-transformer-layer activation bytes (training, mixed precision).

    Based on the activation-memory accounting in "Reducing Activation
    Recomputation in Large Transformer Models" (Korthikanti et al., 2022),
    Eq. 2 (per-layer activations):

        s * b * h * (34 + 5 * a * s / h)

    Notes:
      * The original derivation assumes vanilla scaled dot-product attention
        which materializes the (b, a, s, s) attention probability matrix.
        Modern flash / SDPA fused kernels skip that buffer; we deduct it via
        ``attention_implementation="flash"`` or ``"sdpa"``.
      * With *full* activation recomputation the per-layer cost collapses to
        roughly ``s * b * h`` (only the inputs to each block are kept). HF's
        ``gradient_checkpointing=True`` is effectively this.
      * We multiply by ``dtype_bytes(activation_dtype)`` at the end so the
        scalar count maps to actual bytes.
    """
    s = seq_len
    b = batch_size
    h = arch.hidden_size
    a = max(1, arch.num_attention_heads)

    bytes_per_scalar = dtype_bytes(activation_dtype)

    if use_gradient_checkpointing:
        # Activation recomputation: only the block input is kept.
        scalars = s * b * h * 2.0
        return scalars * bytes_per_scalar

    attn_extra = 5.0 * a * s / max(1, h)
    if attention_implementation.lower() in {"flash", "flash_attention_2", "sdpa"}:
        # Fused attention doesn't materialize the s x s softmax matrix.
        attn_extra = 0.0
    base = 34.0
    scalars = s * b * h * (base + attn_extra)
    return scalars * bytes_per_scalar


def activations_bytes(
    *,
    seq_len: int,
    batch_size: int,
    arch: ArchHints,
    activation_dtype: str = "bf16",
    use_gradient_checkpointing: bool = False,
    attention_implementation: str = "sdpa",
) -> float:
    """Sum per-layer activations across the whole transformer stack."""
    per_layer = per_layer_activation_bytes(
        seq_len=seq_len,
        batch_size=batch_size,
        arch=arch,
        activation_dtype=activation_dtype,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_implementation=attention_implementation,
    )
    return per_layer * arch.num_hidden_layers


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
