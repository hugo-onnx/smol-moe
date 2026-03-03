"""
SmolMoE model implementations.

This package contains:
- components: Core building blocks (attention, MoE, normalization)
- smol_moe: Complete SmolMoE language model
"""

from .components import (
    RotaryEmbedding,
    RMSNorm,
    RoPEAttention,
    MixtureOfExperts,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .smol_moe import (
    SmolMoEModel,
    SmolMoEDecoderLayer,
    SmolMoEForCausalLM,
    smolMoELM,  # Backward compatible alias
)

__all__ = [
    # Components
    "RotaryEmbedding",
    "RMSNorm",
    "RoPEAttention",
    "MixtureOfExperts",
    "rotate_half",
    "apply_rotary_pos_emb",
    "repeat_kv",
    # Models
    "SmolMoEModel",
    "SmolMoEDecoderLayer",
    "SmolMoEForCausalLM",
    "smolMoELM",
]