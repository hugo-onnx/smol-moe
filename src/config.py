"""
Configuration classes for SmolMoE Language Model.

This module provides dataclass-based configuration for the SmolMoE architecture,
enabling easy serialization, validation, and modification of model hyperparameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class SmolMoEConfig:
    """
    Configuration for SmolMoE (Small Mixture of Experts) Language Model.
    
    This configuration follows the architecture patterns from modern MoE models,
    combining efficient sparse expert routing with transformer-based language modeling.
    
    Attributes:
        vocab_size: Size of the vocabulary (number of unique tokens).
        hidden_size: Dimension of the hidden representations.
        intermediate_size: Dimension of the MLP intermediate layer (per expert).
        num_hidden_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        kv_heads: Number of key-value heads (for grouped-query attention).
        num_experts: Total number of experts in the MoE layer.
        num_experts_per_tok: Number of experts activated per token (top-k).
        rope_theta: Base frequency for rotary position embeddings.
        rms_norm_eps: Epsilon for RMS normalization stability.
        max_position_embeddings: Maximum sequence length the model can handle.
        tie_word_embeddings: Whether to share input/output embedding weights.
    """
    
    # Vocabulary and embedding dimensions
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    
    # Architecture depth and attention
    num_hidden_layers: int = 30
    num_heads: int = 9
    kv_heads: int = 3
    
    # Mixture of Experts configuration
    num_experts: int = 3
    num_experts_per_tok: int = 1
    
    # Positional encoding
    rope_theta: float = 10000.0
    
    # Normalization
    rms_norm_eps: float = 1e-5
    
    # Sequence handling
    max_position_embeddings: int = 2048
    
    # Weight sharing
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"kv_heads ({self.kv_heads})"
        )
        assert self.num_experts_per_tok <= self.num_experts, (
            f"num_experts_per_tok ({self.num_experts_per_tok}) cannot exceed "
            f"num_experts ({self.num_experts})"
        )
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_hidden_layers > 0, "num_hidden_layers must be positive"
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_heads
    
    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per key-value head (for GQA)."""
        return self.num_heads // self.kv_heads
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SmolMoEConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str | Path) -> "SmolMoEConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def small(cls) -> "SmolMoEConfig":
        """Create a small configuration for testing/debugging."""
        return cls(
            vocab_size=49152,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_heads=4,
            kv_heads=2,
            num_experts=2,
            num_experts_per_tok=1,
        )
    
    @classmethod
    def default(cls) -> "SmolMoEConfig":
        """Create the default SmolMoE configuration."""
        return cls()


# Backward compatibility: expose config as a simple namespace-like object
config = SmolMoEConfig.default()