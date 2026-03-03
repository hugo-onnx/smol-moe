"""
Core neural network components for SmolMoE Language Model.

This module implements the fundamental building blocks:
- Rotary Position Embeddings (RoPE)
- RMS Normalization
- Grouped-Query Attention with RoPE
- Mixture of Experts (MoE) with SwiGLU activation

References:
- RoPE: https://arxiv.org/abs/2104.09864
- RMSNorm: https://arxiv.org/abs/1910.07467
- SwiGLU: https://arxiv.org/abs/2002.05202
- MoE Load Balancing: https://arxiv.org/abs/2101.03961
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions of the input tensor.
    
    This is a helper function for applying rotary position embeddings.
    It splits the last dimension in half and swaps the two halves with negation.
    
    Args:
        x: Input tensor of shape [..., dim]
        
    Returns:
        Tensor of shape [..., dim] with rotated dimensions
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    RoPE encodes position information by rotating pairs of dimensions
    in the query and key vectors, enabling relative position awareness
    in attention mechanisms.
    
    Args:
        q: Query tensor of shape [batch, heads, seq, head_dim]
        k: Key tensor of shape [batch, kv_heads, seq, head_dim]
        cos: Cosine component of rotary embeddings
        sin: Sine component of rotary embeddings
        position_ids: Optional position indices (unused, for compatibility)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
        
    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    # Ensure embeddings are on the same device as inputs
    if cos.device != q.device:
        cos = cos.to(q.device)
        sin = sin.to(q.device)
    
    # Add dimension for broadcasting over heads
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply rotation: RoPE formula
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value heads for grouped-query attention.
    
    In GQA, we have fewer KV heads than query heads. This function
    repeats the KV heads to match the number of query heads.
    
    Args:
        hidden_states: KV tensor of shape [batch, kv_heads, seq, head_dim]
        n_rep: Number of times to repeat each KV head
        
    Returns:
        Tensor of shape [batch, kv_heads * n_rep, seq, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    if n_rep == 1:
        return hidden_states
    
    # Expand and reshape to repeat heads
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.
    
    RoPE encodes absolute position with a rotation matrix that naturally
    incorporates relative position information in attention scores.
    
    Args:
        dim: Dimension of the embeddings (typically head_dim)
        base: Base frequency for computing rotation frequencies
    """
    
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Precompute inverse frequencies
        # freq_i = 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for the input sequence.
        
        Args:
            x: Input tensor, used only for sequence length and device
            
        Returns:
            Tuple of (cos, sin) embeddings, each of shape [1, seq, dim]
        """
        seq_len = x.shape[-2]
        device = x.device
        
        # Create position indices
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        
        # Compute angles: pos * freq for each position and frequency
        # Shape: [seq_len, dim/2]
        inv_freq = self.inv_freq.to(device)
        angles = torch.einsum("p,f->pf", positions, inv_freq)
        
        # Duplicate angles for full dimension (cos and sin applied to same angles)
        # Shape: [seq_len, dim]
        emb = torch.cat((angles, angles), dim=-1)
        
        # Return with batch dimension for broadcasting
        # Shape: [1, seq_len, dim]
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a simplification of LayerNorm that only normalizes by the
    RMS of the activations, without centering. This is more efficient
    and works well in practice for transformer models.
    
    Args:
        hidden_size: Dimension of the input features
        eps: Small constant for numerical stability
        
    Reference: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute variance (RMS^2) over last dimension
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        
        # Normalize by RMS
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        
        # Scale by learned weight
        return self.weight * hidden_states


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) layer with SwiGLU activation.
    
    This implements a sparse MoE where each token is routed to the top-k
    experts based on a learned gating function. The expert networks use
    SwiGLU activation for improved performance.
    
    Key features:
    - Sparse routing: only k experts process each token
    - Load balancing loss for even expert utilization
    - SwiGLU activation: SiLU(gate) * up
    
    Args:
        num_experts_per_tok: Number of experts to route each token to (top-k)
        num_experts: Total number of expert networks
        emb_dim: Input/output embedding dimension
        moe_dim: Hidden dimension within each expert MLP
        dtype: Parameter dtype
        
    Reference: https://arxiv.org/abs/2101.03961
    """
    
    def __init__(
        self,
        num_experts_per_tok: int,
        num_experts: int,
        emb_dim: int,
        moe_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.k = int(num_experts_per_tok)  # Top-k experts per token
        self.E = int(num_experts)           # Total experts
        self.D = int(emb_dim)               # Embedding dimension
        self.H = int(moe_dim)               # Hidden dimension
        
        # Router/gating network: projects to expert scores
        self.gate = nn.Linear(self.D, self.E, bias=False, dtype=dtype)
        
        # Expert weight banks (all experts in single tensors for efficiency)
        # gate_bank: for gating in SwiGLU
        # up_bank: for up-projection in SwiGLU
        # down_bank: for down-projection back to embedding dim
        self.gate_bank = nn.Parameter(torch.empty(self.E, self.D, self.H, dtype=dtype))
        self.up_bank = nn.Parameter(torch.empty(self.E, self.D, self.H, dtype=dtype))
        self.down_bank = nn.Parameter(torch.empty(self.E, self.H, self.D, dtype=dtype))
        
        # Initialize weights
        self._init_weights()
        
        # Cached metrics for monitoring
        self._expert_utilization: Optional[torch.Tensor] = None
        self._aux_lb: Optional[torch.Tensor] = None
        self._last_router_logits: Optional[torch.Tensor] = None

        # Augmentation state: set by training utilities before forward, reset after backward
        self._router_noise_std: float = 0.0
        self._forced_routing_mask: Optional[torch.Tensor] = None
    
    def _init_weights(self):
        """Initialize expert weights with Kaiming uniform."""
        for param in [self.gate_bank, self.up_bank, self.down_bank]:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    
    def compute_load_balancing_loss(self, logits: torch.Tensor) -> None:
        """
        Compute expert utilization metrics and auxiliary load balancing loss.
        
        The load balancing loss encourages even distribution of tokens
        across experts, preventing expert collapse where only a few
        experts are used.
        
        Loss = E * sum(load_i * importance_i)
        
        Where:
        - load_i: fraction of tokens routed to expert i
        - importance_i: mean routing probability for expert i
        
        Args:
            logits: Router logits of shape [batch, seq, num_experts]
        """
        # Get selected experts (hard assignment)
        selected = torch.argmax(logits, dim=-1)
        selected_one_hot = F.one_hot(selected, num_classes=self.E)
        
        # Compute routing probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Load: fraction of tokens assigned to each expert
        load = selected_one_hot.float().mean(dim=(0, 1))
        
        # Importance: mean probability mass assigned to each expert
        importance = probs.mean(dim=(0, 1))
        
        # Auxiliary loss: encourages load * importance to be uniform
        self._aux_lb = self.E * torch.sum(load * importance)
        
        # Store utilization for monitoring
        self._expert_utilization = load
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [batch, seq, emb_dim]
            
        Returns:
            Output tensor of shape [batch, seq, emb_dim]
        """
        B, T, D = x.shape
        assert D == self.D, f"Expected emb_dim={self.D}, got {D}"
        
        # Compute router logits
        logits = self.gate(x)  # [B, T, E]

        # Apply router noise for exploration (set externally before forward)
        if self._router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self._router_noise_std

        # Apply forced routing mask for expert specialization (set externally before forward)
        if self._forced_routing_mask is not None:
            logits = logits + self._forced_routing_mask

        # Store post-augmentation logits for monitoring and external loss computation
        self._last_router_logits = logits

        # Select top expert for each token
        selected = torch.argmax(logits, dim=-1)  # [B, T]
        
        # Compute all expert outputs in parallel
        # Up projection: [B, T, E, H]
        u = torch.einsum("btd,edh->bteh", x, self.up_bank)
        
        # Gate projection: [B, T, E, H]
        a = torch.einsum("btd,edh->bteh", x, self.gate_bank)
        
        # SwiGLU: SiLU(gate) * up
        h = F.silu(a) * u
        
        # Down projection: [B, T, E, D]
        y = torch.einsum("bteh,ehd->bted", h, self.down_bank)
        
        # Gather output from selected expert for each token
        gather_idx = selected.view(B, T, 1, 1).expand(-1, -1, -1, D)
        y = torch.gather(y, dim=2, index=gather_idx).squeeze(2)  # [B, T, D]
        
        # Compute load balancing metrics
        self.compute_load_balancing_loss(logits)
        
        return y


class RoPEAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embeddings and Grouped-Query Attention.
    
    This attention implementation supports:
    - Rotary Position Embeddings (RoPE) for position encoding
    - Grouped-Query Attention (GQA) for efficiency
    - Causal masking for autoregressive generation
    
    Args:
        config: Model configuration object with attention parameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_heads = config.kv_heads
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        
        # Projections
        self.W_query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.W_key = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_value = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-head attention with RoPE.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq, hidden_size]
            attention_mask: Causal mask of shape [1, 1, seq, seq]
            
        Returns:
            Output tensor of shape [batch, seq, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to queries, keys, values
        q_states = self.W_query(hidden_states)
        k_states = self.W_key(hidden_states)
        v_states = self.W_value(hidden_states)
        
        # Reshape for multi-head attention: [B, heads, seq, head_dim]
        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = k_states.view(batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(q_states)
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)
        
        # Repeat KV heads for grouped-query attention
        num_kv_groups = self.num_heads // self.kv_heads
        k_states = repeat_kv(k_states, num_kv_groups)
        v_states = repeat_kv(v_states, num_kv_groups)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / scale
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout (dropout=0 for inference)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v_states)
        
        # Reshape back: [B, seq, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Output projection
        return self.W_output(attn_output)