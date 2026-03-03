"""
Unit tests for SmolMoE model components.

Run with: pytest tests/test_components.py -v
"""

import math
import pytest
import torch
import torch.nn as nn

from smol_moe.models.components import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    RotaryEmbedding,
    RMSNorm,
    MixtureOfExperts,
    RoPEAttention,
)
from smol_moe.config import SmolMoEConfig


class TestRotaryEmbedding:
    """Tests for rotary position embedding components."""
    
    def test_rotate_half(self):
        """Test that rotate_half correctly rotates dimensions."""
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
        rotated = rotate_half(x)
        
        # First half should be negated second half
        assert rotated.shape == x.shape
        assert torch.allclose(rotated[:, :2], -x[:, 2:])
        assert torch.allclose(rotated[:, 2:], x[:, :2])
    
    def test_rotary_embedding_shape(self):
        """Test RotaryEmbedding output shapes."""
        dim = 64
        seq_len = 128
        batch_size = 2
        
        rope = RotaryEmbedding(dim=dim)
        x = torch.randn(batch_size, 4, seq_len, dim)  # [B, heads, seq, dim]
        
        cos, sin = rope(x)
        
        assert cos.shape == (1, seq_len, dim)
        assert sin.shape == (1, seq_len, dim)
    
    def test_rotary_embedding_device_transfer(self):
        """Test that RoPE handles device correctly."""
        rope = RotaryEmbedding(dim=32)
        x = torch.randn(1, 2, 16, 32)
        
        cos, sin = rope(x)
        assert cos.device == x.device
        assert sin.device == x.device
    
    def test_apply_rotary_pos_emb(self):
        """Test applying rotary embeddings to Q and K."""
        batch, heads, seq, dim = 2, 4, 16, 32
        
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        
        rope = RotaryEmbedding(dim=dim)
        cos, sin = rope(q)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Rotated tensors should be different from originals
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)
    
    def test_repeat_kv(self):
        """Test KV head repetition for GQA."""
        batch, kv_heads, seq, dim = 2, 2, 16, 32
        n_rep = 4  # Repeat each KV head 4 times
        
        kv = torch.randn(batch, kv_heads, seq, dim)
        repeated = repeat_kv(kv, n_rep)
        
        assert repeated.shape == (batch, kv_heads * n_rep, seq, dim)
        
        # Check that heads are correctly repeated
        for i in range(kv_heads):
            for j in range(n_rep):
                assert torch.allclose(repeated[:, i * n_rep + j], kv[:, i])
    
    def test_repeat_kv_no_repeat(self):
        """Test that repeat_kv with n_rep=1 returns input unchanged."""
        kv = torch.randn(2, 4, 16, 32)
        result = repeat_kv(kv, 1)
        assert torch.equal(result, kv)


class TestRMSNorm:
    """Tests for RMS normalization."""
    
    def test_rmsnorm_shape(self):
        """Test RMSNorm preserves shape."""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 16, 64)
        
        out = norm(x)
        assert out.shape == x.shape
    
    def test_rmsnorm_normalization(self):
        """Test that RMSNorm normalizes correctly."""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 16, 64) * 100  # Large values
        
        out = norm(x)
        
        # Check RMS is approximately 1 (before weight scaling)
        # With weight=1, output RMS should be close to 1
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)
    
    def test_rmsnorm_weight(self):
        """Test RMSNorm learned weight."""
        norm = RMSNorm(hidden_size=64)
        assert norm.weight.shape == (64,)
        assert torch.allclose(norm.weight, torch.ones(64))


class TestMixtureOfExperts:
    """Tests for MoE layer."""
    
    @pytest.fixture
    def moe_config(self):
        return {
            "num_experts_per_tok": 1,
            "num_experts": 4,
            "emb_dim": 64,
            "moe_dim": 128,
        }
    
    def test_moe_forward_shape(self, moe_config):
        """Test MoE output shape."""
        moe = MixtureOfExperts(**moe_config)
        x = torch.randn(2, 16, 64)
        
        out = moe(x)
        assert out.shape == x.shape
    
    def test_moe_expert_utilization(self, moe_config):
        """Test expert utilization tracking."""
        moe = MixtureOfExperts(**moe_config)
        x = torch.randn(2, 16, 64)
        
        _ = moe(x)
        
        assert moe._expert_utilization is not None
        assert moe._expert_utilization.shape == (moe_config["num_experts"],)
        
        # Utilization should sum to 1 (it's a distribution)
        assert torch.allclose(moe._expert_utilization.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_moe_load_balancing_loss(self, moe_config):
        """Test load balancing loss computation."""
        moe = MixtureOfExperts(**moe_config)
        x = torch.randn(2, 16, 64)
        
        _ = moe(x)
        
        assert moe._aux_lb is not None
        # LB loss should be positive
        assert moe._aux_lb.item() > 0
    
    def test_moe_router_logits(self, moe_config):
        """Test router logits are stored."""
        moe = MixtureOfExperts(**moe_config)
        x = torch.randn(2, 16, 64)
        
        _ = moe(x)
        
        assert moe._last_router_logits is not None
        assert moe._last_router_logits.shape == (2, 16, moe_config["num_experts"])


class TestRoPEAttention:
    """Tests for attention with RoPE."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    def test_attention_output_shape(self, config):
        """Test attention output shape."""
        attn = RoPEAttention(config)
        x = torch.randn(2, 16, config.hidden_size)
        mask = torch.zeros(1, 1, 16, 16)
        
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape
    
    def test_attention_causal_mask(self, config):
        """Test attention respects causal masking."""
        attn = RoPEAttention(config)
        
        # Create causal mask
        seq_len = 16
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        x = torch.randn(1, seq_len, config.hidden_size)
        out = attn(x, attention_mask=mask)
        
        # Output should be valid (no NaN from masked softmax)
        assert not torch.isnan(out).any()
    
    def test_attention_deterministic(self, config):
        """Test attention is deterministic in eval mode."""
        attn = RoPEAttention(config)
        attn.eval()
        
        x = torch.randn(1, 8, config.hidden_size)
        mask = torch.zeros(1, 1, 8, 8)
        
        out1 = attn(x, attention_mask=mask)
        out2 = attn(x, attention_mask=mask)
        
        assert torch.allclose(out1, out2)


class TestComponentIntegration:
    """Integration tests for component interactions."""
    
    def test_rope_attention_with_gqa(self):
        """Test RoPE attention works with different head configurations."""
        config = SmolMoEConfig(
            hidden_size=64,
            num_heads=8,
            kv_heads=2,  # GQA: 4 query heads per KV head
            num_hidden_layers=1,
        )
        
        attn = RoPEAttention(config)
        x = torch.randn(2, 16, config.hidden_size)
        mask = torch.zeros(1, 1, 16, 16)
        
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape
    
    def test_moe_with_rmsnorm(self):
        """Test MoE combined with RMSNorm (as in decoder)."""
        hidden_size = 64
        
        norm = RMSNorm(hidden_size)
        moe = MixtureOfExperts(
            num_experts_per_tok=1,
            num_experts=4,
            emb_dim=hidden_size,
            moe_dim=128,
        )
        
        x = torch.randn(2, 16, hidden_size)
        
        # Pre-norm + MoE + residual
        residual = x
        x_norm = norm(x)
        x_moe = moe(x_norm)
        out = x_moe + residual
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])