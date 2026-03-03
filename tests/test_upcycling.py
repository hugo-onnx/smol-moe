"""
Unit tests for the upcycling module.

Run with: pytest tests/test_upcycling.py -v

Note: These tests use mock dense models to avoid HuggingFace dependencies
in the test suite. Integration tests with real models are in the notebook.
"""

import pytest
import torch
import torch.nn as nn

from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
from smol_moe.upcycling import (
    UpcyclingError,
    get_moe_layers,
    copy_ffn_to_experts,
    upcycle_dense_to_moe,
    verify_upcycling,
)


class MockAttention(nn.Module):
    """Mock attention module matching LLaMA-style interface."""
    
    def __init__(self, hidden_size: int, num_heads: int, kv_heads: int):
        super().__init__()
        head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class MockMLP(nn.Module):
    """Mock MLP module matching LLaMA-style SwiGLU interface."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class MockDecoderLayer(nn.Module):
    """Mock decoder layer matching LLaMA-style interface."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, kv_heads: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size, num_heads, kv_heads)
        self.mlp = MockMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        
        # Initialize with ones for easy verification
        nn.init.ones_(self.input_layernorm.weight)
        nn.init.ones_(self.post_attention_layernorm.weight)


class MockDenseModel(nn.Module):
    """Mock dense model matching LLaMA-style interface."""
    
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([
            MockDecoderLayer(
                config.hidden_size,
                config.intermediate_size,
                config.num_heads,
                config.kv_heads,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.model.norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)
        nn.init.ones_(self.model.norm.weight)
        
        # Output head (weight-tied in real models)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        
        self.config = config
    
    def forward(self, input_ids):
        """Simplified forward pass for testing."""
        x = self.model.embed_tokens(input_ids)
        # Skip actual computation for mock
        logits = self.lm_head(x)
        return {"logits": logits}


class TestGetLayers:
    """Tests for layer access utilities."""
    
    def test_get_moe_layers(self):
        """Test getting layers from MoE model."""
        config = SmolMoEConfig.small()
        model = SmolMoEForCausalLM(config)
        
        layers = get_moe_layers(model)
        
        assert len(layers) == config.num_hidden_layers
    
    def test_get_moe_layers_invalid(self):
        """Test error on invalid model."""
        with pytest.raises(UpcyclingError):
            get_moe_layers(nn.Linear(10, 10))


class TestCopyFFNToExperts:
    """Tests for FFN to MoE expert conversion."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    @pytest.fixture
    def moe_model(self, config):
        return SmolMoEForCausalLM(config)
    
    @pytest.fixture
    def dense_layer(self, config):
        return MockDecoderLayer(
            config.hidden_size,
            config.intermediate_size,
            config.num_heads,
            config.kv_heads,
        )
    
    def test_copy_ffn_replicates_to_all_experts(self, dense_layer, moe_model, config):
        """Test that FFN weights are replicated to all experts."""
        moe_layer = moe_model.model.layers[0]
        
        # Set known weights in dense layer
        with torch.no_grad():
            dense_layer.mlp.gate_proj.weight.fill_(0.5)
            dense_layer.mlp.up_proj.weight.fill_(0.3)
            dense_layer.mlp.down_proj.weight.fill_(0.2)
        
        # Copy to MoE
        copy_ffn_to_experts(dense_layer, moe_layer, zero_router=True)
        
        # Verify all experts have identical weights
        moe = moe_layer.moe
        for e in range(config.num_experts):
            # gate_bank should be transpose of gate_proj
            assert torch.allclose(
                moe.gate_bank[e],
                dense_layer.mlp.gate_proj.weight.t(),
                atol=1e-6,
            )
            assert torch.allclose(
                moe.up_bank[e],
                dense_layer.mlp.up_proj.weight.t(),
                atol=1e-6,
            )
            assert torch.allclose(
                moe.down_bank[e],
                dense_layer.mlp.down_proj.weight.t(),
                atol=1e-6,
            )
    
    def test_copy_ffn_zeros_router(self, dense_layer, moe_model):
        """Test that router is zeroed when zero_router=True."""
        moe_layer = moe_model.model.layers[0]
        
        # Set non-zero router weights first
        with torch.no_grad():
            moe_layer.moe.gate.weight.fill_(1.0)
        
        copy_ffn_to_experts(dense_layer, moe_layer, zero_router=True)
        
        assert torch.all(moe_layer.moe.gate.weight == 0)
    
    def test_copy_ffn_keeps_router(self, dense_layer, moe_model):
        """Test that router is kept when zero_router=False."""
        moe_layer = moe_model.model.layers[0]
        
        # Set known router weights
        with torch.no_grad():
            moe_layer.moe.gate.weight.fill_(1.0)
        
        copy_ffn_to_experts(dense_layer, moe_layer, zero_router=False)
        
        assert torch.all(moe_layer.moe.gate.weight == 1.0)


class TestUpcycling:
    """Integration tests for the full upcycling process."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    @pytest.fixture
    def dense_model(self, config):
        return MockDenseModel(config)
    
    @pytest.fixture
    def moe_model(self, config):
        return SmolMoEForCausalLM(config)
    
    def test_upcycle_copies_embeddings(self, dense_model, moe_model):
        """Test that embeddings are copied correctly."""
        # Set known embedding values
        with torch.no_grad():
            dense_model.model.embed_tokens.weight.fill_(0.42)
        
        upcycle_dense_to_moe(dense_model, moe_model, verbose=False)
        
        assert torch.allclose(
            moe_model.model.embed_tokens.weight,
            dense_model.model.embed_tokens.weight,
        )
    
    def test_upcycle_copies_attention(self, dense_model, moe_model):
        """Test that attention weights are copied correctly."""
        # Set known values
        with torch.no_grad():
            for layer in dense_model.model.layers:
                layer.self_attn.q_proj.weight.fill_(0.1)
                layer.self_attn.k_proj.weight.fill_(0.2)
                layer.self_attn.v_proj.weight.fill_(0.3)
                layer.self_attn.o_proj.weight.fill_(0.4)
        
        upcycle_dense_to_moe(dense_model, moe_model, verbose=False)
        
        for moe_layer in moe_model.model.layers:
            assert torch.allclose(
                moe_layer.self_attn.W_query.weight,
                torch.full_like(moe_layer.self_attn.W_query.weight, 0.1),
            )
    
    def test_upcycle_copies_norms(self, dense_model, moe_model):
        """Test that normalization weights are copied correctly."""
        upcycle_dense_to_moe(dense_model, moe_model, verbose=False)
        
        # Final norm should be copied
        assert torch.allclose(
            moe_model.model.norm.weight,
            torch.ones_like(moe_model.model.norm.weight),
        )
    
    def test_upcycle_layer_count_mismatch(self, dense_model, config):
        """Test error on layer count mismatch."""
        # Create MoE with different layer count
        bad_config = SmolMoEConfig.small()
        bad_config.num_hidden_layers = config.num_hidden_layers + 1
        moe_model = SmolMoEForCausalLM(bad_config)
        
        with pytest.raises(UpcyclingError, match="Layer count mismatch"):
            upcycle_dense_to_moe(dense_model, moe_model, verbose=False)


class TestVerifyUpcycling:
    """Tests for upcycling verification."""
    
    def test_verify_identical_outputs(self):
        """Test verification passes for identical models."""
        config = SmolMoEConfig.small()
        model1 = SmolMoEForCausalLM(config)
        model2 = SmolMoEForCausalLM(config)
        
        # Copy weights to make identical
        model2.load_state_dict(model1.state_dict())
        
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        
        assert verify_upcycling(model1, model2, input_ids)
    
    def test_verify_different_outputs(self):
        """Test verification fails for different models."""
        config = SmolMoEConfig.small()
        model1 = SmolMoEForCausalLM(config)
        model2 = SmolMoEForCausalLM(config)
        
        # Different random weights
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        
        # Should fail (different random init)
        assert not verify_upcycling(model1, model2, input_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])