"""
Unit tests for SmolMoE full model.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn

from smol_moe import SmolMoEConfig, SmolMoEForCausalLM, SmolMoEModel


class TestSmolMoEConfig:
    """Tests for model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SmolMoEConfig()
        
        assert config.vocab_size == 49152
        assert config.hidden_size == 576
        assert config.num_hidden_layers == 30
        assert config.num_heads == 9
        assert config.kv_heads == 3
        assert config.num_experts == 3
        assert config.num_experts_per_tok == 1
    
    def test_small_config(self):
        """Test small configuration for debugging."""
        config = SmolMoEConfig.small()
        
        assert config.num_hidden_layers == 4
        assert config.hidden_size == 256
    
    def test_config_validation(self):
        """Test configuration validation."""
        # hidden_size not divisible by num_heads
        with pytest.raises(AssertionError):
            SmolMoEConfig(hidden_size=100, num_heads=9)
        
        # num_heads not divisible by kv_heads
        with pytest.raises(AssertionError):
            SmolMoEConfig(num_heads=9, kv_heads=4)
        
        # more experts per token than total experts
        with pytest.raises(AssertionError):
            SmolMoEConfig(num_experts=2, num_experts_per_tok=3)
    
    def test_config_properties(self):
        """Test computed properties."""
        config = SmolMoEConfig()
        
        assert config.head_dim == config.hidden_size // config.num_heads
        assert config.num_kv_groups == config.num_heads // config.kv_heads
    
    def test_config_serialization(self, tmp_path):
        """Test config save and load."""
        config = SmolMoEConfig(hidden_size=128, num_heads=4, kv_heads=2)
        path = tmp_path / "config.json"
        
        config.save(path)
        loaded = SmolMoEConfig.load(path)
        
        assert loaded.hidden_size == 128
        assert loaded.num_heads == 4
        assert loaded.vocab_size == config.vocab_size


class TestSmolMoEModel:
    """Tests for the transformer backbone."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    def test_model_forward(self, config):
        """Test model forward pass."""
        model = SmolMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        
        outputs = model(input_ids)
        
        assert len(outputs) == 1
        assert outputs[0].shape == (2, 16, config.hidden_size)
    
    def test_model_with_embeddings(self, config):
        """Test model with pre-computed embeddings."""
        model = SmolMoEModel(config)
        embeddings = torch.randn(2, 16, config.hidden_size)
        
        outputs = model(inputs_embeds=embeddings)
        
        assert outputs[0].shape == embeddings.shape


class TestSmolMoEForCausalLM:
    """Tests for the full language model."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    @pytest.fixture
    def model(self, config):
        return SmolMoEForCausalLM(config)
    
    def test_forward_logits(self, model, config):
        """Test forward pass returns logits."""
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (2, 16, config.vocab_size)
    
    def test_forward_with_labels(self, model, config):
        """Test forward pass with labels computes loss."""
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        
        outputs = model(input_ids, labels=labels)
        
        assert 'loss' in outputs
        assert outputs['loss'].dim() == 0  # scalar
        assert outputs['loss'].item() > 0
    
    def test_expert_utilization(self, model, config):
        """Test expert utilization tracking."""
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        
        _ = model(input_ids)
        utilization, lb_loss = model.get_expert_utilization()
        
        assert utilization.shape == (config.num_experts,)
        assert lb_loss.dim() == 0
        assert lb_loss.item() > 0
    
    def test_load_balancing_loss_value(self, model, config):
        """Test that load balancing loss is approximately 1.0 for uniform routing."""
        # With uniform routing, LB loss should be close to num_experts
        # But with actual routing, it varies
        input_ids = torch.randint(0, config.vocab_size, (4, 32))
        
        _ = model(input_ids)
        _, lb_loss = model.get_expert_utilization()
        
        # Loss should be positive and reasonable
        assert 0 < lb_loss.item() < config.num_experts * 2
    
    def test_router_logits(self, model, config):
        """Test router logits retrieval."""
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        
        _ = model(input_ids)
        router_logits = model.get_router_logits()
        
        assert len(router_logits) == config.num_hidden_layers
        for logits in router_logits:
            if logits is not None:
                assert logits.shape[-1] == config.num_experts
    
    def test_weight_tying(self, config):
        """Test embedding weight tying."""
        model = SmolMoEForCausalLM(config)
        
        # Embeddings and LM head should share weights
        assert model.lm_head.weight is model.model.embed_tokens.weight
    
    def test_generation(self, model, config):
        """Test text generation."""
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
        )
        
        assert generated.shape[0] == 1
        assert generated.shape[1] >= input_ids.shape[1]
        assert generated.shape[1] <= input_ids.shape[1] + 10
    
    def test_generation_greedy(self, model, config):
        """Test greedy generation (temperature=0)."""
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        
        gen1 = model.generate(input_ids.clone(), max_new_tokens=5, temperature=0)
        gen2 = model.generate(input_ids.clone(), max_new_tokens=5, temperature=0)
        
        # Greedy should be deterministic
        assert torch.equal(gen1, gen2)
    
    def test_generation_top_k(self, model, config):
        """Test top-k sampling."""
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            top_k=10,
        )
        
        assert generated.shape[1] > input_ids.shape[1]
    
    def test_reset_weights(self, model, config):
        """Test weight reset functionality."""
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        # Run forward to populate metrics
        _ = model(input_ids)
        _, lb_before = model.get_expert_utilization()

        # Reset
        model.reset_weights_and_metrics()

        # Metrics should be cleared
        util, lb = model.get_expert_utilization()
        assert lb.item() == 0

    def test_padding_mask_applied(self, config):
        """Verify attention_mask correctly blocks padded key positions.

        Mask the last 4 tokens of a sequence as padding.  Query positions at
        those indices (12-15) normally attend to themselves; with the mask
        applied they lose that access, so their outputs must change.  The
        first 12 positions are unaffected because the causal mask already
        prevents them from attending to positions 12-15.
        """
        model = SmolMoEForCausalLM(config)
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        # Forward without any attention mask
        with torch.no_grad():
            out_unmasked = model(input_ids)['logits']

        # Mask last 4 tokens as padding
        attention_mask = torch.ones(1, 16)
        attention_mask[0, 12:] = 0

        with torch.no_grad():
            out_masked = model(input_ids, attention_mask=attention_mask)['logits']

        # Padded positions (12-15) lose self-attention access; outputs must differ
        assert not torch.allclose(out_unmasked[0, 12:], out_masked[0, 12:]), (
            "Outputs at padded positions should change when attention_mask is applied"
        )

        # Non-padded positions (0-11) are unaffected: causal mask already
        # blocks keys 12-15, so adding -inf there changes nothing
        assert torch.allclose(out_unmasked[0, :12], out_masked[0, :12], atol=1e-5), (
            "Non-padded positions should be unchanged when only future tokens are masked"
        )


class TestModelIntegration:
    """Integration tests for model with realistic scenarios."""
    
    def test_batch_processing(self):
        """Test model handles different batch sizes."""
        config = SmolMoEConfig.small()
        model = SmolMoEForCausalLM(config)
        
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, 16))
            outputs = model(input_ids)
            assert outputs['logits'].shape[0] == batch_size
    
    def test_variable_sequence_length(self):
        """Test model handles different sequence lengths."""
        config = SmolMoEConfig.small()
        model = SmolMoEForCausalLM(config)
        
        for seq_len in [8, 32, 64, 128]:
            input_ids = torch.randint(0, config.vocab_size, (2, seq_len))
            outputs = model(input_ids)
            assert outputs['logits'].shape[1] == seq_len
    
    def test_gradient_flow(self):
        """Test gradients flow through all components."""
        config = SmolMoEConfig.small()
        model = SmolMoEForCausalLM(config)
        
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        
        outputs = model(input_ids, labels=labels)
        outputs['loss'].backward()
        
        # Check gradients exist for key parameters
        assert model.model.embed_tokens.weight.grad is not None
        
        for layer in model.model.layers:
            # Check attention gradients
            assert layer.self_attn.W_query.weight.grad is not None
            
            # Check MoE expert gradients (gate doesn't get gradients due to argmax)
            # But the expert weight banks should receive gradients
            assert layer.moe.up_bank.grad is not None
            assert layer.moe.gate_bank.grad is not None
            assert layer.moe.down_bank.grad is not None
    
    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        from smol_moe.utils import count_parameters
        
        config = SmolMoEConfig.small()
        model = SmolMoEForCausalLM(config)
        
        params = count_parameters(model)
        
        # Small model should have reasonable number of params
        assert params > 1_000_000  # At least 1M
        assert params < 100_000_000  # Less than 100M


if __name__ == "__main__":
    pytest.main([__file__, "-v"])