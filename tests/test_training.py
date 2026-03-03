"""
Unit tests for the training module.

Run with: pytest tests/test_training.py -v
"""

import pytest
import torch
import torch.nn as nn

from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
from smol_moe.training import (
    TrainingConfig,
    causal_lm_loss,
    MoEMetrics,
    get_cosine_schedule_with_warmup,
)


class TestCausalLMLoss:
    """Tests for the causal language modeling loss function."""
    
    def test_basic_loss(self):
        """Test basic loss computation."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = causal_lm_loss(logits, input_ids)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive
        assert not torch.isnan(loss)
    
    def test_loss_with_attention_mask(self):
        """Test loss with attention mask (padding)."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Mask last 3 tokens as padding
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -3:] = 0
        
        loss_masked = causal_lm_loss(logits, input_ids, attention_mask)
        loss_unmasked = causal_lm_loss(logits, input_ids)
        
        # Losses should be different due to masking
        assert loss_masked.item() != loss_unmasked.item()
    
    def test_loss_gradient_flow(self):
        """Test that gradients flow through loss."""
        logits = torch.randn(2, 10, 100, requires_grad=True)
        input_ids = torch.randint(0, 100, (2, 10))
        
        loss = causal_lm_loss(logits, input_ids)
        loss.backward()
        
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestMoEMetrics:
    """Tests for MoE-specific metrics."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    @pytest.fixture
    def model(self, config):
        return SmolMoEForCausalLM(config)
    
    def test_router_confidence(self, model, config):
        """Test router confidence metric."""
        # Run forward pass to populate router logits
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        _ = model(input_ids)
        
        confidence = MoEMetrics.router_confidence(model)
        
        assert 0 <= confidence <= 100
    
    def test_expert_usage_stats(self, model, config):
        """Test expert usage statistics."""
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        _ = model(input_ids)
        
        stats = MoEMetrics.expert_usage_stats(model)
        
        assert "utilization" in stats
        assert "most_used" in stats
        assert "least_used" in stats
        assert "imbalance_ratio" in stats
        assert len(stats["utilization"]) == config.num_experts


class TestCosineScheduler:
    """Tests for learning rate scheduler."""
    
    def test_warmup_phase(self):
        """Test LR increases during warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=10, total_steps=100)
        
        lrs = []
        for _ in range(10):
            lrs.append(scheduler.get_last_lr()[0])
            # Dummy optimizer step to avoid warning
            optimizer.step()
            scheduler.step()
        
        # LR should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i-1]
    
    def test_decay_phase(self):
        """Test LR decreases after warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=10, total_steps=100)
        
        # Skip warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        
        # Collect decay phase LRs
        lrs = []
        for _ in range(50):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
        
        # LR should generally decrease (cosine decay)
        assert lrs[-1] < lrs[0]
    
    def test_final_lr_near_zero(self):
        """Test LR approaches zero at end."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=10, total_steps=100)
        
        for _ in range(100):
            optimizer.step()
            scheduler.step()
        
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < 1e-4  # Near zero


class TestTrainingConfig:
    """Tests for training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.steps == 100
        assert config.learning_rate == 5e-5
        assert config.router_lr_multiplier == 2.0
        assert config.lb_loss_coef == 0.01
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            steps=500,
            learning_rate=1e-4,
            router_lr_multiplier=5.0,
        )
        
        assert config.steps == 500
        assert config.learning_rate == 1e-4
        assert config.router_lr_multiplier == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])