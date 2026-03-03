"""
Unit tests for domain-specialized expert training.

Run with: pytest tests/test_domain_expert.py -v
"""

import pytest
import torch
import torch.nn as nn

from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
from smol_moe.domain_expert import (
    DomainTrainingConfig,
    DomainPrefixEmbedding,
    CurriculumBatcher,
    DomainExpertMetrics,
    get_domain_mapping,
    get_curriculum_probs,
    get_gate_temperature,
    strip_think_tags,
    find_subsequence,
    router_supervision_loss,
    apply_expert_dropout,
    apply_router_noise,
)


class TestTextProcessing:
    """Tests for text processing utilities."""
    
    def test_strip_think_tags(self):
        """Test removal of think tags."""
        text = "Hello <think>internal reasoning</think> world"
        result = strip_think_tags(text)
        assert result == "Hello  world"
    
    def test_strip_think_tags_incomplete(self):
        """Test removal of incomplete think tags at end."""
        text = "Hello <think>incomplete"
        result = strip_think_tags(text)
        assert result == "Hello"
    
    def test_strip_think_tags_none(self):
        """Test with no think tags."""
        text = "Hello world"
        result = strip_think_tags(text)
        assert result == "Hello world"
    
    def test_find_subsequence(self):
        """Test finding subsequence."""
        seq = [1, 2, 3, 4, 5]
        assert find_subsequence(seq, [3, 4]) == 2
        assert find_subsequence(seq, [1, 2]) == 0
        assert find_subsequence(seq, [6, 7]) == -1


class TestDomainMapping:
    """Tests for domain mapping utilities."""
    
    def test_get_domain_mapping(self):
        """Test domain mapping creation."""
        domains = ["chat", "code", "math"]
        domain_to_id, domain_to_expert = get_domain_mapping(domains)
        
        assert domain_to_id == {"chat": 0, "code": 1, "math": 2}
        assert domain_to_expert == {0: 0, 1: 1, 2: 2}


class TestDomainPrefixEmbedding:
    """Tests for domain prefix embeddings."""
    
    @pytest.fixture
    def prefix_emb(self):
        return DomainPrefixEmbedding(
            num_domains=3,
            prefix_length=2,
            hidden_size=64,
        )
    
    @pytest.fixture
    def embed_tokens(self):
        return nn.Embedding(1000, 64)
    
    def test_forward_shape(self, prefix_emb, embed_tokens):
        """Test output shapes."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        domain_ids = torch.tensor([0, 1])
        
        inputs_embeds, new_mask = prefix_emb(
            input_ids, attention_mask, domain_ids, embed_tokens
        )
        
        # Should have prefix_length extra tokens
        assert inputs_embeds.shape == (batch_size, seq_len + 2, 64)
        assert new_mask.shape == (batch_size, seq_len + 2)
    
    def test_prefix_mask_ones(self, prefix_emb, embed_tokens):
        """Test that prefix positions have attention mask = 1."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        domain_ids = torch.tensor([0, 1])
        
        _, new_mask = prefix_emb(input_ids, attention_mask, domain_ids, embed_tokens)
        
        # First 2 positions (prefix) should be 1
        assert torch.all(new_mask[:, :2] == 1)


class TestCurriculumLearning:
    """Tests for curriculum learning utilities."""
    
    def test_get_curriculum_probs_start(self):
        """Test probabilities at start of training."""
        probs = get_curriculum_probs(
            step=0, total_steps=100,
            start_probs=[0.8, 0.1, 0.1],
            end_probs=[0.33, 0.33, 0.34],
        )
        
        assert len(probs) == 3
        assert abs(probs[0] - 0.8) < 0.01
    
    def test_get_curriculum_probs_end(self):
        """Test probabilities at end of training."""
        probs = get_curriculum_probs(
            step=100, total_steps=100,
            start_probs=[0.8, 0.1, 0.1],
            end_probs=[0.33, 0.33, 0.34],
        )
        
        assert abs(probs[0] - 0.33) < 0.01
        assert abs(probs[1] - 0.33) < 0.01
    
    def test_get_curriculum_probs_middle(self):
        """Test probabilities interpolate correctly."""
        probs = get_curriculum_probs(
            step=50, total_steps=100,
            start_probs=[1.0, 0.0, 0.0],
            end_probs=[0.0, 0.0, 1.0],
        )
        
        # Should be halfway between
        assert abs(probs[0] - 0.5) < 0.01
        assert abs(probs[2] - 0.5) < 0.01


class TestGateTemperature:
    """Tests for gating temperature annealing."""
    
    def test_temperature_at_start(self):
        """Test temperature at start of training."""
        tau = get_gate_temperature(
            step=1, tau_start=1.0, tau_end=0.3,
            warmup_frac=0.2, total_steps=100,
        )
        
        assert tau > 0.9  # Near start value
    
    def test_temperature_at_end(self):
        """Test temperature at end of training."""
        tau = get_gate_temperature(
            step=100, tau_start=1.0, tau_end=0.3,
            warmup_frac=0.2, total_steps=100,
        )
        
        assert abs(tau - 0.3) < 0.01  # At end value


class TestRouterSupervision:
    """Tests for router supervision loss."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    @pytest.fixture
    def model(self, config):
        return SmolMoEForCausalLM(config)
    
    def test_router_supervision_loss(self, model, config):
        """Test router supervision loss computation."""
        # Run forward to populate router logits
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        _ = model(input_ids)
        
        domain_ids = torch.tensor([0, 1])
        domain_to_expert = {0: 0, 1: 1}
        
        loss = router_supervision_loss(model, domain_ids, domain_to_expert)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative


class TestExpertDropout:
    """Tests for expert dropout."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    @pytest.fixture
    def model(self, config):
        return SmolMoEForCausalLM(config)
    
    def test_apply_expert_dropout(self, model, config):
        """Test expert dropout sets the forced routing mask on all MoE layers."""
        domain_ids = torch.tensor([0, 1])
        domain_to_expert = {0: 0, 1: 1}

        # Apply dropout with prob=1.0 to guarantee it runs
        apply_expert_dropout(model, domain_ids, domain_to_expert, prob=1.0)

        for layer in model.model.layers:
            moe = getattr(layer, 'moe', None)
            if moe is None:
                continue

            assert moe._forced_routing_mask is not None, (
                "Mask should be set on every MoE layer after apply_expert_dropout"
            )
            mask = moe._forced_routing_mask  # [B, 1, E]

            # Sequence 0 → expert 0: target slot is 0, all others -inf
            assert mask[0, 0, 0] == 0.0
            assert torch.all(mask[0, 0, 1:] == float('-inf'))

            # Sequence 1 → expert 1: target slot is 1, others -inf
            assert mask[1, 0, 0] == float('-inf')
            assert mask[1, 0, 1] == 0.0

        # Cleanup
        for layer in model.model.layers:
            moe = getattr(layer, 'moe', None)
            if moe is not None:
                moe._forced_routing_mask = None

    def test_expert_dropout_forces_routing(self, model, config):
        """With prob=1.0 the actual argmax of routing logits must equal the target expert."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        domain_ids = torch.tensor([0, 1])
        domain_to_expert = {0: 0, 1: 1}

        # Set forced routing mask before forward
        apply_expert_dropout(model, domain_ids, domain_to_expert, prob=1.0)

        model.eval()
        with torch.no_grad():
            model(input_ids)

        for layer in model.model.layers:
            moe = getattr(layer, 'moe', None)
            if moe is None or moe._last_router_logits is None:
                continue

            selections = torch.argmax(moe._last_router_logits, dim=-1)  # [B, T]

            assert torch.all(selections[0] == 0), (
                f"Domain 0 tokens should route to expert 0, got {selections[0]}"
            )
            assert torch.all(selections[1] == 1), (
                f"Domain 1 tokens should route to expert 1, got {selections[1]}"
            )

        # Cleanup
        for layer in model.model.layers:
            moe = getattr(layer, 'moe', None)
            if moe is not None:
                moe._forced_routing_mask = None

    def test_router_noise_affects_routing(self, model, config):
        """Very large router noise must change at least some routing decisions."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        # Forward without noise (baseline)
        model.eval()
        with torch.no_grad():
            model(input_ids)

        no_noise_selections = [
            torch.argmax(layer.moe._last_router_logits, dim=-1).clone()
            for layer in model.model.layers
            if layer.moe._last_router_logits is not None
        ]

        # Set large noise std on all MoE layers
        for layer in model.model.layers:
            layer.moe._router_noise_std = 100.0

        # Forward with noise (different random state so noise is non-zero)
        torch.manual_seed(7)
        with torch.no_grad():
            model(input_ids)

        noise_selections = [
            torch.argmax(layer.moe._last_router_logits, dim=-1).clone()
            for layer in model.model.layers
            if layer.moe._last_router_logits is not None
        ]

        any_change = any(
            not torch.equal(a, b)
            for a, b in zip(no_noise_selections, noise_selections)
        )
        assert any_change, "Large router noise should change at least some routing decisions"

        # Cleanup
        for layer in model.model.layers:
            layer.moe._router_noise_std = 0.0


class TestDomainExpertMetrics:
    """Tests for domain expert metrics."""
    
    @pytest.fixture
    def config(self):
        return SmolMoEConfig.small()
    
    @pytest.fixture
    def model(self, config):
        return SmolMoEForCausalLM(config)
    
    def test_compute_confusion_matrix(self, model, config):
        """Test confusion matrix computation."""
        # Create mock dataloader with domain_id
        class MockDataset:
            def __iter__(self):
                for _ in range(5):
                    yield {
                        "input_ids": torch.randint(0, config.vocab_size, (2, 16)),
                        "attention_mask": torch.ones(2, 16),
                        "domain_id": torch.tensor([0, 1]),
                    }
        
        mock_loader = MockDataset()
        
        cm = DomainExpertMetrics.compute_confusion_matrix(
            model, mock_loader,
            num_domains=2, num_experts=config.num_experts,
            device=torch.device("cpu"), num_batches=3,
        )
        
        assert cm.shape == (2, config.num_experts)
        # Rows should sum to 100%
        assert torch.allclose(cm.sum(dim=1), torch.tensor([100.0, 100.0]), atol=0.1)


class TestDomainTrainingConfig:
    """Tests for domain training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DomainTrainingConfig()
        
        assert config.steps == 500
        assert config.lambda_route == 0.5
        assert config.lambda_kd == 0.2
        assert config.expert_dropout_prob == 0.2
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DomainTrainingConfig(
            steps=1000,
            lambda_route=1.0,
            lambda_kd=0.5,
        )
        
        assert config.steps == 1000
        assert config.lambda_route == 1.0
        assert config.lambda_kd == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])