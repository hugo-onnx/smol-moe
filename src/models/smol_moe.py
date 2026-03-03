"""
SmolMoE Language Model Implementation.

This module implements the complete SmolMoE (Small Mixture of Experts) language model,
consisting of stacked decoder layers with MoE feedforward networks.

Architecture overview:
- Token embeddings with optional weight tying
- N decoder layers, each containing:
  - Pre-norm RMSNorm + RoPE Multi-Head Attention
  - Pre-norm RMSNorm + MoE with SwiGLU experts
- Final RMSNorm + Language model head

The model supports:
- Grouped-Query Attention (GQA) for efficient KV caching
- Rotary Position Embeddings (RoPE) for position encoding
- Sparse Mixture of Experts for scaling model capacity
- Load balancing loss for expert utilization
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from .components import (
    RMSNorm,
    RoPEAttention,
    MixtureOfExperts,
)


class SmolMoEDecoderLayer(nn.Module):
    """
    Single decoder layer of the SmolMoE model.
    
    Each layer consists of:
    1. Pre-norm attention with residual connection
    2. Pre-norm MoE feedforward with residual connection
    
    Args:
        config: Model configuration object
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Self-attention with RoPE
        self.self_attn = RoPEAttention(config)
        
        # Mixture of Experts feedforward
        self.moe = MixtureOfExperts(
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            emb_dim=config.hidden_size,
            moe_dim=config.intermediate_size,
        )
        
        # Pre-normalization layers
        eps = getattr(config, 'rms_norm_eps', 1e-5)
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_moe_rmsnorm = RMSNorm(config.hidden_size, eps=eps)
    
    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create causal attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device for the mask tensor
            
        Returns:
            Causal mask of shape [1, 1, seq_len, seq_len]
        """
        # Upper triangular matrix with -inf above diagonal
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        # Add batch and head dimensions for broadcasting
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Forward pass through the decoder layer.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq, hidden_size]
            attention_mask: Optional pre-computed attention mask
            
        Returns:
            Tuple containing output hidden states
        """
        # Create causal mask
        seq_len = hidden_states.size(1)
        causal_mask = self._create_causal_mask(seq_len, hidden_states.device)

        # Combine with padding mask so that padded key positions are never attended to.
        # attention_mask: [B, T], 1 = valid token, 0 = padding.
        # Convert to additive float mask [B, 1, 1, T]: 0 for valid, -inf for padding.
        if attention_mask is not None:
            padding_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype)
            padding_mask = padding_mask.masked_fill(attention_mask == 0, float('-inf'))
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            causal_mask = causal_mask + padding_mask

        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.pre_attn_rmsnorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
        )
        hidden_states = hidden_states + residual
        
        # MoE feedforward with pre-norm and residual
        residual = hidden_states
        hidden_states = self.pre_moe_rmsnorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = hidden_states + residual
        
        return (hidden_states,)


class SmolMoEModel(nn.Module):
    """
    SmolMoE transformer backbone (without language model head).
    
    This is the core transformer that processes token embeddings through
    N decoder layers to produce contextualized representations.
    
    Args:
        config: Model configuration object
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            SmolMoEDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        eps = getattr(config, 'rms_norm_eps', 1e-5)
        self.norm = RMSNorm(config.hidden_size, eps=eps)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass through the transformer backbone.
        
        Args:
            input_ids: Token IDs of shape [batch, seq]
            attention_mask: Attention mask (unused in current implementation)
            inputs_embeds: Optional pre-computed embeddings
            
        Returns:
            List containing final hidden states of shape [batch, seq, hidden_size]
        """
        # Get embeddings
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        
        # Pass through decoder layers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return [hidden_states]


class SmolMoEForCausalLM(nn.Module):
    """
    SmolMoE model with language modeling head for causal language modeling.
    
    This is the complete model used for text generation. It combines
    the transformer backbone with a linear head that projects hidden
    states to vocabulary logits.
    
    Features:
    - Optional weight tying between embeddings and LM head
    - Expert utilization tracking for monitoring
    - Load balancing loss computation
    
    Args:
        config: Model configuration object
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Transformer backbone
        self.model = SmolMoEModel(config)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying: share embedding and output weights
        tie_weights = getattr(config, 'tie_word_embeddings', True)
        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the module."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.
        
        Args:
            input_ids: Token IDs of shape [batch, seq]
            attention_mask: Attention mask (currently unused)
            labels: Optional labels for loss computation
            
        Returns:
            Dictionary containing:
                - logits: Vocabulary logits of shape [batch, seq, vocab_size]
                - loss: Cross-entropy loss (if labels provided)
        """
        # Get hidden states from backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # Ensure float32 for loss computation
        
        result = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross-entropy loss
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss
        
        return result
    
    def get_expert_utilization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get expert utilization statistics and load balancing loss.
        
        Returns:
            Tuple of (expert_utilization, load_balancing_loss):
                - expert_utilization: Mean utilization per expert across layers
                - load_balancing_loss: Mean auxiliary loss for load balancing
        """
        utilizations = []
        aux_losses = []
        
        for layer in self.model.layers:
            moe = getattr(layer, 'moe', None)
            if moe is not None:
                if hasattr(moe, '_expert_utilization') and moe._expert_utilization is not None:
                    utilizations.append(moe._expert_utilization)
                if hasattr(moe, '_aux_lb') and moe._aux_lb is not None:
                    aux_losses.append(moe._aux_lb)
        
        if len(utilizations) == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Average across layers
        expert_utilization = torch.stack(utilizations, dim=0).mean(dim=0)
        lb_loss = torch.stack(aux_losses, dim=0).mean()
        
        return expert_utilization, lb_loss
    
    def get_router_logits(self) -> List[Optional[torch.Tensor]]:
        """
        Get router logits from all MoE layers.
        
        Useful for computing additional losses (e.g., domain routing).
        
        Returns:
            List of router logits tensors, one per layer
        """
        router_logits = []
        for layer in self.model.layers:
            moe = getattr(layer, 'moe', None)
            if moe is not None and hasattr(moe, '_last_router_logits'):
                router_logits.append(moe._last_router_logits)
            else:
                router_logits.append(None)
        return router_logits
    
    def reset_weights_and_metrics(self) -> None:
        """
        Reset all weights and MoE metrics.
        
        Useful for reinitializing the model for new training runs.
        """
        with torch.no_grad():
            # Reset all module weights
            for module in self.modules():
                if module is self:
                    continue
                
                # Try standard reset methods
                reset_fn = getattr(module, 'reset_parameters_', None) or \
                          getattr(module, 'reset_parameters', None)
                if callable(reset_fn):
                    reset_fn()
                    continue
                
                # Manual reset for modules without reset method
                for name, param in module.named_parameters(recurse=False):
                    if param.dim() == 1:
                        if name == 'bias':
                            param.zero_()
                        else:
                            param.fill_(1.0)
                    else:
                        nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            
            # Reset MoE metrics
            for layer in self.model.layers:
                moe = getattr(layer, 'moe', None)
                if moe is not None:
                    moe._expert_utilization = None
                    moe._aux_lb = None
                    moe._last_router_logits = None
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial token IDs of shape [batch, seq]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top-k tokens
            top_p: If set, use nucleus sampling with this probability mass
            eos_token_id: If set, stop generation when this token is produced
            
        Returns:
            Generated token IDs including input, shape [batch, seq + generated]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get logits for the last position
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :]
            
            # Apply temperature
            if 0 < temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
        
        return input_ids


# Backward compatible alias
smolMoELM = SmolMoEForCausalLM