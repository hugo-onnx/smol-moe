"""
Domain-Specialized Expert Training for Mixture of Experts.

This module implements techniques to train MoE models where each expert
specializes in a specific domain (e.g., chat, code, math). Key features:

1. **Domain-Supervised Routing (DSR)**: Cross-entropy loss encouraging
   the router to send domain tokens to their designated expert.

2. **Domain Prefix Embeddings**: Learnable prefix tokens prepended to
   inputs that signal the domain to the model.

3. **Curriculum Learning**: Gradually transition from domain-focused
   to uniform sampling during training.

4. **Expert Dropout**: Randomly mask router decisions to force
   domain-based routing during training.

5. **Knowledge Distillation**: Domain-conditional KD from a teacher model.

6. **Router Noise Injection**: Annealed Gaussian noise for exploration.

Reference:
    This implements ideas from various MoE papers including domain-specific
    routing and curriculum learning strategies.

Example usage:
    from smol_moe.domain_expert import (
        DomainExpertTrainer,
        DomainTrainingConfig,
        build_domain_dataloaders,
    )
    
    # Build domain-aware dataloaders
    train_loader, val_loader = build_domain_dataloaders(
        dataset_id="nvidia/Llama-Nemotron-Post-Training-Dataset",
        subset="SFT",
        splits=["chat", "code", "math"],
        tokenizer=tokenizer,
    )
    
    # Configure domain-specialized training
    config = DomainTrainingConfig(
        steps=500,
        lambda_route=0.5,
        lambda_kd=0.2,
    )
    
    # Train!
    trainer = DomainExpertTrainer(moe_model, train_loader, val_loader, config)
    metrics = trainer.train()
"""

import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


# =============================================================================
# Domain Mapping
# =============================================================================

# Default domain to expert mapping
DOMAIN_TO_ID: Dict[str, int] = {"chat": 0, "code": 1, "math": 2}
DOMAIN_TO_EXPERT: Dict[int, int] = {0: 0, 1: 1, 2: 2}


def get_domain_mapping(domains: List[str]) -> Tuple[Dict[str, int], Dict[int, int]]:
    """
    Create domain ID and domain-to-expert mappings.
    
    Args:
        domains: List of domain names.
        
    Returns:
        Tuple of (domain_to_id, domain_to_expert) mappings.
    """
    domain_to_id = {d: i for i, d in enumerate(domains)}
    domain_to_expert = {i: i for i in range(len(domains))}
    return domain_to_id, domain_to_expert


# =============================================================================
# Text Processing Utilities
# =============================================================================

def strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> annotations from model outputs.
    
    These are sometimes used for reasoning traces that should
    not be included in the final output.
    
    Args:
        text: Input text possibly containing think tags.
        
    Returns:
        Cleaned text with think tags removed.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # Remove incomplete <think>... at end
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    
    return text.strip()


def find_subsequence(sequence: List[int], pattern: List[int]) -> int:
    """
    Find the first occurrence of a pattern in a sequence.
    
    Args:
        sequence: Token sequence to search in.
        pattern: Token pattern to find.
        
    Returns:
        Starting index if found, -1 otherwise.
    """
    seq_len, pat_len = len(sequence), len(pattern)
    for i in range(seq_len - pat_len + 1):
        if sequence[i:i + pat_len] == pattern:
            return i
    return -1


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DomainTrainingConfig:
    """
    Configuration for domain-specialized MoE training.
    
    Attributes:
        steps: Total training steps.
        learning_rate: Base learning rate.
        router_lr_multiplier: LR multiplier for router parameters.
        prefix_lr_multiplier: LR multiplier for domain prefix embeddings.
        
        lambda_route: Weight for domain-supervised routing loss.
        lambda_kd: Weight for knowledge distillation loss.
        
        gate_tau_start: Initial gating temperature (soft routing).
        gate_tau_end: Final gating temperature (sharp routing).
        gate_tau_warmup_frac: Fraction of steps for temperature annealing.
        
        expert_dropout_prob: Probability of forcing domain-based routing.
        
        router_noise_start: Initial router noise std.
        router_noise_end: Final router noise std.
        router_noise_warmup_frac: Fraction of steps for noise annealing.
        
        kd_temperature: Temperature for knowledge distillation.
        
        prefix_length: Number of prefix tokens per domain.
        
        curriculum_start_probs: Initial domain sampling probabilities.
        curriculum_end_probs: Final domain sampling probabilities.
    """
    # Training basics
    steps: int = 500
    learning_rate: float = 5e-5
    router_lr_multiplier: float = 2.0
    prefix_lr_multiplier: float = 1.0
    weight_decay: float = 0.1
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    
    # Domain-supervised routing
    lambda_route: float = 0.5
    
    # Knowledge distillation
    lambda_kd: float = 0.2
    kd_temperature: float = 2.0
    
    # Gating temperature annealing
    gate_tau_start: float = 1.0
    gate_tau_end: float = 0.3
    gate_tau_warmup_frac: float = 0.2
    
    # Expert dropout
    expert_dropout_prob: float = 0.2
    
    # Router noise
    router_noise_start: float = 0.5
    router_noise_end: float = 0.0
    router_noise_warmup_frac: float = 0.3
    
    # Domain prefix embeddings
    prefix_length: int = 1
    
    # Curriculum learning
    curriculum_start_probs: Optional[List[float]] = None  # e.g., [0.8, 0.1, 0.1]
    curriculum_end_probs: Optional[List[float]] = None    # e.g., [0.33, 0.33, 0.34]
    
    # Reporting
    report_every: int = 10
    eval_max_batches: int = 20
    
    # Mixed precision
    use_amp: bool = True
    
    # Saving
    save_dir: Optional[str] = None


# =============================================================================
# Domain Prefix Embeddings
# =============================================================================

class DomainPrefixEmbedding(nn.Module):
    """
    Learnable domain-specific prefix embeddings.
    
    These are prepended to the input sequence to signal the domain
    to the model, helping the router make domain-aware decisions.
    
    Args:
        num_domains: Number of domains.
        prefix_length: Number of prefix tokens per domain.
        hidden_size: Model hidden dimension.
    """
    
    def __init__(
        self,
        num_domains: int,
        prefix_length: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_domains = num_domains
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        
        # Learnable embedding: [num_domains, prefix_length * hidden_size]
        self.embedding = nn.Embedding(num_domains, prefix_length * hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domain_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add domain prefix embeddings to input.
        
        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            domain_ids: Domain IDs [batch].
            embed_tokens: Token embedding layer from the model.
            
        Returns:
            Tuple of (inputs_embeds, new_attention_mask):
                - inputs_embeds: [batch, prefix_length + seq_len, hidden_size]
                - new_attention_mask: [batch, prefix_length + seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        token_embeds = embed_tokens(input_ids)  # [B, T, H]
        
        # Get domain prefix embeddings
        prefix_flat = self.embedding(domain_ids)  # [B, prefix_length * H]
        prefix = prefix_flat.view(batch_size, self.prefix_length, self.hidden_size)
        
        # Concatenate prefix + tokens
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        
        # Extend attention mask
        prefix_mask = torch.ones(
            batch_size, self.prefix_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return inputs_embeds, new_attention_mask


# =============================================================================
# Loss Functions
# =============================================================================

def causal_lm_loss_with_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute causal LM loss with explicit labels (supports -100 masking).
    
    Args:
        logits: Model logits [batch, seq_len, vocab_size].
        labels: Token labels [batch, seq_len], -100 for masked positions.
        attention_mask: Attention mask [batch, seq_len].
        
    Returns:
        Scalar loss tensor.
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()
    
    # Flatten
    batch_size, seq_len_minus_1, vocab_size = shift_logits.shape
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)
    
    # Cross-entropy with ignore_index=-100
    loss_per_token = F.cross_entropy(
        shift_logits, shift_labels,
        reduction='none',
        ignore_index=-100,
    )
    
    # Apply attention mask
    loss_per_token = loss_per_token * shift_mask
    
    # Normalize
    num_valid = shift_mask.sum().clamp(min=1.0)
    return loss_per_token.sum() / num_valid


def router_supervision_loss(
    model: nn.Module,
    domain_ids: torch.Tensor,
    domain_to_expert: Dict[int, int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Cross-entropy loss for domain-supervised routing.

    Encourages the router to send tokens from each domain to their
    designated expert based on the domain_to_expert mapping.

    Supervision is applied to ALL token positions, not just the prefix.
    Using only position 0 covers 1/T of the sequence, leaving the vast
    majority of routing decisions unsupervised and allowing pretraining
    biases (e.g. a dominant expert) to persist.

    Args:
        model: MoE model with router logits stored in layers.
        domain_ids: Domain ID for each sequence [batch].
        domain_to_expert: Mapping from domain ID to target expert.
        temperature: Softmax temperature for router logits.

    Returns:
        Scalar supervision loss.
    """
    router_logits_list = []

    for layer in model.model.layers:
        moe = getattr(layer, 'moe', None)
        if moe is not None and hasattr(moe, '_last_router_logits'):
            logits = moe._last_router_logits
            if logits is not None:
                router_logits_list.append(logits)  # [B, T, E]

    if not router_logits_list:
        return torch.tensor(0.0, device=domain_ids.device)

    # Average router logits across layers
    avg_logits = torch.stack(router_logits_list, dim=0).mean(dim=0)  # [B, T, E]
    B, T, E = avg_logits.shape

    # Create per-sequence target expert labels, then expand to all positions.
    # Shape [B] → [B, T] → [B*T] so every token position is supervised.
    targets = torch.tensor(
        [domain_to_expert[int(d.item())] for d in domain_ids],
        device=avg_logits.device,
        dtype=torch.long,
    )
    targets_all = targets.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]

    # Cross-entropy loss over all [B*T] positions
    all_logits = avg_logits.reshape(B * T, E)  # [B*T, E]
    log_probs = F.log_softmax(all_logits / temperature, dim=-1)
    return F.nll_loss(log_probs, targets_all)


def domain_conditional_kd_loss(
    teacher_model: nn.Module,
    student_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    domain_ids: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Domain-conditional knowledge distillation loss.
    
    Computes KL divergence between teacher and student distributions,
    separately for each domain, then averages.
    
    Args:
        teacher_model: Teacher (dense) model.
        student_model: Student (MoE) model.
        input_ids: Token IDs [batch, seq_len].
        attention_mask: Attention mask [batch, seq_len].
        domain_ids: Domain IDs [batch].
        temperature: Softmax temperature.
        
    Returns:
        Scalar KD loss.
    """
    # Get teacher predictions (no grad)
    with torch.no_grad():
        teacher_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_out.logits if hasattr(teacher_out, 'logits') else teacher_out['logits']
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Get student predictions
    student_out = student_model(input_ids=input_ids, attention_mask=attention_mask)
    student_logits = student_out['logits'] if isinstance(student_out, dict) else student_out.logits
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    # Compute KL divergence per domain
    losses = []
    for domain in torch.unique(domain_ids):
        mask = domain_ids == domain
        if mask.any():
            t_probs = teacher_probs[mask]
            s_logp = student_log_probs[mask]
            domain_loss = F.kl_div(s_logp, t_probs, reduction='batchmean')
            losses.append(domain_loss)
    
    if not losses:
        return torch.tensor(0.0, device=input_ids.device)
    
    # Scale by temperature squared (standard KD scaling)
    kd_loss = torch.stack(losses).mean() * (temperature ** 2)
    return kd_loss


# =============================================================================
# Training Augmentations
# =============================================================================

def apply_expert_dropout(
    model: nn.Module,
    domain_ids: torch.Tensor,
    domain_to_expert: Dict[int, int],
    prob: float = 0.2,
) -> None:
    """
    With probability `prob`, set a forced routing mask on all MoE layers so that
    each token is constrained to its domain-designated expert on the next forward pass.

    The mask is stored in ``moe._forced_routing_mask`` and applied inside
    ``MixtureOfExperts.forward()`` **before** the argmax routing decision.
    Call this function before the forward pass; reset the attribute after backward.

    Args:
        model: MoE model.
        domain_ids: Domain IDs [batch], already on the correct device.
        domain_to_expert: Domain to expert mapping.
        prob: Probability of applying dropout.
    """
    if random.random() > prob:
        return

    batch_size = domain_ids.shape[0]
    device = domain_ids.device

    # Infer num_experts from the first MoE layer
    num_experts = None
    for layer in model.model.layers:
        moe = getattr(layer, 'moe', None)
        if moe is not None:
            num_experts = moe.E
            break

    if num_experts is None:
        return

    # Build mask [B, 1, E]: -inf everywhere, 0 at the target expert.
    # The shape broadcasts over the T dimension inside MixtureOfExperts.forward().
    mask = torch.full((batch_size, 1, num_experts), float('-inf'), device=device)
    for b in range(batch_size):
        target_expert = domain_to_expert[int(domain_ids[b].item())]
        mask[b, 0, target_expert] = 0.0

    for layer in model.model.layers:
        moe = getattr(layer, 'moe', None)
        if moe is not None:
            moe._forced_routing_mask = mask


def apply_router_noise(
    model: nn.Module,
    step: int,
    noise_start: float,
    noise_end: float,
    warmup_frac: float,
    total_steps: int,
) -> None:
    """
    Set the router noise standard deviation on all MoE layers so that annealed
    Gaussian noise is injected into router logits on the next forward pass.

    The noise is applied inside ``MixtureOfExperts.forward()`` **before** the
    argmax routing decision. Call this function before the forward pass; reset
    ``moe._router_noise_std = 0.0`` after backward.

    Args:
        model: MoE model.
        step: Current training step.
        noise_start: Initial noise std.
        noise_end: Final noise std.
        warmup_frac: Fraction of steps for annealing.
        total_steps: Total training steps.
    """
    # Compute current noise level
    warmup_steps = int(total_steps * warmup_frac)
    if step <= warmup_steps:
        t = step / max(1, warmup_steps)
        sigma = (1 - t) * noise_start + t * noise_end
    else:
        sigma = noise_end

    for layer in model.model.layers:
        moe = getattr(layer, 'moe', None)
        if moe is not None:
            moe._router_noise_std = float(sigma)


def get_gate_temperature(
    step: int,
    tau_start: float,
    tau_end: float,
    warmup_frac: float,
    total_steps: int,
) -> float:
    """
    Compute annealed gating temperature.
    
    Temperature decreases to make routing decisions sharper over time.
    
    Args:
        step: Current step.
        tau_start: Initial temperature.
        tau_end: Final temperature.
        warmup_frac: Fraction of steps for annealing.
        total_steps: Total steps.
        
    Returns:
        Current temperature value.
    """
    warmup_steps = max(1, int(warmup_frac * total_steps))
    if step <= warmup_steps:
        t = step / warmup_steps
        return (1 - t) * tau_start + t * tau_end
    return tau_end


# =============================================================================
# Curriculum Learning
# =============================================================================

def get_curriculum_probs(
    step: int,
    total_steps: int,
    start_probs: List[float],
    end_probs: List[float],
) -> List[float]:
    """
    Compute domain sampling probabilities for curriculum learning.
    
    Linearly interpolates from start to end probabilities.
    
    Args:
        step: Current step.
        total_steps: Total steps.
        start_probs: Initial probabilities.
        end_probs: Final probabilities.
        
    Returns:
        Current probability distribution.
    """
    progress = step / max(1, total_steps)
    return [
        (1 - progress) * s + progress * e
        for s, e in zip(start_probs, end_probs)
    ]


class CurriculumBatcher:
    """
    Batcher that samples from domain-specific dataloaders according
    to a curriculum schedule.
    
    Early in training, sampling is biased toward certain domains.
    Over time, it anneals toward uniform sampling.
    """
    
    def __init__(
        self,
        domain_loaders: Dict[int, DataLoader],
        total_steps: int,
        start_probs: Optional[List[float]] = None,
        end_probs: Optional[List[float]] = None,
        batch_size: int = 4,
    ):
        """
        Args:
            domain_loaders: Mapping from domain ID to DataLoader.
            total_steps: Total training steps.
            start_probs: Initial sampling probabilities per domain.
            end_probs: Final sampling probabilities per domain.
            batch_size: Batch size for recreating exhausted loaders.
        """
        self.domain_loaders = domain_loaders
        self.domain_iters = {d: iter(loader) for d, loader in domain_loaders.items()}
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.step = 0
        
        num_domains = len(domain_loaders)
        self.start_probs = start_probs or [1.0 / num_domains] * num_domains
        self.end_probs = end_probs or [1.0 / num_domains] * num_domains
    
    def next_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch according to curriculum schedule."""
        # Get current sampling probabilities
        probs = get_curriculum_probs(
            self.step, self.total_steps,
            self.start_probs, self.end_probs,
        )
        
        # Sample domain
        domain_ids = list(self.domain_loaders.keys())
        domain = random.choices(domain_ids, weights=probs)[0]
        
        # Get batch from domain
        try:
            batch = next(self.domain_iters[domain])
        except StopIteration:
            # Reset iterator
            self.domain_iters[domain] = iter(self.domain_loaders[domain])
            batch = next(self.domain_iters[domain])
        
        self.step += 1
        return batch


# =============================================================================
# Metrics
# =============================================================================

class DomainExpertMetrics:
    """Metrics for domain-specialized MoE training."""
    
    @staticmethod
    @torch.no_grad()
    def expert_domain_alignment(
        model: nn.Module,
        dataloader: DataLoader,
        domain_to_expert: Dict[int, int],
        device: torch.device,
        num_batches: int = 5,
        use_amp: bool = False,
    ) -> float:
        """
        Measure expert-domain alignment score (EDAS).
        
        Computes the percentage of tokens routed to their
        expected domain-specific expert.
        
        Args:
            model: MoE model.
            dataloader: Validation dataloader with domain_id.
            domain_to_expert: Domain to expert mapping.
            device: Device to run on.
            num_batches: Number of batches to evaluate.
            use_amp: Use automatic mixed precision.
            
        Returns:
            Alignment percentage [0, 100].
        """
        was_training = model.training
        model.eval()
        
        correct, total = 0, 0
        data_iter = iter(dataloader)
        
        for _ in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            domain_ids = batch["domain_id"].to(device)
            
            # Forward pass
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Check routing decisions
            for layer in model.model.layers:
                moe = getattr(layer, 'moe', None)
                if moe is None or not hasattr(moe, '_last_router_logits'):
                    continue
                
                assignments = torch.argmax(moe._last_router_logits, dim=-1)  # [B, T]
                
                # Expected expert for each sequence
                targets = torch.tensor(
                    [domain_to_expert[int(d.item())] for d in domain_ids],
                    device=assignments.device,
                ).unsqueeze(1).expand_as(assignments)
                
                correct += (assignments == targets).sum().item()
                total += assignments.numel()
        
        if was_training:
            model.train()
        
        return 100.0 * correct / max(1, total)
    
    @staticmethod
    @torch.no_grad()
    def compute_confusion_matrix(
        model: nn.Module,
        dataloader: DataLoader,
        num_domains: int,
        num_experts: int,
        device: torch.device,
        num_batches: int = 10,
    ) -> torch.Tensor:
        """
        Compute confusion matrix of domain-to-expert routing.
        
        Args:
            model: MoE model.
            dataloader: Validation dataloader.
            num_domains: Number of domains.
            num_experts: Number of experts.
            device: Device.
            num_batches: Batches to process.
            
        Returns:
            Confusion matrix [num_domains, num_experts] with percentages.
        """
        model.eval()
        cm = torch.zeros(num_domains, num_experts, device=device)
        
        data_iter = iter(dataloader)
        for _ in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            domain_ids = batch["domain_id"].to(device)
            
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            for layer in model.model.layers:
                moe = getattr(layer, 'moe', None)
                if moe is None or not hasattr(moe, '_last_router_logits'):
                    continue
                
                assignments = torch.argmax(moe._last_router_logits, dim=-1)
                
                for b, domain in enumerate(domain_ids):
                    for expert in range(num_experts):
                        cm[domain, expert] += (assignments[b] == expert).sum()
        
        # Normalize rows to percentages
        row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1)
        cm = cm / row_sums * 100
        
        return cm.cpu()


# =============================================================================
# Dataset Building
# =============================================================================

def build_domain_dataloaders(
    dataset_id: str,
    tokenizer,
    splits: List[str],
    block_size: int = 256,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    val_fraction: float = 0.1,
    subset: Optional[str] = None,
    seed: int = 42,
    input_column: str = "input",
    output_column: str = "output",
) -> Tuple[DataLoader, DataLoader, Dict[int, DataLoader]]:
    """
    Build dataloaders for domain-specialized training.
    
    Each sample is formatted as:
        ### Input:
        {input text}
        
        ### Output:
        {output text}
    
    Labels mask the input portion so loss is only computed on outputs.
    
    Args:
        dataset_id: HuggingFace dataset ID.
        tokenizer: Tokenizer instance.
        splits: List of domain splits (e.g., ["chat", "code", "math"]).
        block_size: Maximum sequence length.
        batch_size: Batch size.
        max_samples: Max samples per domain.
        val_fraction: Validation split fraction.
        subset: Dataset subset name.
        seed: Random seed.
        input_column: Name of input column in dataset.
        output_column: Name of output column in dataset.
        
    Returns:
        Tuple of (train_loader, val_loader, domain_loaders).
    """
    try:
        from datasets import load_dataset, interleave_datasets
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    domain_to_id, _ = get_domain_mapping(splits)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    domain_datasets = []
    MARKER = "### Output:\n"
    
    for split in splits:
        # Load dataset
        if subset:
            ds = load_dataset(dataset_id, subset, split=split)
        else:
            ds = load_dataset(dataset_id, split=split)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        domain_id = domain_to_id[split]
        
        # Format samples
        def format_sample(example):
            inp = example[input_column]
            if isinstance(inp, list):
                inp = "\n".join(
                    f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
                    for m in inp
                )
            out = strip_think_tags(example[output_column])
            text = f"### Input:\n{inp}\n\n{MARKER}{out}"
            return {"text": text, "domain_id": domain_id}
        
        ds = ds.map(format_sample)
        
        # Tokenize
        marker_ids = tokenizer(MARKER, add_special_tokens=False)["input_ids"]
        pad_id = tokenizer.pad_token_id
        
        def tokenize(batch):
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_attention_mask=True,
            )
            
            # Create labels with masking
            labels = [row[:] for row in encoded["input_ids"]]
            
            for i, row in enumerate(labels):
                # Mask padding
                labels[i] = [t if t != pad_id else -100 for t in row]
                
                # Mask input (before output marker)
                marker_pos = find_subsequence(row, marker_ids)
                if marker_pos != -1:
                    cutoff = marker_pos + len(marker_ids)
                    labels[i][:cutoff] = [-100] * cutoff
            
            encoded["labels"] = labels
            encoded["domain_id"] = batch["domain_id"]
            return encoded
        
        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        domain_datasets.append(ds)
    
    # Interleave for balanced training
    balanced = interleave_datasets(
        domain_datasets,
        probabilities=[1.0 / len(splits)] * len(splits),
        seed=seed,
    )
    
    # Train/val split
    split_ds = balanced.train_test_split(
        test_size=val_fraction,
        seed=seed,
        shuffle=True,
    )
    
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]
    
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "domain_id"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "domain_id"])
    
    # Create main dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    
    # Create per-domain dataloaders for curriculum learning
    domain_loaders = {}
    for domain_id in domain_to_id.values():
        domain_subset = train_ds.filter(lambda x: x["domain_id"] == domain_id)
        domain_loaders[domain_id] = DataLoader(
            domain_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    
    return train_loader, val_loader, domain_loaders


# =============================================================================
# Forward with Prefix
# =============================================================================

def forward_with_prefix(
    model: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Forward pass using pre-computed embeddings (with prefix).
    
    Args:
        model: SmolMoE model.
        inputs_embeds: Input embeddings [batch, seq_len, hidden].
        attention_mask: Attention mask [batch, seq_len].
        
    Returns:
        Dictionary with 'logits' key.
    """
    hidden_states = inputs_embeds
    
    # Pass through decoder layers
    for layer in model.model.layers:
        layer_outputs = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = layer_outputs[0]
    
    # Final norm and LM head
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    
    return {"logits": logits.float()}


# =============================================================================
# Trainer
# =============================================================================

class DomainExpertTrainer:
    """
    Trainer for domain-specialized MoE expert routing.
    
    Combines multiple techniques:
    - Domain-supervised routing loss
    - Domain prefix embeddings
    - Curriculum learning
    - Expert dropout
    - Router noise injection
    - Knowledge distillation (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DomainTrainingConfig,
        domain_loaders: Optional[Dict[int, DataLoader]] = None,
        teacher_model: Optional[nn.Module] = None,
        domains: List[str] = ["chat", "code", "math"],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: SmolMoE model to train.
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            config: Training configuration.
            domain_loaders: Per-domain dataloaders for curriculum.
            teacher_model: Optional teacher for KD.
            domains: List of domain names.
            device: Device to train on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.teacher_model = teacher_model
        self.domains = domains
        
        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        if teacher_model:
            teacher_model.to(device)
            teacher_model.eval()
        
        # Domain mappings
        self.domain_to_id, self.domain_to_expert = get_domain_mapping(domains)
        
        # Mixed precision
        self.use_amp = config.use_amp and device.type == "cuda"
        
        # Domain prefix embeddings
        hidden_size = model.model.embed_tokens.embedding_dim
        self.prefix_emb = DomainPrefixEmbedding(
            num_domains=len(domains),
            prefix_length=config.prefix_length,
            hidden_size=hidden_size,
        ).to(device)
        
        # Curriculum batcher
        if domain_loaders:
            num_domains = len(domains)
            start_probs = config.curriculum_start_probs or [0.8] + [0.1] * (num_domains - 1)
            end_probs = config.curriculum_end_probs or [1.0 / num_domains] * num_domains
            
            self.curriculum_batcher = CurriculumBatcher(
                domain_loaders,
                config.steps,
                start_probs=start_probs,
                end_probs=end_probs,
            )
        else:
            self.curriculum_batcher = None
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics_history: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "route_loss": [],
            "kd_loss": [],
            "edas": [],
        }
        
        # Save directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with parameter groups."""
        router_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if ".moe.gate." in name:
                router_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {"params": other_params, "lr": self.config.learning_rate},
            {"params": router_params, "lr": self.config.learning_rate * self.config.router_lr_multiplier},
            {"params": self.prefix_emb.parameters(), "lr": self.config.learning_rate * self.config.prefix_lr_multiplier},
        ]
        
        return AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler."""
        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return float(step + 1) / float(max(1, self.config.warmup_steps))
            progress = (step - self.config.warmup_steps) / float(max(1, self.config.steps - self.config.warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        
        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    
    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
    ) -> Tuple[float, float, float]:
        """Execute one training step."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        domain_ids = batch["domain_id"].to(self.device)
        
        # Add domain prefix
        inputs_embeds, new_attention_mask = self.prefix_emb(
            input_ids, attention_mask, domain_ids,
            self.model.model.embed_tokens,
        )
        
        # Extend labels for prefix
        prefix_pad = torch.full(
            (labels.size(0), self.config.prefix_length),
            -100, dtype=labels.dtype, device=self.device,
        )
        new_labels = torch.cat([prefix_pad, labels], dim=1)
        
        # Apply augmentations BEFORE forward so they affect routing decisions
        apply_router_noise(
            self.model, step,
            self.config.router_noise_start, self.config.router_noise_end,
            self.config.router_noise_warmup_frac, self.config.steps,
        )
        apply_expert_dropout(
            self.model, domain_ids, self.domain_to_expert,
            self.config.expert_dropout_prob,
        )

        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = forward_with_prefix(self.model, inputs_embeds, new_attention_mask)
                logits = outputs["logits"]
                
                # Losses
                lm_loss = causal_lm_loss_with_labels(logits, new_labels, new_attention_mask)
                
                tau = get_gate_temperature(
                    step, self.config.gate_tau_start, self.config.gate_tau_end,
                    self.config.gate_tau_warmup_frac, self.config.steps,
                )
                route_loss = router_supervision_loss(
                    self.model, domain_ids, self.domain_to_expert, tau,
                )
                
                kd_loss = torch.tensor(0.0, device=self.device)
                if self.teacher_model is not None and self.config.lambda_kd > 0:
                    kd_loss = domain_conditional_kd_loss(
                        self.teacher_model, self.model,
                        input_ids, attention_mask, domain_ids,
                        self.config.kd_temperature,
                    )
                
                total_loss = (
                    lm_loss +
                    self.config.lambda_route * route_loss +
                    self.config.lambda_kd * kd_loss
                )
        else:
            outputs = forward_with_prefix(self.model, inputs_embeds, new_attention_mask)
            logits = outputs["logits"]
            
            lm_loss = causal_lm_loss_with_labels(logits, new_labels, new_attention_mask)
            
            tau = get_gate_temperature(
                step, self.config.gate_tau_start, self.config.gate_tau_end,
                self.config.gate_tau_warmup_frac, self.config.steps,
            )
            route_loss = router_supervision_loss(
                self.model, domain_ids, self.domain_to_expert, tau,
            )
            
            kd_loss = torch.tensor(0.0, device=self.device)
            if self.teacher_model is not None and self.config.lambda_kd > 0:
                kd_loss = domain_conditional_kd_loss(
                    self.teacher_model, self.model,
                    input_ids, attention_mask, domain_ids,
                    self.config.kd_temperature,
                )
            
            total_loss = (
                lm_loss +
                self.config.lambda_route * route_loss +
                self.config.lambda_kd * kd_loss
            )
        
        # Backward
        total_loss.backward()

        # Reset augmentation state so it does not bleed into evaluation
        for layer in self.model.model.layers:
            moe = getattr(layer, 'moe', None)
            if moe is not None:
                moe._router_noise_std = 0.0
                moe._forced_routing_mask = None
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.prefix_emb.parameters()),
            self.config.max_grad_norm,
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return lm_loss.item(), route_loss.item(), kd_loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            if num_batches >= self.config.eval_max_batches:
                break
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            domain_ids = batch["domain_id"].to(self.device)
            
            inputs_embeds, new_attention_mask = self.prefix_emb(
                input_ids, attention_mask, domain_ids,
                self.model.model.embed_tokens,
            )
            
            prefix_pad = torch.full(
                (labels.size(0), self.config.prefix_length),
                -100, dtype=labels.dtype, device=self.device,
            )
            new_labels = torch.cat([prefix_pad, labels], dim=1)
            
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = forward_with_prefix(self.model, inputs_embeds, new_attention_mask)
                    loss = causal_lm_loss_with_labels(outputs["logits"], new_labels, new_attention_mask)
            else:
                outputs = forward_with_prefix(self.model, inputs_embeds, new_attention_mask)
                loss = causal_lm_loss_with_labels(outputs["logits"], new_labels, new_attention_mask)
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / max(1, num_batches)
    
    def train(self) -> Dict[str, List[float]]:
        """Run training loop."""
        print(f"Starting domain-specialized training for {self.config.steps} steps")
        print(f"  Domains: {self.domains}")
        print(f"  Lambda route: {self.config.lambda_route}")
        print(f"  Lambda KD: {self.config.lambda_kd}")
        print()
        
        # Initial EDAS
        initial_edas = DomainExpertMetrics.expert_domain_alignment(
            self.model, self.val_loader, self.domain_to_expert,
            self.device, num_batches=5, use_amp=self.use_amp,
        )
        print(f"[Before Training] EDAS: {initial_edas:.1f}%\n")
        
        self.model.train()
        
        if self.curriculum_batcher:
            get_batch = lambda: self.curriculum_batcher.next_batch()
        else:
            train_iter = iter(self.train_loader)
            def get_batch():
                nonlocal train_iter
                try:
                    return next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    return next(train_iter)
        
        start_time = time.time()
        step_start = time.time()
        
        for step in range(1, self.config.steps + 1):
            batch = get_batch()
            lm_loss, route_loss, kd_loss = self._train_step(batch, step)
            
            if step % self.config.report_every == 0:
                eval_loss = self.evaluate()
                edas = DomainExpertMetrics.expert_domain_alignment(
                    self.model, self.val_loader, self.domain_to_expert,
                    self.device, num_batches=5, use_amp=self.use_amp,
                )
                
                self.metrics_history["train_loss"].append(lm_loss)
                self.metrics_history["eval_loss"].append(eval_loss)
                self.metrics_history["route_loss"].append(route_loss)
                self.metrics_history["kd_loss"].append(kd_loss)
                self.metrics_history["edas"].append(edas)
                
                elapsed = time.time() - step_start
                print(f"Step {step:4d}/{self.config.steps} | "
                      f"LM: {lm_loss:.3f} | Eval: {eval_loss:.3f} | "
                      f"Route: {route_loss:.3f} | KD: {kd_loss:.3f} | "
                      f"EDAS: {edas:.1f}% | Time: {elapsed:.1f}s")
                step_start = time.time()
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        
        return self.metrics_history
    
    def plot_confusion_matrix(self, num_batches: int = 50) -> torch.Tensor:
        """
        Plot and return the domain-expert confusion matrix.
        
        Requires matplotlib and seaborn.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Please install matplotlib and seaborn for plotting")
            return None
        
        num_experts = self.model.model.layers[0].moe.E
        cm = DomainExpertMetrics.compute_confusion_matrix(
            self.model, self.val_loader,
            len(self.domains), num_experts,
            self.device, num_batches,
        )
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm.numpy(),
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=[f"Expert {i}" for i in range(num_experts)],
            yticklabels=[d.capitalize() for d in self.domains],
            cbar_kws={'label': 'Routing (%)'},
        )
        plt.title("Expert-Domain Alignment (Confusion Matrix)")
        plt.xlabel("Expert")
        plt.ylabel("Domain")
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / "confusion_matrix.png", dpi=150)
        
        plt.show()
        return cm