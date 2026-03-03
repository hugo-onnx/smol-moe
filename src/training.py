"""
Training utilities for SmolMoE continued pretraining.

This module provides:
- Loss functions for causal language modeling
- Training loop with MoE-specific features
- Evaluation utilities
- MoE-specific metrics (expert utilization, entropy, etc.)
- Learning rate scheduling

Example usage:
    from smol_moe.training import Trainer, TrainingConfig
    
    config = TrainingConfig(steps=100, learning_rate=5e-5)
    trainer = Trainer(model, train_loader, val_loader, config)
    metrics = trainer.train()
"""

import math
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
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for MoE continued pretraining.
    
    Attributes:
        steps: Total number of training steps.
        learning_rate: Base learning rate for non-router parameters.
        router_lr_multiplier: Multiplier for router learning rate (encourages faster learning).
        weight_decay: AdamW weight decay.
        warmup_steps: Number of warmup steps for learning rate.
        lb_loss_coef: Coefficient for load balancing auxiliary loss.
        max_grad_norm: Maximum gradient norm for clipping.
        report_every: Report metrics every N steps.
        eval_max_batches: Maximum batches for evaluation.
        use_amp: Use automatic mixed precision (bfloat16).
        save_dir: Directory to save checkpoints and plots.
    """
    # Training
    steps: int = 100
    learning_rate: float = 5e-5
    router_lr_multiplier: float = 2.0
    weight_decay: float = 0.1
    warmup_steps: int = 10
    
    # MoE-specific
    lb_loss_coef: float = 0.01
    
    # Optimization
    max_grad_norm: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    
    # Reporting
    report_every: int = 10
    eval_max_batches: int = 20
    
    # Mixed precision
    use_amp: bool = True
    
    # Saving
    save_dir: Optional[str] = None
    save_every: Optional[int] = None


# =============================================================================
# Loss Functions
# =============================================================================

def causal_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute causal language modeling (next-token prediction) loss.
    
    The loss is computed by shifting logits and labels so that position i
    predicts token i+1. Padding tokens (where attention_mask=0) are ignored.
    
    Args:
        logits: Model outputs of shape [batch, seq_len, vocab_size].
        input_ids: Ground truth token IDs of shape [batch, seq_len].
        attention_mask: Optional mask of shape [batch, seq_len].
            1 for valid tokens, 0 for padding. If None, all tokens are valid.
    
    Returns:
        Scalar tensor containing the average cross-entropy loss.
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()   # [B, T-1]
    
    # Flatten for efficient computation
    batch_size, seq_len_minus_1, vocab_size = shift_logits.shape
    shift_logits = shift_logits.view(-1, vocab_size)  # [B*(T-1), V]
    shift_labels = shift_labels.view(-1)              # [B*(T-1)]
    
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous().float()
        shift_mask = shift_mask.view(-1)  # [B*(T-1)]
        
        # Compute per-token loss
        loss_per_token = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        
        # Mask and normalize
        masked_loss = loss_per_token * shift_mask
        num_valid = shift_mask.sum().clamp(min=1.0)
        loss = masked_loss.sum() / num_valid
    else:
        loss = F.cross_entropy(shift_logits, shift_labels)
    
    return loss


# =============================================================================
# MoE Metrics
# =============================================================================

class MoEMetrics:
    """
    Collection of metrics for monitoring MoE training.
    
    Available metrics:
    - utilization_entropy: Measures how evenly tokens are distributed across experts
    - expert_utilization: Per-expert token distribution
    - load_balancing_loss: Auxiliary loss for balanced routing
    - router_confidence: How confident the router is in its decisions
    """
    
    @staticmethod
    @torch.no_grad()
    def utilization_entropy(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 5,
        use_amp: bool = False,
    ) -> float:
        """
        Compute utilization entropy: measures expert usage balance.
        
        Entropy is normalized to [0, 100]%:
        - 0% = all tokens routed to one expert (worst)
        - 100% = tokens evenly distributed across all experts (best)
        
        This metric helps track whether the router is learning meaningful
        specialization or collapsing to a single expert.
        
        Args:
            model: MoE model with get_expert_utilization() method.
            dataloader: Validation dataloader.
            device: Device to run on.
            num_batches: Number of batches to process.
            use_amp: Use automatic mixed precision.
            
        Returns:
            Utilization entropy score in [0, 100].
        """
        was_training = model.training
        model.eval()
        
        # Run forward passes to collect routing statistics
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
            
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get utilization statistics
        utilization, _ = model.get_expert_utilization()
        
        if isinstance(utilization, (int, float)) and utilization == 0:
            score = 0.0
        else:
            # Normalize to probability distribution
            p = utilization.float().clamp(min=1e-8)
            p = p / p.sum()
            num_experts = p.numel()
            
            # Compute entropy
            entropy = -(p * p.log()).sum()
            max_entropy = math.log(num_experts)
            
            # Normalize to percentage
            score = float((entropy / max_entropy).item()) * 100.0
        
        if was_training:
            model.train()
        
        return score
    
    @staticmethod
    @torch.no_grad()
    def router_confidence(model: nn.Module) -> float:
        """
        Compute average router confidence across layers.
        
        Higher confidence means the router has stronger preferences for
        specific experts. Very high confidence early in training might
        indicate premature expert collapse.
        
        Returns:
            Average maximum softmax probability across all router decisions.
        """
        confidences = []
        
        for layer in model.model.layers:
            moe = getattr(layer, 'moe', None)
            if moe is not None and hasattr(moe, '_last_router_logits'):
                logits = moe._last_router_logits
                if logits is not None:
                    probs = F.softmax(logits, dim=-1)
                    max_probs = probs.max(dim=-1).values
                    confidences.append(max_probs.mean().item())
        
        return sum(confidences) / max(1, len(confidences)) * 100.0
    
    @staticmethod
    @torch.no_grad()
    def expert_usage_stats(model: nn.Module) -> Dict[str, Any]:
        """
        Get detailed expert usage statistics.
        
        Returns:
            Dictionary with:
            - utilization: per-expert utilization
            - most_used: index of most used expert
            - least_used: index of least used expert
            - imbalance_ratio: ratio of max to min utilization
        """
        utilization, lb_loss = model.get_expert_utilization()
        
        if isinstance(utilization, (int, float)) and utilization == 0:
            return {"utilization": [], "most_used": -1, "least_used": -1, "imbalance_ratio": 0}
        
        util_list = utilization.tolist()
        max_util = max(util_list)
        min_util = min(util_list)
        
        return {
            "utilization": util_list,
            "most_used": util_list.index(max_util),
            "least_used": util_list.index(min_util),
            "imbalance_ratio": max_util / max(min_util, 1e-8),
            "lb_loss": lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss,
        }


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Trainer for MoE continued pretraining.
    
    Features:
    - Separate learning rates for router vs other parameters
    - Load balancing loss integration
    - MoE-specific metrics tracking
    - Mixed precision training
    - Gradient clipping
    - Checkpoint saving
    
    Example:
        trainer = Trainer(model, train_loader, val_loader, config)
        metrics = trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The SmolMoE model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration.
            device: Device to train on. Auto-detected if None.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        
        # Mixed precision
        self.use_amp = config.use_amp and device.type == "cuda"
        self.autocast_dtype = torch.bfloat16 if self.use_amp else None
        
        # Optimizer with separate router LR
        self.optimizer = self._create_optimizer()
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            config.warmup_steps,
            config.steps,
        )
        
        # Metrics storage
        self.metrics_history: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "lb_loss": [],
            "utilization_entropy": [],
            "router_confidence": [],
            "learning_rate": [],
        }
        
        # Save directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with separate router learning rate."""
        router_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Router parameters get higher LR
            if ".moe.gate." in name or name.endswith(".moe.gate.weight"):
                router_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {"params": other_params, "lr": self.config.learning_rate},
            {"params": router_params, "lr": self.config.learning_rate * self.config.router_lr_multiplier},
        ]
        
        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )
    
    def _get_batch(self, data_iter: Iterator) -> Tuple[torch.Tensor, Optional[torch.Tensor], Iterator]:
        """Get next batch, resetting iterator if needed."""
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(self.train_loader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)
        
        return input_ids, attention_mask, data_iter
    
    def _train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        """
        Execute one training step.
        
        Returns:
            Tuple of (lm_loss, lb_loss) as floats.
        """
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                lm_loss = causal_lm_loss(logits, input_ids, attention_mask)
                _, lb_loss = self.model.get_expert_utilization()
                total_loss = lm_loss + self.config.lb_loss_coef * lb_loss
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            lm_loss = causal_lm_loss(logits, input_ids, attention_mask)
            _, lb_loss = self.model.get_expert_utilization()
            total_loss = lm_loss + self.config.lb_loss_coef * lb_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm,
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return lm_loss.item(), lb_loss.item()
    
    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> float:
        """
        Evaluate model on validation set.
        
        Args:
            max_batches: Maximum batches to evaluate. Uses config default if None.
            
        Returns:
            Average validation loss.
        """
        if max_batches is None:
            max_batches = self.config.eval_max_batches
        
        was_training = self.model.training
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        data_iter = iter(self.val_loader)
        for _ in range(max_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = causal_lm_loss(outputs["logits"], input_ids, attention_mask)
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = causal_lm_loss(outputs["logits"], input_ids, attention_mask)
            
            total_loss += loss.item()
            num_batches += 1
        
        if was_training:
            self.model.train()
        
        return total_loss / max(1, num_batches)
    
    def _report(self, step: int, train_loss: float, lb_loss: float, elapsed: float) -> None:
        """Report training progress."""
        eval_loss = self.evaluate()
        
        # Compute MoE metrics
        util_entropy = MoEMetrics.utilization_entropy(
            self.model, self.val_loader, self.device,
            num_batches=5, use_amp=self.use_amp,
        )
        router_conf = MoEMetrics.router_confidence(self.model)
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Store metrics
        self.metrics_history["train_loss"].append(train_loss)
        self.metrics_history["eval_loss"].append(eval_loss)
        self.metrics_history["lb_loss"].append(lb_loss)
        self.metrics_history["utilization_entropy"].append(util_entropy)
        self.metrics_history["router_confidence"].append(router_conf)
        self.metrics_history["learning_rate"].append(current_lr)
        
        # Print report
        print(f"Step {step:4d}/{self.config.steps} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Eval Loss: {eval_loss:.4f} | "
              f"LB Loss: {lb_loss:.4f} | "
              f"Util Entropy: {util_entropy:.1f}% | "
              f"Time: {self._format_time(elapsed)}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, secs = divmod(seconds, 60)
        return f"{int(minutes)}m {int(secs)}s"
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run the training loop.
        
        Returns:
            Dictionary of metrics history.
        """
        print(f"Starting training for {self.config.steps} steps on {self.device}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Router LR multiplier: {self.config.router_lr_multiplier}")
        print(f"  Load balancing coefficient: {self.config.lb_loss_coef}")
        print(f"  Mixed precision: {self.use_amp}")
        print()
        
        # Initial metrics
        initial_entropy = MoEMetrics.utilization_entropy(
            self.model, self.val_loader, self.device,
            num_batches=5, use_amp=self.use_amp,
        )
        print(f"[Before Training] Utilization Entropy: {initial_entropy:.1f}%\n")
        
        self.model.train()
        train_iter = iter(self.train_loader)
        start_time = time.time()
        step_start = time.time()
        
        for step in range(1, self.config.steps + 1):
            # Get batch
            input_ids, attention_mask, train_iter = self._get_batch(train_iter)
            
            # Train step
            train_loss, lb_loss = self._train_step(input_ids, attention_mask)
            
            # Report
            if step % self.config.report_every == 0:
                elapsed = time.time() - step_start
                self._report(step, train_loss, lb_loss, elapsed)
                step_start = time.time()
                
                # Save checkpoint
                if self.save_dir and self.config.save_every and step % self.config.save_every == 0:
                    self._save_checkpoint(step)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {self._format_time(total_time)}")
        
        # Final checkpoint
        if self.save_dir:
            self._save_checkpoint(self.config.steps, final=True)
        
        return self.metrics_history
    
    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """Save a training checkpoint."""
        from smol_moe.utils import save_checkpoint
        
        suffix = "final" if final else f"step_{step}"
        path = self.save_dir / f"checkpoint_{suffix}.pt"
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=step,
            path=path,
            metrics=self.metrics_history,
        )
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot training metrics."""
        from smol_moe.utils import plot_metrics
        
        # Prepare metrics for plotting
        steps = [
            self.config.report_every * (i + 1)
            for i in range(len(self.metrics_history["train_loss"]))
        ]
        
        # Split into training and MoE metrics
        training_metrics = {
            "Train Loss": self.metrics_history["train_loss"],
            "Eval Loss": self.metrics_history["eval_loss"],
            "LB Loss": self.metrics_history["lb_loss"],
        }
        
        moe_metrics = {
            "Utilization Entropy (%)": self.metrics_history["utilization_entropy"],
            "Router Confidence (%)": self.metrics_history["router_confidence"],
        }
        
        plot_metrics(training_metrics, x_vals=steps, suptitle="Training Metrics", save_path=save_path)
        
        if save_path:
            moe_save_path = save_path.replace(".png", "_moe.png")
        else:
            moe_save_path = None
        plot_metrics(moe_metrics, x_vals=steps, suptitle="MoE Metrics", save_path=moe_save_path)


# =============================================================================
# Dataset Utilities
# =============================================================================

def build_dataloaders(
    dataset_id: str,
    tokenizer,
    block_size: int = 256,
    batch_size: int = 4,
    max_samples: int = 1000,
    val_fraction: float = 0.2,
    subset: Optional[str] = None,
    split: str = "train",
    text_column: str = "text",
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation dataloaders from a HuggingFace dataset.
    
    Args:
        dataset_id: HuggingFace dataset identifier.
        tokenizer: Tokenizer with encode/decode methods.
        block_size: Sequence length for training.
        batch_size: Batch size.
        max_samples: Maximum samples to use from dataset.
        val_fraction: Fraction of data for validation.
        subset: Dataset subset/config name.
        split: Dataset split.
        text_column: Name of text column in dataset.
        seed: Random seed for reproducibility.
        num_workers: DataLoader workers.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Load dataset
    if subset:
        ds = load_dataset(dataset_id, subset, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)
    
    ds = ds.select(range(min(max_samples, len(ds))))
    
    # Tokenize
    eos_token_id = tokenizer.eos_token_id
    
    def tokenize(batch):
        encoded = tokenizer(
            batch[text_column],
            add_special_tokens=False,
            return_attention_mask=True,
        )
        # Add EOS token
        encoded["input_ids"] = [ids + [eos_token_id] for ids in encoded["input_ids"]]
        encoded["attention_mask"] = [m + [1] for m in encoded["attention_mask"]]
        return encoded
    
    ds = ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("input_ids", "attention_mask")],
    )
    
    # Group into fixed-length blocks
    def group_into_blocks(batch):
        out_ids = []
        out_masks = []
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            length = len(ids)
            num_blocks = length // block_size
            for i in range(num_blocks):
                start = i * block_size
                end = start + block_size
                out_ids.append(ids[start:end])
                out_masks.append(mask[start:end])
        return {"input_ids": out_ids, "attention_mask": out_masks}
    
    ds = ds.map(group_into_blocks, batched=True)
    
    # Train/val split
    ds = ds.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
    train_ds = ds["train"]
    val_ds = ds["test"]
    
    # Set format
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader