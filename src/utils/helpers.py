"""
Utility functions for SmolMoE training, evaluation, and inference.

This module provides helper functions for:
- Timing and profiling
- Text generation with comparison
- Metrics tracking and visualization
- Model loading and saving
"""

import time
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps
from contextlib import contextmanager

import torch
import torch.nn as nn


# ============================================================================
# Timing Utilities
# ============================================================================

def timed(fn: Callable) -> Callable:
    """
    Decorator that prints execution time of a function.
    
    Example:
        @timed
        def slow_function():
            time.sleep(1)
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        total_time = time.perf_counter() - start_time
        print(f"time={total_time:.3f}s")
        return result
    return wrapper


def labelthis(label: str) -> Callable:
    """
    Decorator that assigns a label attribute to a function.
    
    Useful for organizing test cases or categorizing functions.
    
    Example:
        @labelthis("math_operations")
        def add(a, b):
            return a + b
    """
    def decorator(fn: Callable) -> Callable:
        fn.label = label
        return fn
    return decorator


def pretty_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "150 ms", "2.5 s", "1h 30m 45s")
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 60:
        return f"{seconds:.3f} s"
    
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    if hours < 1:
        return f"{int(minutes)}m {int(secs)}s"
    return f"{int(hours)}h {int(minutes)}m {int(secs)}s"


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing code blocks.
    
    Example:
        with timer("Data loading"):
            data = load_dataset()
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {pretty_duration(elapsed)}")


class Timer:
    """
    Reusable timer class for tracking multiple operations.
    
    Example:
        timer = Timer()
        timer.start("loading")
        # ... loading code ...
        timer.stop("loading")
        print(timer.summary())
    """
    
    def __init__(self):
        self._times: Dict[str, List[float]] = {}
        self._starts: Dict[str, float] = {}
    
    def start(self, name: str) -> None:
        """Start timing an operation."""
        self._starts[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing and return elapsed time."""
        if name not in self._starts:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.perf_counter() - self._starts.pop(name)
        
        if name not in self._times:
            self._times[name] = []
        self._times[name].append(elapsed)
        
        return elapsed
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all timed operations."""
        summary = {}
        for name, times in self._times.items():
            summary[name] = {
                "count": len(times),
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }
        return summary


# ============================================================================
# Generation Utilities
# ============================================================================

@timed
def _generate_tokens(
    model: nn.Module,
    tokenizer,
    inputs: Dict[str, torch.Tensor],
    num_tokens: int,
) -> str:
    """
    Generate tokens from a model (internal helper).
    
    Args:
        model: Language model with forward() returning {'logits': ...}
        tokenizer: Tokenizer with eos_token_id and decoding methods
        inputs: Dictionary with 'input_ids' and 'attention_mask'
        num_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text (excluding input)
    """
    model.eval()
    collected_ids = []
    
    with torch.no_grad():
        for _ in range(num_tokens):
            output = model(**inputs)
            next_token_id = torch.argmax(output['logits'][0, -1]).item()
            collected_ids.append(next_token_id)
            
            # Check for end of sequence
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Append to inputs for next iteration
            next_token = torch.tensor([[next_token_id]], device=inputs['input_ids'].device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=-1)
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(collected_ids)
    )


def generation_compare(
    prompt: str,
    num_tokens: int,
    tokenizer,
    model_a: nn.Module,
    model_b: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, str]:
    """
    Compare generations from one or two models.
    
    Args:
        prompt: Input text prompt
        num_tokens: Maximum tokens to generate
        tokenizer: Tokenizer for encoding/decoding
        model_a: First model to generate from
        model_b: Optional second model for comparison
        device: Device to run generation on
        
    Returns:
        Dictionary with generated texts keyed by model name
    """
    print(f"\n{'>' * 20}\n\tPrompt\n{'<' * 20}\n{prompt}\n\n")
    
    results = {}
    
    # Generate from model A
    model_inputs = tokenizer(prompt, return_tensors='pt')
    if device is not None:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    print(f"{'>' * 30}\n\tModel A Generation\n{'<' * 30}")
    results['model_a'] = _generate_tokens(model_a, tokenizer, model_inputs, num_tokens)
    print(results['model_a'])
    print("\n")
    
    # Generate from model B if provided
    if model_b is not None:
        model_inputs = tokenizer(prompt, return_tensors='pt')
        if device is not None:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        print(f"{'>' * 30}\n\tModel B Generation\n{'<' * 30}")
        results['model_b'] = _generate_tokens(model_b, tokenizer, model_inputs, num_tokens)
        print(results['model_b'])
    
    return results


# ============================================================================
# Metrics Utilities
# ============================================================================

def detach_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively detach and move metrics to CPU.
    
    Converts torch.Tensor values to Python scalars or lists for
    serialization and logging.
    
    Args:
        metrics: Dictionary of metric values
        
    Returns:
        Dictionary with all tensors converted to Python types
    """
    def to_python(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            return x.item() if x.dim() == 0 else x.tolist()
        elif isinstance(x, list):
            return [to_python(item) for item in x]
        elif isinstance(x, dict):
            return {k: to_python(v) for k, v in x.items()}
        return x
    
    return {k: to_python(v) for k, v in metrics.items()}


def plot_metrics(
    metrics: Dict[str, List[float]],
    x_vals: Optional[List[int]] = None,
    suptitle: str = "Training Metrics",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot training metrics in a grid layout.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        x_vals: Optional x-axis values (defaults to 1, 2, 3, ...)
        suptitle: Title for the entire figure
        save_path: If provided, save figure to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return
    
    metrics = detach_metrics(metrics)
    
    keys = list(metrics.keys())
    n = len(keys)
    
    if n == 0:
        print("No metrics to plot")
        return
    
    length = len(next(iter(metrics.values())))
    if x_vals is None:
        x_vals = list(range(1, length + 1))
    
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), constrained_layout=True)
    if n == 1:
        axes = [axes]
    
    palette = plt.cm.tab10.colors
    
    for i, (ax, key) in enumerate(zip(axes, keys)):
        y_vals = metrics[key]
        ax.plot(x_vals, y_vals, marker="o", color=palette[i % len(palette)])
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(suptitle)
    fig.supxlabel("Steps")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


class MetricsTracker:
    """
    Track and aggregate training metrics over time.
    
    Example:
        tracker = MetricsTracker()
        for step in range(100):
            loss = train_step()
            tracker.update({"loss": loss, "step": step})
        tracker.plot()
    """
    
    def __init__(self):
        self._history: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Add new metric values."""
        metrics = detach_metrics(metrics)
        for key, value in metrics.items():
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(value)
    
    def get(self, key: str) -> List[float]:
        """Get history for a specific metric."""
        return self._history.get(key, [])
    
    def latest(self) -> Dict[str, float]:
        """Get the most recent value for each metric."""
        return {k: v[-1] for k, v in self._history.items() if v}
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for key, values in self._history.items():
            if values:
                summary[key] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                }
        return summary
    
    def plot(self, **kwargs) -> None:
        """Plot all tracked metrics."""
        plot_metrics(self._history, **kwargs)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save metrics history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._history, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MetricsTracker":
        """Load metrics from JSON file."""
        tracker = cls()
        with open(path, 'r') as f:
            tracker._history = json.load(f)
        return tracker


# ============================================================================
# Model I/O Utilities
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: Union[str, Path],
    metrics: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optional optimizer state
        epoch: Current epoch number
        path: Path to save checkpoint
        metrics: Optional metrics to include
        **kwargs: Additional items to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint["metrics"] = detach_metrics(metrics)
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        strict: Whether to require exact key matching for model
        
    Returns:
        Dictionary with checkpoint contents
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        print(f"Loaded model state from {path}")
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded optimizer state from {path}")
    
    return checkpoint


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module) -> str:
    """
    Generate a summary of model architecture and parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Formatted summary string
    """
    lines = []
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append("-" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        lines.append(f"{name}: {list(param.shape)} ({num_params:,})")
    
    lines.append("-" * 60)
    lines.append(f"Total parameters: {total_params:,}")
    lines.append(f"Trainable parameters: {trainable_params:,}")
    lines.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    return "\n".join(lines)