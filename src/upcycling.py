"""
Upcycling: Convert Dense Models to Mixture of Experts.

This module implements "Sparse Upcycling" - a technique to transform pre-trained
dense models into Mixture of Experts (MoE) models by replicating the feedforward
network weights across all experts.

The key insight is that by initializing all experts with identical weights from
the dense model's FFN, and zeroing the router, the MoE model initially behaves
identically to the dense model. This provides a strong initialization for
subsequent MoE fine-tuning.

Reference:
    Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from
    Dense Checkpoints", https://arxiv.org/abs/2212.05055

Supported dense model architectures:
    - LLaMA-style (model.model.layers, model.embed_tokens)
    - GPT-style (model.transformer.h, transformer.wte)

Example usage:
    from transformers import AutoModelForCausalLM
    from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
    from smol_moe.upcycling import upcycle_dense_to_moe
    
    # Load dense model
    dense_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    # Create MoE model with matching config
    config = SmolMoEConfig(num_experts=4)
    moe_model = SmolMoEForCausalLM(config)
    
    # Upcycle!
    upcycle_dense_to_moe(dense_model, moe_model)
    
    # Now moe_model produces identical outputs to dense_model
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn


class UpcyclingError(Exception):
    """Exception raised for errors during the upcycling process."""
    pass


# =============================================================================
# Layer Access Utilities
# =============================================================================

def get_dense_layers(dense_model: nn.Module) -> nn.ModuleList:
    """
    Retrieve decoder layers from a dense model.
    
    Supports multiple architectures:
    - LLaMA-style: model.model.layers
    - GPT-style: model.transformer.h
    
    Args:
        dense_model: The dense (non-MoE) language model.
        
    Returns:
        ModuleList of decoder layers.
        
    Raises:
        UpcyclingError: If decoder layers cannot be located.
    """
    # LLaMA-style models (LLaMA, Mistral, SmolLM, etc.)
    if hasattr(dense_model, "model") and hasattr(dense_model.model, "layers"):
        return dense_model.model.layers
    
    # GPT-style models (GPT-2, GPT-Neo, etc.)
    if hasattr(dense_model, "transformer") and hasattr(dense_model.transformer, "h"):
        return dense_model.transformer.h
    
    raise UpcyclingError(
        "Could not find decoder layers in the dense model. "
        "Expected 'model.model.layers' (LLaMA-style) or 'model.transformer.h' (GPT-style)."
    )


def get_moe_layers(moe_model: nn.Module) -> nn.ModuleList:
    """
    Retrieve decoder layers from a SmolMoE model.
    
    Args:
        moe_model: The MoE model.
        
    Returns:
        ModuleList of decoder layers.
        
    Raises:
        UpcyclingError: If decoder layers cannot be located.
    """
    if hasattr(moe_model, "model") and hasattr(moe_model.model, "layers"):
        return moe_model.model.layers
    
    raise UpcyclingError(
        "Could not find decoder layers in the MoE model. "
        "Expected 'model.model.layers'."
    )


# =============================================================================
# Weight Copying Utilities
# =============================================================================

def _copy_tensor(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy tensor with automatic dtype and device conversion."""
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def copy_embeddings(dense_model: nn.Module, moe_model: nn.Module) -> None:
    """
    Copy embedding weights from dense model to MoE model.
    
    Handles:
    - LLaMA-style: model.embed_tokens
    - GPT-style: transformer.wte
    
    Note: If weight tying is enabled (default), the lm_head weights are
    automatically updated since they share the same tensor.
    
    Args:
        dense_model: Source dense model.
        moe_model: Target MoE model.
        
    Raises:
        UpcyclingError: If embedding weights cannot be found.
    """
    # Find source embeddings
    if hasattr(dense_model, "model") and hasattr(dense_model.model, "embed_tokens"):
        src_embed = dense_model.model.embed_tokens.weight.data
    elif hasattr(dense_model, "transformer") and hasattr(dense_model.transformer, "wte"):
        src_embed = dense_model.transformer.wte.weight.data
    else:
        raise UpcyclingError(
            "Could not locate embedding weights in dense model. "
            "Expected 'model.embed_tokens' or 'transformer.wte'."
        )
    
    # Copy to MoE model
    dst_embed = moe_model.model.embed_tokens.weight.data
    _copy_tensor(src_embed, dst_embed)


def copy_final_norm(dense_model: nn.Module, moe_model: nn.Module) -> None:
    """
    Copy final normalization layer weights.
    
    Typical in LLaMA-style models: model.norm (RMSNorm).
    
    Args:
        dense_model: Source dense model.
        moe_model: Target MoE model.
    """
    # LLaMA-style final norm
    if hasattr(dense_model, "model") and hasattr(dense_model.model, "norm"):
        src_weight = dense_model.model.norm.weight.data
        dst_weight = moe_model.model.norm.weight.data
        _copy_tensor(src_weight, dst_weight)
    
    # GPT-style final norm (ln_f)
    elif hasattr(dense_model, "transformer") and hasattr(dense_model.transformer, "ln_f"):
        src_weight = dense_model.transformer.ln_f.weight.data
        dst_weight = moe_model.model.norm.weight.data
        _copy_tensor(src_weight, dst_weight)


def copy_attention_weights(dense_layer: nn.Module, moe_layer: nn.Module) -> None:
    """
    Copy self-attention projection weights from dense to MoE layer.
    
    Mapping (LLaMA-style):
        dense.self_attn.q_proj -> moe.self_attn.W_query
        dense.self_attn.k_proj -> moe.self_attn.W_key
        dense.self_attn.v_proj -> moe.self_attn.W_value
        dense.self_attn.o_proj -> moe.self_attn.W_output
    
    Args:
        dense_layer: One decoder block from the dense model.
        moe_layer: Corresponding decoder block from the MoE model.
        
    Raises:
        UpcyclingError: If attention projections are not found.
    """
    attn = getattr(dense_layer, "self_attn", None)
    if attn is None:
        raise UpcyclingError("Dense layer missing 'self_attn' module.")
    
    # Get source weights
    try:
        q_weight = attn.q_proj.weight.data
        k_weight = attn.k_proj.weight.data
        v_weight = attn.v_proj.weight.data
        o_weight = attn.o_proj.weight.data
    except AttributeError as e:
        raise UpcyclingError(f"Dense attention missing projection: {e}")
    
    # Copy to MoE attention
    moe_attn = moe_layer.self_attn
    _copy_tensor(q_weight, moe_attn.W_query.weight.data)
    _copy_tensor(k_weight, moe_attn.W_key.weight.data)
    _copy_tensor(v_weight, moe_attn.W_value.weight.data)
    _copy_tensor(o_weight, moe_attn.W_output.weight.data)


def copy_layer_norms(dense_layer: nn.Module, moe_layer: nn.Module) -> None:
    """
    Copy normalization layer weights from dense to MoE layer.
    
    Mapping (LLaMA-style):
        dense.input_layernorm -> moe.pre_attn_rmsnorm
        dense.post_attention_layernorm -> moe.pre_moe_rmsnorm
    
    Args:
        dense_layer: One decoder block from the dense model.
        moe_layer: Corresponding decoder block from the MoE model.
    """
    # Pre-attention norm
    if hasattr(dense_layer, "input_layernorm"):
        src = dense_layer.input_layernorm.weight.data
        dst = moe_layer.pre_attn_rmsnorm.weight.data
        _copy_tensor(src, dst)
    
    # Pre-MLP/MoE norm
    if hasattr(dense_layer, "post_attention_layernorm"):
        src = dense_layer.post_attention_layernorm.weight.data
        dst = moe_layer.pre_moe_rmsnorm.weight.data
        _copy_tensor(src, dst)


def copy_ffn_to_experts(
    dense_layer: nn.Module,
    moe_layer: nn.Module,
    zero_router: bool = True,
) -> None:
    """
    Convert dense FFN to MoE by replicating weights across all experts.
    
    The dense FFN uses SwiGLU activation:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Weight mapping (with transpose for bank layout):
        dense.mlp.gate_proj: [H, D] -> transpose -> [D, H] -> gate_bank[e]
        dense.mlp.up_proj:   [H, D] -> transpose -> [D, H] -> up_bank[e]
        dense.mlp.down_proj: [D, H] -> transpose -> [H, D] -> down_bank[e]
    
    By replicating identical weights to all experts and zeroing the router,
    the MoE layer produces identical outputs to the dense FFN regardless of
    which expert is selected (since they're all the same).
    
    Args:
        dense_layer: One decoder block from the dense model.
        moe_layer: Corresponding decoder block from the MoE model.
        zero_router: If True, zero the router weights for deterministic routing.
        
    Raises:
        UpcyclingError: If the dense layer lacks an MLP block.
    """
    mlp = getattr(dense_layer, "mlp", None)
    if mlp is None:
        raise UpcyclingError("Dense layer missing 'mlp' module.")
    
    # Get dense FFN weights
    try:
        gate_weight = mlp.gate_proj.weight.data  # [H, D]
        up_weight = mlp.up_proj.weight.data      # [H, D]
        down_weight = mlp.down_proj.weight.data  # [D, H]
    except AttributeError as e:
        raise UpcyclingError(f"Dense MLP missing projection: {e}")
    
    # Transpose to match MoE bank layout [E, D, H] or [E, H, D]
    gate_T = gate_weight.t()  # [D, H]
    up_T = up_weight.t()      # [D, H]
    down_T = down_weight.t()  # [H, D]
    
    moe = moe_layer.moe
    num_experts = moe.E
    device = moe.gate_bank.device
    dtype = moe.gate_bank.dtype
    
    # Replicate to all experts
    with torch.no_grad():
        for e in range(num_experts):
            moe.gate_bank.data[e].copy_(gate_T.to(device=device, dtype=dtype))
            moe.up_bank.data[e].copy_(up_T.to(device=device, dtype=dtype))
            moe.down_bank.data[e].copy_(down_T.to(device=device, dtype=dtype))
        
        # Zero router for deterministic behavior
        if zero_router:
            moe.gate.weight.zero_()


# =============================================================================
# Main Upcycling Function
# =============================================================================

def upcycle_dense_to_moe(
    dense_model: nn.Module,
    moe_model: nn.Module,
    zero_router: bool = True,
    verbose: bool = True,
) -> None:
    """
    Upcycle a dense model into a Mixture of Experts model.
    
    This function copies all weights from a pre-trained dense model into
    a SmolMoE model, replicating the FFN weights across all experts. After
    upcycling, the MoE model produces identical outputs to the dense model.
    
    The upcycling process:
    1. Copy embedding weights (and lm_head if weight-tied)
    2. For each layer:
       - Copy attention weights (Q, K, V, O projections)
       - Copy layer normalization weights
       - Replicate FFN weights to all experts
       - Zero router weights (optional, for deterministic routing)
    3. Copy final normalization weights
    
    Args:
        dense_model: Source pre-trained dense model (e.g., from HuggingFace).
        moe_model: Target SmolMoE model to receive the weights.
        zero_router: If True, zero router weights so all experts produce
            identical outputs. Set to False to keep random router weights.
        verbose: If True, print progress information.
    
    Raises:
        UpcyclingError: If models are incompatible or weights cannot be copied.
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> dense = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
        >>> moe = SmolMoEForCausalLM(config)
        >>> upcycle_dense_to_moe(dense, moe)
        >>> # moe now produces same outputs as dense
    """
    # Set models to eval mode
    dense_model.eval()
    moe_model.eval()
    
    with torch.no_grad():
        # Step 1: Copy embeddings
        if verbose:
            print("Copying embeddings...")
        copy_embeddings(dense_model, moe_model)
        
        # Step 2: Get layers
        dense_layers = get_dense_layers(dense_model)
        moe_layers = get_moe_layers(moe_model)
        
        if len(dense_layers) != len(moe_layers):
            raise UpcyclingError(
                f"Layer count mismatch: dense has {len(dense_layers)} layers, "
                f"MoE has {len(moe_layers)} layers."
            )
        
        # Step 3: Copy each layer
        if verbose:
            print(f"Copying {len(dense_layers)} layers...")
        
        for i, (dense_layer, moe_layer) in enumerate(zip(dense_layers, moe_layers)):
            copy_attention_weights(dense_layer, moe_layer)
            copy_layer_norms(dense_layer, moe_layer)
            copy_ffn_to_experts(dense_layer, moe_layer, zero_router=zero_router)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Copied layer {i + 1}/{len(dense_layers)}")
        
        # Step 4: Copy final norm
        if verbose:
            print("Copying final normalization...")
        copy_final_norm(dense_model, moe_model)
    
    if verbose:
        num_experts = moe_layers[0].moe.E
        print(f"✓ Upcycling complete! FFN replicated to {num_experts} experts.")
        if zero_router:
            print("  Router weights zeroed for deterministic routing.")


def verify_upcycling(
    dense_model: nn.Module,
    moe_model: nn.Module,
    input_ids: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """
    Verify that an upcycled MoE model produces identical outputs to the dense model.
    
    Args:
        dense_model: Source dense model.
        moe_model: Upcycled MoE model.
        input_ids: Sample input token IDs for testing.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        
    Returns:
        True if outputs match within tolerance, False otherwise.
    """
    dense_model.eval()
    moe_model.eval()
    
    with torch.no_grad():
        # Get dense model output
        dense_out = dense_model(input_ids)
        if hasattr(dense_out, "logits"):
            dense_logits = dense_out.logits
        else:
            dense_logits = dense_out["logits"]
        
        # Get MoE model output
        moe_out = moe_model(input_ids)
        if hasattr(moe_out, "logits"):
            moe_logits = moe_out.logits
        else:
            moe_logits = moe_out["logits"]
        
        # Compare
        match = torch.allclose(
            dense_logits.float(),
            moe_logits.float(),
            rtol=rtol,
            atol=atol,
        )
        
        if not match:
            max_diff = (dense_logits - moe_logits).abs().max().item()
            print(f"Max difference: {max_diff}")
        
        return match