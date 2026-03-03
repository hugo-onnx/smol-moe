"""
SmolMoE: Small Mixture of Experts Language Model

A clean, modular implementation of a Mixture of Experts language model
featuring:
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA)
- Sparse MoE with SwiGLU activation
- Load balancing for expert utilization

Example usage:
    from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
    
    config = SmolMoEConfig()
    model = SmolMoEForCausalLM(config)
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs['logits']
    
    # Get expert utilization
    utilization, lb_loss = model.get_expert_utilization()
"""

__version__ = "0.1.0"
__author__ = "Hugo Gonzalez"

from .config import SmolMoEConfig, config

from .models import (
    # Components
    RotaryEmbedding,
    RMSNorm,
    RoPEAttention,
    MixtureOfExperts,
    # Models
    SmolMoEModel,
    SmolMoEDecoderLayer,
    SmolMoEForCausalLM,
    smolMoELM,
)

from .utils import (
    timed,
    generation_compare,
    detach_metrics,
    plot_metrics,
    MetricsTracker,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    model_summary,
    load_env,
    get_hf_token,
    require_hf_token,
)

from .upcycling import (
    upcycle_dense_to_moe,
    verify_upcycling,
    UpcyclingError,
    get_dense_layers,
    get_moe_layers,
    copy_embeddings,
    copy_attention_weights,
    copy_layer_norms,
    copy_ffn_to_experts,
    copy_final_norm,
)

from .training import (
    TrainingConfig,
    Trainer,
    MoEMetrics,
    causal_lm_loss,
    build_dataloaders,
    get_cosine_schedule_with_warmup,
)

from .domain_expert import (
    DomainTrainingConfig,
    DomainExpertTrainer,
    DomainExpertMetrics,
    DomainPrefixEmbedding,
    CurriculumBatcher,
    build_domain_dataloaders,
    router_supervision_loss,
    domain_conditional_kd_loss,
    apply_expert_dropout,
    apply_router_noise,
    DOMAIN_TO_ID,
    DOMAIN_TO_EXPERT,
)

__all__ = [
    # Config
    "SmolMoEConfig",
    "config",
    # Components
    "RotaryEmbedding",
    "RMSNorm",
    "RoPEAttention",
    "MixtureOfExperts",
    # Models
    "SmolMoEModel",
    "SmolMoEDecoderLayer",
    "SmolMoEForCausalLM",
    "smolMoELM",
    # Upcycling
    "upcycle_dense_to_moe",
    "verify_upcycling",
    "UpcyclingError",
    "get_dense_layers",
    "get_moe_layers",
    "copy_embeddings",
    "copy_attention_weights",
    "copy_layer_norms",
    "copy_ffn_to_experts",
    "copy_final_norm",
    # Training
    "TrainingConfig",
    "Trainer",
    "MoEMetrics",
    "causal_lm_loss",
    "build_dataloaders",
    "get_cosine_schedule_with_warmup",
    # Domain Expert Training
    "DomainTrainingConfig",
    "DomainExpertTrainer",
    "DomainExpertMetrics",
    "DomainPrefixEmbedding",
    "CurriculumBatcher",
    "build_domain_dataloaders",
    "router_supervision_loss",
    "domain_conditional_kd_loss",
    "apply_expert_dropout",
    "apply_router_noise",
    "DOMAIN_TO_ID",
    "DOMAIN_TO_EXPERT",
    # Utils
    "timed",
    "generation_compare",
    "detach_metrics",
    "plot_metrics",
    "MetricsTracker",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "model_summary",
    "load_env",
    "get_hf_token",
    "require_hf_token",
]