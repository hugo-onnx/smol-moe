#!/usr/bin/env python3
"""
Train each expert to specialize in a different domain (chat, code, math).

This stage adds domain-supervised routing loss, curriculum learning, and
optional knowledge distillation on top of continued pretraining.

Usage:
    python scripts/domain_train.py
    python scripts/domain_train.py --weights weights/pretrained_moe.pt --steps 500
    python scripts/domain_train.py --max-samples 1000 --lambda-route 0.5

Requires a HuggingFace token to access the Nemotron dataset (gated).
Any multi-domain SFT dataset with input/output columns works as a drop-in.
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Domain-specialized expert training for SmolMoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/pretrained_moe.pt",
        help="Path to pretrained MoE weights (default: weights/pretrained_moe.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nvidia/Llama-Nemotron-Post-Training-Dataset",
        help="HuggingFace dataset ID (default: nvidia/Llama-Nemotron-Post-Training-Dataset)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="SFT",
        help="Dataset subset/config (default: SFT)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["chat", "code", "math"],
        help="Domain splits to use — one per expert (default: chat code math)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of training steps (default: 500)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples per domain (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--lambda-route",
        type=float,
        default=0.5,
        help="Weight for domain routing supervision loss (default: 0.5)",
    )
    parser.add_argument(
        "--expert-dropout",
        type=float,
        default=0.2,
        help="Fraction of steps that force domain-based routing (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/domain_expert_moe.pt",
        help="Output path for domain-specialized weights (default: weights/domain_expert_moe.pt)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/domain",
        help="Directory for checkpoints and plots (default: outputs/domain)",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
        from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
        from smol_moe.domain_expert import (
            DomainExpertTrainer,
            DomainTrainingConfig,
            DomainExpertMetrics,
            build_domain_dataloaders,
        )
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run 'uv sync' to install dependencies.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    # Load model
    config = SmolMoEConfig()
    moe = SmolMoEForCausalLM(config)
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights file not found: {weights_path}")
        print("Run scripts/pretrain.py first to produce the pretrained weights.")
        sys.exit(1)
    moe.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"Loaded weights from {weights_path}")

    # Per-domain dataloaders
    print(f"\nLoading dataset: {args.dataset} [{args.subset}]")
    print(f"Domains: {args.domains}")
    train_loader, val_loader, domain_loaders = build_domain_dataloaders(
        dataset_id=args.dataset,
        subset=args.subset,
        splits=args.domains,
        tokenizer=tokenizer,
        block_size=256,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Training config
    domain_config = DomainTrainingConfig(
        steps=args.steps,
        learning_rate=args.lr,
        router_lr_multiplier=3.0,
        lambda_route=args.lambda_route,
        lambda_kd=0.0,
        expert_dropout_prob=args.expert_dropout,
        router_noise_start=0.5,
        prefix_length=1,
        report_every=max(1, args.steps // 10),
        save_dir=args.save_dir,
    )

    trainer = DomainExpertTrainer(
        moe,
        train_loader,
        val_loader,
        domain_config,
        domain_loaders=domain_loaders,
        domains=args.domains,
    )
    trainer.train()

    # Expert-domain alignment score
    device = next(moe.parameters()).device
    domain_to_expert = {i: i for i in range(len(args.domains))}
    print("\nPost-training domain alignment:")
    edas = DomainExpertMetrics.expert_domain_alignment(
        moe, val_loader, domain_to_expert, device, num_batches=10
    )
    print(f"  EDAS: {edas:.1f}%  (higher = better domain specialization)")

    trainer.plot_confusion_matrix(num_batches=20)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(moe.state_dict(), output_path)
    print(f"\nSaved domain-specialized weights to: {output_path}")
    print("\nNext step: run scripts/generate.py to generate text with the final model.")


if __name__ == "__main__":
    main()
