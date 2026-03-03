#!/usr/bin/env python3
"""
Continued pretraining of an upcycled SmolMoE model on a text corpus.

The router learns to differentiate between experts while the load-balancing
loss prevents expert collapse.

Usage:
    python scripts/pretrain.py
    python scripts/pretrain.py --weights weights/upcycled_moe.pt --steps 500
    python scripts/pretrain.py --dataset HuggingFaceTB/cosmopedia-100k --steps 100 --max-samples 500

GPU is recommended. On CPU, use --steps 50 --max-samples 500 for a quick smoke test.
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Continued pretraining for SmolMoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/upcycled_moe.pt",
        help="Path to upcycled MoE weights (default: weights/upcycled_moe.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceTB/cosmopedia-100k",
        help="HuggingFace dataset ID (default: HuggingFaceTB/cosmopedia-100k)",
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
        default=5000,
        help="Maximum dataset samples to use (default: 5000)",
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
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--lb-coef",
        type=float,
        default=0.01,
        help="Load-balancing loss coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/pretrained_moe.pt",
        help="Output path for pretrained weights (default: weights/pretrained_moe.pt)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/pretraining",
        help="Directory for checkpoints and metric plots (default: outputs/pretraining)",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
        from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
        from smol_moe.training import TrainingConfig, Trainer, build_dataloaders, MoEMetrics
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
        print("Run scripts/upcycle.py first to produce the upcycled weights.")
        sys.exit(1)
    moe.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"Loaded weights from {weights_path}")

    # Dataloaders
    print(f"\nLoading dataset: {args.dataset} (max {args.max_samples} samples)...")
    train_loader, val_loader = build_dataloaders(
        dataset_id=args.dataset,
        tokenizer=tokenizer,
        block_size=256,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        val_fraction=0.1,
    )

    # Training config
    train_config = TrainingConfig(
        steps=args.steps,
        learning_rate=args.lr,
        router_lr_multiplier=2.0,
        lb_loss_coef=args.lb_coef,
        warmup_steps=max(1, args.steps // 10),
        report_every=max(1, args.steps // 10),
        save_dir=args.save_dir,
    )

    # Train
    trainer = Trainer(moe, train_loader, val_loader, train_config)
    trainer.train()

    # Post-training MoE metrics
    device = next(moe.parameters()).device
    print("\nPost-training MoE metrics:")
    entropy = MoEMetrics.utilization_entropy(moe, val_loader, device)
    stats = MoEMetrics.expert_usage_stats(moe)
    print(f"  Utilization entropy : {entropy:.1f}%  (100% = perfectly balanced)")
    print(f"  Most used expert    : {stats['most_used']}")
    print(f"  Imbalance ratio     : {stats['imbalance_ratio']:.2f}x")

    # Plot and save
    trainer.plot_metrics(save_path=f"{args.save_dir}/metrics.png")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(moe.state_dict(), output_path)
    print(f"\nSaved pretrained weights to: {output_path}")
    print("\nNext step: run scripts/domain_train.py to specialize each expert by domain.")


if __name__ == "__main__":
    main()
