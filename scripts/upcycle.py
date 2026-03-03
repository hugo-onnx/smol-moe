#!/usr/bin/env python3
"""
Convert a pre-trained dense SmolLM-135M model into a Mixture-of-Experts model.

Sparse upcycling copies the FFN weights from the dense model into every expert,
so the MoE output exactly matches the dense model — providing a warm start for
continued MoE training.

Usage:
    python scripts/upcycle.py
    python scripts/upcycle.py --num-experts 4 --output weights/upcycled_moe.pt
    python scripts/upcycle.py --dense-model HuggingFaceTB/SmolLM-135M
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Upcycle a dense SmolLM-135M checkpoint into a MoE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="HuggingFace model ID for the dense base model (default: HuggingFaceTB/SmolLM-135M)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=3,
        help="Number of experts in the MoE model (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/upcycled_moe.pt",
        help="Output path for the upcycled weights (default: weights/upcycled_moe.pt)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip output verification against the dense model",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM
        from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
        from smol_moe.upcycling import upcycle_dense_to_moe, verify_upcycling
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run 'uv sync' to install dependencies.")
        sys.exit(1)

    # Load dense base model in float32 to match the MoE model's dtype.
    # Without this, HuggingFace may respect the config's torch_dtype (e.g.
    # bfloat16 for SmolLM-135M), causing bfloat16 vs float32 rounding
    # differences that accumulate over 30 layers and break verification.
    print(f"Loading dense model: {args.dense_model}")
    dense = AutoModelForCausalLM.from_pretrained(args.dense_model, torch_dtype=torch.float32)

    # Build matching MoE config
    config = SmolMoEConfig(
        vocab_size=49152,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_heads=9,
        kv_heads=3,
        num_experts=args.num_experts,
        num_experts_per_tok=1,
    )
    moe = SmolMoEForCausalLM(config)
    print(f"MoE model: {args.num_experts} experts, {sum(p.numel() for p in moe.parameters()):,} params")

    # Upcycle
    upcycle_dense_to_moe(dense, moe)

    # Verify
    if not args.no_verify:
        print("\nVerifying upcycling (outputs must match the dense model)...")
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        ok = verify_upcycling(dense, moe, input_ids)
        if ok:
            print("Verification passed — MoE outputs match the dense model.")
        else:
            print("Verification FAILED. The upcycled weights may be incorrect.")
            sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(moe.state_dict(), output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved upcycled weights to: {output_path} ({size_mb:.1f} MB)")
    print("\nNext step: run scripts/pretrain.py to continue pretraining the MoE model.")


if __name__ == "__main__":
    main()
