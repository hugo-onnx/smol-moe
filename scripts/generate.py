#!/usr/bin/env python3
"""
Generate text with a trained SmolMoE model.

Usage:
    python scripts/generate.py --prompt "Explain what a Mixture of Experts model is:"
    python scripts/generate.py --weights weights/domain_expert_moe.pt --prompt "def fibonacci"
    python scripts/generate.py --prompt "Solve: 2x + 3 = 7" --temperature 0.8 --top-k 50
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a trained SmolMoE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/domain_expert_moe.pt",
        help="Path to model weights (default: weights/domain_expert_moe.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain what a Mixture of Experts model is:",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 = greedy decoding (default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (used when temperature > 0, default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (used when temperature > 0, default: 0.95)",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
        from smol_moe import SmolMoEConfig, SmolMoEForCausalLM
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run 'uv sync' to install dependencies.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    # Load model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights file not found: {weights_path}")
        print("Available weight files:")
        for p in Path("weights").glob("*.pt"):
            print(f"  {p}")
        sys.exit(1)

    model = SmolMoEForCausalLM(SmolMoEConfig())
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    print(f"Loaded weights from {weights_path}\n")

    # Encode prompt
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]

    # Generate
    print(f"Prompt: {args.prompt}\n")
    with torch.no_grad():
        if args.temperature == 0:
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=0,
            )
        else:
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(output)


if __name__ == "__main__":
    main()
