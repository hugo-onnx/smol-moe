#!/usr/bin/env python3
"""
Download SmolMoE pretrained weights from HuggingFace Hub.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --output-dir ./my_weights
    python scripts/download_weights.py --token YOUR_HF_TOKEN
    
The script will download the trial_weights.pt file from the
dsouzadaniel/C4AI_SmolMoELM repository.

Authentication (in order of priority):
1. --token command line argument
2. HF_TOKEN in .env file (recommended)
3. HF_TOKEN environment variable
4. Cached credentials from `huggingface-cli login`

Setup:
1. Copy .env.example to .env
2. Add your HuggingFace token to .env
3. Run this script
"""

import argparse
import os
import sys
from pathlib import Path


def load_env():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        
        # Look for .env in project root
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return True
        
        # Also check current directory
        if Path(".env").exists():
            load_dotenv()
            return True
            
        return False
    except ImportError:
        print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
        return False


def get_hf_token(cli_token: str | None = None) -> str | None:
    """
    Get HuggingFace token from various sources.
    
    Priority:
    1. CLI argument
    2. .env file (via load_env)
    3. Environment variable
    4. None (will use cached credentials if available)
    """
    if cli_token:
        return cli_token
    
    # load_env() already loaded .env into os.environ
    return os.environ.get("HF_TOKEN")


def download_weights(
    output_dir: str = "./weights",
    repo_id: str = "dsouzadaniel/C4AI_SmolMoELM",
    filename: str = "trial_weights.pt",
    token: str | None = None,
) -> Path:
    """
    Download SmolMoE weights from HuggingFace Hub.
    
    Args:
        output_dir: Directory to save weights
        repo_id: HuggingFace repository ID
        filename: Name of the weights file
        token: HuggingFace API token (optional)
        
    Returns:
        Path to downloaded weights file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is required. Install with:")
        print("  pip install huggingface-hub")
        sys.exit(1)
    
    # Resolve output directory
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading weights from {repo_id}...")
    print(f"  File: {filename}")
    print(f"  Destination: {output_dir}")
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_dir),
            token=token,
        )
        
        path = Path(path)
        print(f"\n✓ Successfully downloaded to: {path}")
        print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return path
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        
        if "401" in str(e) or "authentication" in str(e).lower():
            print("\nAuthentication required. Options:")
            print("  1. Set HF_TOKEN environment variable")
            print("  2. Use --token argument")
            print("  3. Run 'huggingface-cli login'")
        
        sys.exit(1)


def verify_weights(path: Path) -> bool:
    """
    Verify downloaded weights by attempting to load them.
    
    Args:
        path: Path to weights file
        
    Returns:
        True if weights are valid
    """
    try:
        import torch
    except ImportError:
        print("Warning: torch not installed, skipping verification")
        return True
    
    print("\nVerifying weights...")
    
    try:
        state_dict = torch.load(path, map_location='cpu')
        
        print(f"  Number of parameters: {len(state_dict)}")
        print(f"  Sample keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            shape = tuple(state_dict[key].shape)
            print(f"    - {key}: {shape}")
        
        print("\n✓ Weights verified successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False


def main():
    # Load .env file first
    env_loaded = load_env()
    
    parser = argparse.ArgumentParser(
        description="Download SmolMoE weights from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./weights",
        help="Directory to save weights (default: ./weights)",
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        default="dsouzadaniel/C4AI_SmolMoELM",
        help="HuggingFace repository ID",
    )
    
    parser.add_argument(
        "--filename",
        type=str,
        default="trial_weights.pt",
        help="Weights filename to download",
    )
    
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="HuggingFace API token (or use .env file / HF_TOKEN env var)",
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of downloaded weights",
    )
    
    args = parser.parse_args()
    
    # Get token with priority: CLI > .env > env var
    token = get_hf_token(args.token)
    
    if env_loaded:
        print("✓ Loaded configuration from .env file")
    
    # Download
    path = download_weights(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        filename=args.filename,
        token=token,
    )
    
    # Verify
    if not args.no_verify:
        verify_weights(path)
    
    print(f"\nTo load these weights:")
    print(f"  import torch")
    print(f"  from smol_moe import SmolMoEConfig, SmolMoEForCausalLM")
    print(f"  ")
    print(f"  model = SmolMoEForCausalLM(SmolMoEConfig())")
    print(f"  model.load_state_dict(torch.load('{path}'))")


if __name__ == "__main__":
    main()