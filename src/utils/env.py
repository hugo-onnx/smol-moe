"""
Environment and configuration utilities.

This module provides utilities for loading environment variables
from .env files, which is useful for managing secrets like API tokens.

Usage:
    from smol_moe.utils.env import load_env, get_hf_token
    
    # Load .env file (call once at startup)
    load_env()
    
    # Get HuggingFace token
    token = get_hf_token()
"""

import os
from pathlib import Path
from typing import Optional


def find_project_root(marker_files: tuple = (".env", "pyproject.toml", ".git")) -> Optional[Path]:
    """
    Find the project root directory by looking for marker files.
    
    Args:
        marker_files: Files that indicate project root
        
    Returns:
        Path to project root, or None if not found
    """
    current = Path.cwd()
    
    for parent in [current] + list(current.parents):
        for marker in marker_files:
            if (parent / marker).exists():
                return parent
    
    return None


def load_env(env_path: Optional[str | Path] = None) -> bool:
    """
    Load environment variables from .env file.
    
    Searches for .env in:
    1. Provided path
    2. Current directory
    3. Project root (found by looking for pyproject.toml or .git)
    
    Args:
        env_path: Optional explicit path to .env file
        
    Returns:
        True if .env was found and loaded, False otherwise
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    
    # Check explicit path
    if env_path:
        path = Path(env_path)
        if path.exists():
            load_dotenv(path)
            return True
        return False
    
    # Check current directory
    if Path(".env").exists():
        load_dotenv()
        return True
    
    # Check project root
    root = find_project_root()
    if root:
        env_file = root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return True
    
    return False


def get_hf_token(token: Optional[str] = None) -> Optional[str]:
    """
    Get HuggingFace token from various sources.
    
    Priority order:
    1. Provided token argument
    2. HF_TOKEN environment variable
    3. HUGGING_FACE_HUB_TOKEN environment variable
    4. None (will use cached credentials from huggingface-cli login)
    
    Args:
        token: Optional explicit token
        
    Returns:
        HuggingFace token or None
    """
    if token:
        return token
    
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def require_hf_token(token: Optional[str] = None) -> str:
    """
    Get HuggingFace token, raising an error if not found.
    
    Args:
        token: Optional explicit token
        
    Returns:
        HuggingFace token
        
    Raises:
        ValueError: If no token is found
    """
    result = get_hf_token(token)
    
    if result is None:
        raise ValueError(
            "HuggingFace token not found. Please either:\n"
            "  1. Create a .env file with HF_TOKEN=your_token\n"
            "  2. Set the HF_TOKEN environment variable\n"
            "  3. Run 'huggingface-cli login'\n"
            "\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )
    
    return result


# Auto-load .env when module is imported
_env_loaded = load_env()