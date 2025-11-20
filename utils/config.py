"""
Configuration utilities for Trinity CodeGen Training.

This module provides common configuration loading and directory setup functions
used across training and benchmarking scripts.
"""

import os
import json
from typing import Dict


def load_model_config(model: str) -> Dict[str, int]:
    """
    Load model configuration from model_configs.json.

    Args:
        model: Model name (e.g., 'falcon', 'llama')

    Returns:
        dict: Model configuration containing M, N, D, H, P parameters

    Raises:
        ValueError: If model type is not found in configuration file
        FileNotFoundError: If model_configs.json is not found
    """
    config_path = "/home/chani227/Trinity/CodeGen/Training/model_configs.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        configs = json.load(f)

    if model not in configs:
        available_models = ', '.join(configs.keys())
        raise ValueError(f"Unknown model type: {model}. Available models: {available_models}")

    return configs[model]


def setup_directories(seq: int, base_dir: str = None) -> str:
    """
    Create necessary directory structure if it doesn't exist.

    Structure:
    base_dir/
      seq{seq}/
        bwd/
        bwd_json/
        eval/
        fwd/

    Args:
        seq: Sequence identifier
        base_dir: Base directory path (default: /home/chani227/Trinity/CodeGen/Training/data)

    Returns:
        str: Path to the created sequence directory
    """
    if base_dir is None:
        base_dir = "/home/chani227/Trinity/CodeGen/Training/data"

    seq_dir = os.path.join(base_dir, f"seq{seq}")

    # Create sequence directory if it doesn't exist
    os.makedirs(seq_dir, exist_ok=True)

    # Create subdirectories
    subdirs = ['bwd', 'bwd_json', 'eval', 'fwd']
    for subdir in subdirs:
        os.makedirs(os.path.join(seq_dir, subdir), exist_ok=True)

    print(f"✓ Directory structure verified/created: {seq_dir}")
    return seq_dir
