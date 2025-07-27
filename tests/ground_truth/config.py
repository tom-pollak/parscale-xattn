"""Configuration for ground truth ParScale model."""

import sys
from pathlib import Path

# Add the original source to path for importing
sys.path.append(str(Path(__file__).parent))

# Import original configuration
from orig_parscale import Qwen2ParScaleConfig as OrigQwen2ParScaleConfig
from orig_parscale import Qwen2ParScaleForCausalLM as OrigQwen2ParScaleForCausalLM
from orig_parscale import Qwen2Model as OrigQwen2Model


def create_ground_truth_config(**kwargs):
    """Create a ground truth configuration with the original implementation."""
    return OrigQwen2ParScaleConfig(**kwargs)


def create_ground_truth_model(config):
    """Create a ground truth model with the original implementation."""
    return OrigQwen2ParScaleForCausalLM(config)


def create_ground_truth_base_model(config):
    """Create a ground truth base model with the original implementation."""
    return OrigQwen2Model(config)
