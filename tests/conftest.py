"""Configuration for ground truth ParScale model."""

import pytest
import torch

from parscale_xattn import ParScaleConfig, ParScaleCrossAttnModel
from parscale_xattn.modeling_ground_truth import (
    ParScaleConfig as OrigQwen2ParScaleConfig,
    Qwen2ParScaleForCausalLM as OrigQwen2ParScaleForCausalLM,
    Qwen2Model as OrigQwen2Model,
)


@pytest.fixture(scope="module")
def small_config_dict():
    """A small config as a dictionary for fast testing."""
    return {
        "parscale_n": 2,
        "parscale_n_tokens": 4,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 256,
    }


@pytest.fixture(scope="module")
def small_config(small_config_dict):
    return ParScaleConfig(**small_config_dict)


@pytest.fixture(scope="module")
def small_config_no_replica_dict():
    """A small standard config (parscale_n=1) as a dictionary."""
    return {
        "parscale_n": 1,
        "parscale_n_tokens": 0,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 256,
    }


@pytest.fixture(scope="module")
def small_config_no_replica(small_config_no_replica_dict):
    return ParScaleConfig(**small_config_no_replica_dict)


@pytest.fixture(scope="module")
def create_ground_truth_config(**kwargs):
    """Create a ground truth configuration with the original implementation."""
    return OrigQwen2ParScaleConfig(**kwargs)


@pytest.fixture(scope="module")
def create_ground_truth_model(config):
    """Create a ground truth model with the original implementation."""
    return OrigQwen2ParScaleForCausalLM(config)


@pytest.fixture(scope="module")
def create_ground_truth_base_model(config):
    """Create a ground truth base model with the original implementation."""
    return OrigQwen2Model(config)


def create_model_pair(orig_config_dict, cross_attn_config_dict=None):
    """Create a pair of new and original models with synced weights."""
    # Set seeds for reproducibility
    if cross_attn_config_dict is None:
        cross_attn_config_dict = {}

    torch.manual_seed(42)
    orig_config = OrigQwen2ParScaleConfig(**orig_config_dict)
    orig_model = OrigQwen2ParScaleForCausalLM(orig_config)

    torch.manual_seed(42)
    new_config = ParScaleConfig(**orig_config_dict, **cross_attn_config_dict)
    new_model = ParScaleCrossAttnModel(new_config)

    # Sync model weights
    new_state = new_model.state_dict()
    orig_state = orig_model.state_dict()
    for name, param in new_state.items():
        if name in orig_state:
            param.data.copy_(orig_state[name].data)

    return orig_model, new_model
