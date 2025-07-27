import pytest
import sys
from pathlib import Path

# Add src to path. This allows pytest to find the src modules.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parscale_xattn import Qwen2ParScaleConfig


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
    """A small Qwen2ParScaleConfig object for fast testing."""
    return Qwen2ParScaleConfig(**small_config_dict)


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
    """A small Qwen2ParScaleConfig object for standard mode (no replicas)."""
    return Qwen2ParScaleConfig(**small_config_no_replica_dict)
