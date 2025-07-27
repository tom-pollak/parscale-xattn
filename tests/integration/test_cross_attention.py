"""
Pytest-based tests for cross-attention functionality and configuration.

This module verifies:
- The successful forward pass of models with cross-attention enabled.
- The successful forward pass of standard models (parscale_n=1) as a control.
- That invalid model configurations raise appropriate validation errors.

Tests use small model configurations defined in `tests/conftest.py` for speed.
"""
import pytest
import torch

from src.parscale_xattn import Qwen2ParScaleConfig, Qwen2ParScaleForCausalLM

# Note: conftest.py provides the following fixtures used in this file:
# - small_config_dict: A dictionary for a small model config (parscale_n > 1).
# - small_config_no_replica: A Qwen2ParScaleConfig for a standard model (parscale_n=1).


def test_cross_attention_forward_pass(small_config_dict):
    """
    Tests that a model with cross-attention enabled can perform a forward pass
    without shape-related errors.
    """
    # Enable cross-attention on the single layer of the small config
    config_dict = small_config_dict.copy()
    config_dict["enable_cross_attn"] = True
    # There is only one hidden layer in the small config
    config_dict["parscale_cross_attn_layers"] = [0]
    config = Qwen2ParScaleConfig(**config_dict)

    model = Qwen2ParScaleForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 8
    parscale_n = config.parscale_n

    # Input shape for parscale is (parscale_n * batch_size, seq_len)
    input_ids = torch.randint(0, config.vocab_size, (parscale_n * batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits is not None
    assert outputs.logits.shape == (
        parscale_n * batch_size,
        seq_len,
        config.vocab_size,
    )


def test_standard_attention_forward_pass(small_config_no_replica):
    """
    Tests that a standard model (parscale_n=1) can perform a forward pass.
    This acts as a control test.
    """
    config = small_config_no_replica
    model = Qwen2ParScaleForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 8

    # Standard input shape is (batch_size, seq_len)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits is not None
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)


# Test cases for configuration validation, adapted from the original script
INVALID_CONFIG_TEST_CASES = [
    pytest.param(
        {"parscale_n": 1, "enable_cross_attn": True},
        True,
        id="cross-attn-with-parscale_n=1",
    ),
    pytest.param(
        {"parscale_n": 1, "parscale_n_tokens": 4},
        True,
        id="prefix-tokens-with-parscale_n=1",
    ),
    pytest.param(
        {
            "parscale_n": 2,
            "enable_cross_attn": False,
            "parscale_cross_attn_layers": [0],
        },
        True,
        id="cross-attn-layers-without-enable-flag",
    ),
    pytest.param(
        {
            "parscale_n": 1,
            "enable_cross_attn": False,
            "parscale_n_tokens": 0,
        },
        False,
        id="valid-parscale_n=1-config",
    ),
]


@pytest.mark.parametrize("config_changes, should_fail", INVALID_CONFIG_TEST_CASES)
def test_config_validation(small_config_dict, config_changes, should_fail):
    """
    Tests that Qwen2ParScaleConfig validation catches invalid configurations.
    """
    config_dict = small_config_dict.copy()
    config_dict.update(config_changes)

    if should_fail:
        with pytest.raises(ValueError):
            Qwen2ParScaleConfig(**config_dict)
    else:
        try:
            Qwen2ParScaleConfig(**config_dict)
        except ValueError as e:
            pytest.fail(f"Configuration should be valid, but raised ValueError: {e}")
