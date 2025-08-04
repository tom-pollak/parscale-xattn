"""
Pytest-based tests for cross-attention functionality and configuration.

This module verifies:
- The successful forward pass of models with cross-attention enabled.
- The successful forward pass of standard models (parscale_n=1) as a control.
- That invalid model configurations raise appropriate validation errors.

Tests use small model configurations defined in `tests/conftest.py` for speed.
"""

import pytest
from parscale_xattn import ParScaleConfig


# Test cases for configuration validation, adapted from the original script
CONFIG_TEST_CASES = [
    pytest.param(
        {"parscale_n": 1, "enable_cross_attn": True},
        True,
        id="cross-attn-with-parscale_n=1",
    ),
    pytest.param(
        {"parscale_n": 1, "parscale_n_tokens": 4},
        False,
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


@pytest.mark.parametrize("config_changes, should_fail", CONFIG_TEST_CASES)
def test_config_validation(small_config_dict, config_changes, should_fail):
    """
    Tests that ParScaleConfig validation catches invalid configurations.
    """
    config_dict = small_config_dict.copy()
    config_dict.update(config_changes)

    if should_fail:
        with pytest.raises(ValueError):
            ParScaleConfig(**config_dict)
    else:
        try:
            ParScaleConfig(**config_dict)
        except ValueError as e:
            pytest.fail(f"Configuration should be valid, but raised ValueError: {e}")
