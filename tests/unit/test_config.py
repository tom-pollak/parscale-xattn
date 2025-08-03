"""Unit tests for ParScale configuration classes."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from parscale_xattn import Qwen2ParScaleConfig
from parscale_xattn.config_base import ParScaleBaseConfig
from parscale_xattn.config_cross_attn import ParScaleCrossAttnConfig

# Import ground truth for comparison
sys.path.append(str(Path(__file__).parent.parent / "ground_truth"))
from config import create_ground_truth_config


class TestParScaleBaseConfig:
    """Test the base ParScale configuration."""

    def test_default_config(self):
        """Test default configuration values match expected ParScale defaults."""
        config = ParScaleBaseConfig()

        # Test base Transformer defaults
        assert config.vocab_size == 151936
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 32

        # Test ParScale defaults
        assert config.parscale_n == 1
        assert config.parscale_n_tokens == 0
        assert config.parscale_attn_smooth == 0.01
        assert config.model_type == "parscale_base"

    def test_standard_qwen2_mode(self):
        """Test that parscale_n=1 enforces standard Qwen2 behavior."""
        config = ParScaleBaseConfig(parscale_n=1, parscale_n_tokens=0)
        assert config.parscale_n == 1
        assert config.parscale_n_tokens == 0

        config = ParScaleBaseConfig(parscale_n=1, parscale_n_tokens=48)
        assert config.parscale_n == 1
        assert config.parscale_n_tokens == 48

    def test_parscale_mode(self):
        """Test ParScale mode with multiple replicas."""
        config = ParScaleBaseConfig(parscale_n=4, parscale_n_tokens=48)
        assert config.parscale_n == 4
        assert config.parscale_n_tokens == 48

    def test_validation_errors(self):
        """Test configuration validation catches invalid parameters."""
        # Invalid parscale_n
        with pytest.raises(ValueError, match="parscale_n must be >= 1"):
            ParScaleBaseConfig(parscale_n=0)

        # Invalid parscale_n_tokens
        with pytest.raises(ValueError, match="parscale_n_tokens must be >= 0"):
            ParScaleBaseConfig(parscale_n_tokens=-1)


class TestParScaleCrossAttnConfig:
    """Test the cross-attention configuration extension."""

    def test_default_cross_attn_config(self):
        """Test default cross-attention configuration."""
        config = ParScaleCrossAttnConfig()

        # Cross-attention defaults
        assert config.enable_cross_attn == False
        assert config.parscale_cross_attn_layers is None
        assert config.enable_replica_rope == False
        assert config.model_type == "parscale_cross_attn"

        # Should inherit base ParScale defaults
        assert config.parscale_n == 1
        assert config.parscale_n_tokens == 0

    def test_cross_attn_enabled(self):
        """Test cross-attention enabled configuration."""
        config = ParScaleCrossAttnConfig(
            parscale_n=4,
            enable_cross_attn=True,
            parscale_cross_attn_layers=[0, 4, 8, 12],
        )

        assert config.parscale_n == 4
        assert config.enable_cross_attn == True
        assert config.parscale_cross_attn_layers == [0, 4, 8, 12]

    def test_cross_attn_validation(self):
        """Test cross-attention specific validation."""
        # Cross-attention requires parscale_n > 1
        with pytest.raises(
            ValueError, match="Cross-attention.*requires parscale_n > 1"
        ):
            ParScaleCrossAttnConfig(parscale_n=1, enable_cross_attn=True)

        # Cross-attention layers specified without enabling cross-attention
        with pytest.raises(
            ValueError,
            match="parscale_cross_attn_layers is specified but enable_cross_attn=False",
        ):
            ParScaleCrossAttnConfig(parscale_n=2, parscale_cross_attn_layers=[0, 1])

        # Invalid layer indices
        with pytest.raises(ValueError, match="Layer index.*is >= num_hidden_layers"):
            ParScaleCrossAttnConfig(
                parscale_n=2,
                num_hidden_layers=4,
                enable_cross_attn=True,
                parscale_cross_attn_layers=[0, 1, 2, 5],  # 5 is invalid
            )

    def test_replica_rope_validation(self):
        """Test replica RoPE validation."""
        # Replica RoPE requires cross-attention
        with pytest.raises(ValueError, match="Replica RoPE.*requires cross-attention"):
            ParScaleCrossAttnConfig(parscale_n=2, enable_replica_rope=True)

        # Replica RoPE requires parscale_n > 1
        with pytest.raises(ValueError, match="Replica RoPE.*requires parscale_n > 1"):
            ParScaleCrossAttnConfig(
                parscale_n=1, enable_cross_attn=True, enable_replica_rope=True
            )


class TestQwen2ParScaleConfig:
    """Test the main Qwen2ParScaleConfig class for backward compatibility."""

    def test_backward_compatibility(self):
        """Test that Qwen2ParScaleConfig maintains backward compatibility."""
        # Should work with original parameter names
        config = Qwen2ParScaleConfig(
            parscale_n=4,
            parscale_n_tokens=48,
            enable_cross_attn=True,
            parscale_cross_attn_layers=[0, 4, 8],
        )

        assert config.parscale_n == 4
        assert config.enable_cross_attn == True
        assert config.parscale_cross_attn_layers == [0, 4, 8]
        assert config.model_type == "qwen2_parscale"

    def test_inheritance_chain(self):
        """Test that the inheritance chain works correctly."""
        config = Qwen2ParScaleConfig(parscale_n=4, enable_cross_attn=True)

        # Should be instance of all parent classes
        assert isinstance(config, ParScaleCrossAttnConfig)
        assert isinstance(config, ParScaleBaseConfig)
        assert hasattr(config, "parscale_n")  # From base
        assert hasattr(config, "enable_cross_attn")  # From cross-attn


class TestConfigCompatibility:
    """Test compatibility with ground truth configuration."""

    def test_parameter_equivalence(self):
        """Test that our configs have the same parameters as ground truth."""
        # Test with standard Qwen2 mode
        new_config = Qwen2ParScaleConfig(parscale_n=1, parscale_n_tokens=0)
        orig_config = create_ground_truth_config(parscale_n=1, parscale_n_tokens=0)

        # Test key parameters match
        assert new_config.parscale_n == orig_config.parscale_n
        assert new_config.parscale_n_tokens == orig_config.parscale_n_tokens
        assert new_config.vocab_size == orig_config.vocab_size
        assert new_config.hidden_size == orig_config.hidden_size
        assert new_config.num_hidden_layers == orig_config.num_hidden_layers

    def test_parscale_mode_equivalence(self):
        """Test ParScale mode parameter equivalence."""
        new_config = Qwen2ParScaleConfig(parscale_n=4, parscale_n_tokens=48)
        orig_config = create_ground_truth_config(parscale_n=4, parscale_n_tokens=48)

        assert new_config.parscale_n == orig_config.parscale_n
        assert new_config.parscale_n_tokens == orig_config.parscale_n_tokens
        assert new_config.parscale_attn_smooth == orig_config.parscale_attn_smooth


if __name__ == "__main__":
    pytest.main([__file__])
