"""Tests for backward compatibility and API compatibility."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from parscale_xattn import Qwen2ParScaleConfig, Qwen2ParScaleForCausalLM


class TestStandardQwen2Compatibility:
    """Test compatibility when used as standard Qwen2 (parscale_n=1)."""

    def test_standard_mode_no_parscale_overhead(self, small_config_no_replica):
        """Test that standard mode has no ParScale overhead."""
        model = Qwen2ParScaleForCausalLM(small_config_no_replica)

        # Should not have ParScale-specific parameters when parscale_n=1
        for name, param in model.named_parameters():
            assert "prefix_k" not in name
            assert "prefix_v" not in name
            assert "aggregate_layer" not in name
            assert "cross_replica" not in name

    def test_standard_mode_forward_pass(self, small_config_no_replica):
        """Test that standard mode forward pass works identically to Qwen2."""
        model = Qwen2ParScaleForCausalLM(small_config_no_replica)

        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(
            0, small_config_no_replica.vocab_size, (batch_size, seq_len)
        )

        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (
            batch_size,
            seq_len,
            small_config_no_replica.vocab_size,
        )

    def test_generation_compatibility(self, small_config_no_replica):
        """Test that generation works in standard mode."""
        model = Qwen2ParScaleForCausalLM(small_config_no_replica)

        input_ids = torch.randint(0, small_config_no_replica.vocab_size, (1, 3))

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=2, do_sample=False, pad_token_id=0
            )

        # Should generate additional tokens
        assert output.shape[1] > input_ids.shape[1]


class TestParScaleModeCompatibility:
    """Test compatibility in ParScale mode (parscale_n > 1)."""

    def test_parscale_mode_initialization(self, small_config):
        """Test that ParScale mode initializes correctly."""
        model = Qwen2ParScaleForCausalLM(small_config)

        # Should have ParScale-specific parameters
        parscale_params = [
            name
            for name, _ in model.named_parameters()
            if "prefix_k" in name or "prefix_v" in name or "aggregate_layer" in name
        ]
        assert len(parscale_params) > 0

    def test_parscale_forward_batch_handling(self, small_config):
        """Test that ParScale mode handles batches correctly."""
        model = Qwen2ParScaleForCausalLM(small_config)

        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            seq_len = 3
            input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                output = model(input_ids)

            assert output.logits.shape[0] == batch_size
            assert output.logits.shape[1] == seq_len
            assert output.logits.shape[2] == small_config.vocab_size

    def test_cache_compatibility(self, small_config):
        """Test that caching works in ParScale mode."""
        model = Qwen2ParScaleForCausalLM(small_config)

        input_ids = torch.randint(0, small_config.vocab_size, (1, 3))

        # First forward pass with cache
        output1 = model(input_ids, use_cache=True)
        assert output1.past_key_values is not None

        # Second forward pass with cached states
        new_tokens = torch.randint(0, small_config.vocab_size, (1, 1))
        with torch.no_grad():
            output2 = model(
                new_tokens,
                past_key_values=output1.past_key_values,
                use_cache=True,
            )

        assert output2.logits.shape == (1, 1, small_config.vocab_size)


class TestCrossAttentionCompatibility:
    """Test compatibility with cross-attention features."""

    def test_cross_attention_disabled_compatibility(self):
        """Test that cross-attention disabled mode is compatible."""
        config = Qwen2ParScaleConfig(
            parscale_n=4,
            parscale_n_tokens=24,
            enable_cross_attn=False,  # Disabled
            hidden_size=64,
            num_hidden_layers=1,
            vocab_size=100,
        )
        with torch.no_grad():
            model = Qwen2ParScaleForCausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 3))
        output = model(input_ids)

        # Should work normally
        assert output.logits.shape == (2, 3, config.vocab_size)

    def test_cross_attention_enabled_compatibility(self):
        """Test that cross-attention enabled mode is compatible."""
        config = Qwen2ParScaleConfig(
            parscale_n=4,
            parscale_n_tokens=24,
            enable_cross_attn=True,
            parscale_cross_attn_layers=[0],
            hidden_size=64,
            num_hidden_layers=2,
            vocab_size=100,
        )
        model = Qwen2ParScaleForCausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 3))
        output = model(input_ids)

        # Should work with cross-attention enabled
        assert output.logits.shape == (2, 3, config.vocab_size)

    def test_partial_cross_attention_layers(self):
        """Test compatibility with cross-attention on subset of layers."""
        config = Qwen2ParScaleConfig(
            parscale_n=4,
            enable_cross_attn=True,
            parscale_cross_attn_layers=[0, 2],  # Only layers 0 and 2
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=100,
        )
        model = Qwen2ParScaleForCausalLM(config)

        # Should initialize without error
        assert len(model.model.layers) == 4

        # Should work normally
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        output = model(input_ids)
        assert output.logits.shape == (1, 5, config.vocab_size)


class TestConfigurationBackwardCompatibility:
    """Test that configuration maintains backward compatibility."""

    def test_old_parameter_names_still_work(self):
        """Test that old parameter names from research still work."""
        # These parameter names should all work
        config = Qwen2ParScaleConfig(
            parscale_n=4,
            parscale_n_tokens=48,
            parscale_attn_smooth=0.02,
            enable_cross_attn=True,
            parscale_cross_attn_layers=[0, 1],
            enable_replica_rope=True,
        )

        # All parameters should be accessible
        assert config.parscale_n == 4
        assert config.parscale_n_tokens == 48
        assert config.parscale_attn_smooth == 0.02
        assert config.enable_cross_attn == True
        assert config.parscale_cross_attn_layers == [0, 1]
        assert config.enable_replica_rope == True

    def test_config_serialization_compatibility(self):
        """Test that config can be serialized and deserialized."""
        config = Qwen2ParScaleConfig(
            parscale_n=4, parscale_n_tokens=48, enable_cross_attn=True, hidden_size=64
        )

        # Should be able to convert to dict and back
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["parscale_n"] == 4

        # Should be able to create from dict
        new_config = Qwen2ParScaleConfig.from_dict(config_dict)
        assert new_config.parscale_n == 4
        assert new_config.parscale_n_tokens == 48
        assert new_config.enable_cross_attn == True


if __name__ == "__main__":
    pytest.main([__file__])
