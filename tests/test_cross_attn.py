"""Tests for backward compatibility and API compatibility."""

import torch


from parscale_xattn import ParScaleConfig, ParScaleForCausalLM


class TestParScaleModeCompatibility:
    """Test compatibility in ParScale mode (parscale_n > 1)."""

    def test_cache_compatibility(self, small_config):
        """Test that caching works in ParScale mode."""
        model = ParScaleForCausalLM(small_config)

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

    def test_cross_attention_enabled_compatibility(self):
        """Test that cross-attention enabled mode is compatible."""
        config = ParScaleConfig(
            parscale_n=4,
            parscale_n_tokens=24,
            enable_cross_attn=True,
            parscale_cross_attn_layers=[0],
            hidden_size=64,
            num_hidden_layers=2,
            vocab_size=100,
        )
        model = ParScaleForCausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 3))
        output = model(input_ids)

        # Should work with cross-attention enabled
        assert output.logits.shape == (2, 3, config.vocab_size)

    def test_partial_cross_attention_layers(self):
        """Test compatibility with cross-attention on subset of layers."""
        config = ParScaleConfig(
            parscale_n=4,
            enable_cross_attn=True,
            parscale_cross_attn_layers=[0, 2],  # Only layers 0 and 2
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=100,
        )
        model = ParScaleForCausalLM(config)

        # Should initialize without error
        assert len(model.model.layers) == 4

        # Should work normally
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        output = model(input_ids)
        assert output.logits.shape == (1, 5, config.vocab_size)
