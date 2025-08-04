"""Integration tests for numerical equivalence between refactored and ground truth implementations."""

import pytest
import torch

from tests.conftest import create_model_pair


class TestNumericalEquivalence:
    """Test numerical equivalence between new and original models."""

    batch_size = 2
    seq_len = 6

    def test_forward_pass_equivalence(self, small_config_dict):
        """Test that forward passes produce identical outputs."""
        orig_model, new_model = create_model_pair(small_config_dict)
        input_ids = torch.randint(
            0, orig_model.config.vocab_size, (self.batch_size, self.seq_len)
        )

        with torch.no_grad():
            new_output = new_model(input_ids)
            orig_output = orig_model(input_ids)

        # Check logits are identical
        assert torch.allclose(
            new_output.logits, orig_output.logits, atol=1e-5, rtol=1e-4
        ), (
            f"Logits differ: max diff = {(new_output.logits - orig_output.logits).abs().max()}"
        )


class TestCacheEquivalence:
    """Test equivalence during text generation."""

    def test_cache_behavior_equivalence(self, small_config_dict):
        """Test that generation is deterministic and matches between implementations."""

        orig_model, new_model = create_model_pair(
            {**small_config_dict, **{"use_cache": True}}
        )

        input_ids = torch.randint(0, orig_model.config.vocab_size, (1, 5))

        with torch.no_grad():
            new_output = new_model(input_ids)
            orig_output = orig_model(input_ids)

        # Both should have past_key_values
        assert new_output.past_key_values is not None
        assert orig_output.past_key_values is not None

        # Cache should have same structure
        assert len(new_output.past_key_values.key_cache) == len(
            orig_output.past_key_values.key_cache
        )
        assert len(new_output.past_key_values.value_cache) == len(
            orig_output.past_key_values.value_cache
        )

        with torch.no_grad():
            torch.manual_seed(123)  # Set generation seed
            new_generated = new_model.generate(
                input_ids, max_new_tokens=5, do_sample=False, pad_token_id=0
            )

            torch.manual_seed(123)  # Same generation seed
            orig_generated = orig_model.generate(
                input_ids, max_new_tokens=5, do_sample=False, pad_token_id=0
            )

        # Generated sequences should be identical
        assert torch.equal(new_generated, orig_generated), (  # type: ignore
            f"Generated sequences differ:\nNew: {new_generated}\nOrig: {orig_generated}"
        )


class TestParScaleToStandardEquivalence:
    """Test that ParScale with no prefix tokens behaves like standard Qwen2."""

    def test_no_prefix_models_have_no_parscale_cache(self, small_config_dict):
        """Test that models with parscale_n_tokens=0 don't create ParscaleCache."""

        _, new_model = create_model_pair(
            {**small_config_dict, **{"parscale_n_tokens": 0}}
        )

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, new_model.config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = new_model(input_ids, use_cache=True)

        # Should use DynamicCache, not ParscaleCache
        from transformers.cache_utils import DynamicCache
        from parscale_xattn import ParscaleCache

        assert isinstance(output.past_key_values, DynamicCache), (
            "With parscale_n_tokens=0, should use DynamicCache"
        )
        assert not isinstance(output.past_key_values, ParscaleCache), (
            "With parscale_n_tokens=0, should NOT use ParscaleCache"
        )


if __name__ == "__main__":
    pytest.main([__file__])
