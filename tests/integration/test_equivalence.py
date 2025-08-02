"""Integration tests for numerical equivalence between refactored and ground truth implementations."""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from parscale_xattn import Qwen2ParScaleConfig, Qwen2ParScaleForCausalLM
from parscale_xattn.modeling_base import ParScaleBaseForCausalLM, ParScaleBaseModel

# Import ground truth for comparison
sys.path.append(str(Path(__file__).parent.parent / "ground_truth"))
from config import (
    create_ground_truth_config,
    create_ground_truth_model,
    create_ground_truth_base_model,
)


@pytest.fixture(scope="class")
def model_pair(small_config_dict):
    """Create a pair of new and original models with synced weights."""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    new_config = Qwen2ParScaleConfig(**small_config_dict)
    new_model = Qwen2ParScaleForCausalLM(new_config)

    torch.manual_seed(42)
    orig_config = create_ground_truth_config(**small_config_dict)
    orig_model = create_ground_truth_model(orig_config)

    # Sync model weights
    new_state = new_model.state_dict()
    orig_state = orig_model.state_dict()
    for name, param in new_state.items():
        if name in orig_state:
            param.data.copy_(orig_state[name].data)

    return new_model, orig_model, new_config


@pytest.mark.usefixtures("model_pair")
class TestNumericalEquivalence:
    """Test numerical equivalence between new and original models."""

    batch_size = 2
    seq_len = 6

    def test_forward_pass_equivalence(self, model_pair):
        """Test that forward passes produce identical outputs."""
        new_model, orig_model, config = model_pair
        input_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.seq_len))

        with torch.no_grad():
            new_output = new_model(input_ids)
            orig_output = orig_model(input_ids)

        # Check logits are identical
        assert torch.allclose(
            new_output.logits, orig_output.logits, atol=1e-5, rtol=1e-4
        ), (
            f"Logits differ: max diff = {(new_output.logits - orig_output.logits).abs().max()}"
        )

    def test_parameter_count_equivalence(self, model_pair):
        """Test that both models have the same number of parameters."""
        new_model, orig_model, _ = model_pair
        new_params = sum(p.numel() for p in new_model.parameters())
        orig_params = sum(p.numel() for p in orig_model.parameters())

        assert new_params == orig_params, (
            f"Parameter count differs: new={new_params}, orig={orig_params}"
        )

    def test_gradient_equivalence(self, model_pair):
        """Test that gradients are identical during backpropagation."""
        new_model, orig_model, config = model_pair
        input_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.seq_len))
        labels = input_ids.clone()

        # Forward and backward pass for new model
        new_output = new_model(input_ids, labels=labels)
        new_loss = new_output.loss
        new_loss.backward()

        # Store gradients
        new_grads = {
            name: param.grad.clone()
            for name, param in new_model.named_parameters()
            if param.grad is not None
        }

        # Reset and do the same for original model
        orig_model.zero_grad()
        orig_output = orig_model(input_ids, labels=labels)
        orig_loss = orig_output.loss
        orig_loss.backward()

        # Compare losses
        assert torch.allclose(new_loss, orig_loss, atol=1e-6, rtol=1e-5), (
            f"Losses differ: new={new_loss}, orig={orig_loss}"
        )

        # Compare gradients
        for name, param in orig_model.named_parameters():
            if param.grad is not None and name in new_grads:
                assert torch.allclose(
                    new_grads[name], param.grad, atol=1e-5, rtol=1e-4
                ), (
                    f"Gradients for {name} differ: max diff = {(new_grads[name] - param.grad).abs().max()}"
                )

    def test_cache_behavior_equivalence(self, model_pair):
        """Test that KV cache behavior is equivalent."""
        new_model, orig_model, config = model_pair
        input_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.seq_len))

        with torch.no_grad():
            new_output = new_model(input_ids, use_cache=True)
            orig_output = orig_model(input_ids, use_cache=True)

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

        # Cache contents should be identical
        for i, (new_k, orig_k) in enumerate(
            zip(
                new_output.past_key_values.key_cache,
                orig_output.past_key_values.key_cache,
            )
        ):
            assert torch.allclose(new_k, orig_k, atol=1e-5, rtol=1e-4), (
                f"Key cache layer {i} differs: max diff = {(new_k - orig_k).abs().max()}"
            )

        for i, (new_v, orig_v) in enumerate(
            zip(
                new_output.past_key_values.value_cache,
                orig_output.past_key_values.value_cache,
            )
        ):
            assert torch.allclose(new_v, orig_v, atol=1e-5, rtol=1e-4), (
                f"Value cache layer {i} differs: max diff = {(new_v - orig_v).abs().max()}"
            )


class TestCrossAttentionNonInterference:
    """Test that cross-attention doesn't interfere when disabled."""

    def test_cross_attn_disabled_equals_base(self):
        """Test that cross-attention disabled equals base model behavior."""
        config_params = {
            "parscale_n": 4,
            "parscale_n_tokens": 24,
            "enable_cross_attn": False,  # Disabled
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 1000,
        }

        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, config_params["vocab_size"], (batch_size, seq_len))

        # Set seeds for reproducibility
        torch.manual_seed(42)
        base_config = Qwen2ParScaleConfig(**config_params)
        base_model = ParScaleBaseForCausalLM(base_config)

        torch.manual_seed(42)
        cross_config = Qwen2ParScaleConfig(**config_params)
        cross_model = Qwen2ParScaleForCausalLM(cross_config)

        # Sync weights
        base_state = base_model.state_dict()
        cross_state = cross_model.state_dict()

        for name, param in cross_state.items():
            if name in base_state:
                param.data.copy_(base_state[name].data)

        with torch.no_grad():
            base_output = base_model(input_ids)
            cross_output = cross_model(input_ids)

        # Outputs should be identical when cross-attention is disabled
        assert torch.allclose(
            base_output.logits, cross_output.logits, atol=1e-6, rtol=1e-5
        )


class TestGenerationEquivalence:
    """Test equivalence during text generation."""

    def test_generation_deterministic(self):
        """Test that generation is deterministic and matches between implementations."""
        config_params = {
            "parscale_n": 2,
            "parscale_n_tokens": 16,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 1000,
        }

        # Set seeds
        torch.manual_seed(42)
        new_config = Qwen2ParScaleConfig(**config_params)
        new_model = Qwen2ParScaleForCausalLM(new_config)

        torch.manual_seed(42)
        orig_config = create_ground_truth_config(**config_params)
        orig_model = create_ground_truth_model(orig_config)

        # Sync weights
        new_state = new_model.state_dict()
        orig_state = orig_model.state_dict()
        for name, param in new_state.items():
            if name in orig_state:
                param.data.copy_(orig_state[name].data)

        # Generate text
        input_ids = torch.randint(0, config_params["vocab_size"], (1, 5))

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
        assert torch.equal(new_generated, orig_generated), (
            f"Generated sequences differ:\nNew: {new_generated}\nOrig: {orig_generated}"
        )


class TestParScaleToStandardEquivalence:
    """Test that ParScale with no prefix tokens behaves like standard Qwen2."""

    def test_single_replica_no_prefix_equals_standard(self):
        """Test that parscale_n=1, parscale_n_tokens=0 behaves like standard model."""
        config_params = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
        }

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config_params["vocab_size"], (batch_size, seq_len))

        # Create standard-like ParScale model (no replicas, no prefix tokens)
        torch.manual_seed(42)
        standard_config = Qwen2ParScaleConfig(
            parscale_n=1, 
            parscale_n_tokens=0,
            **config_params
        )
        standard_model = Qwen2ParScaleForCausalLM(standard_config)

        # Create ground truth model with same settings for comparison
        torch.manual_seed(42)
        gt_config = create_ground_truth_config(
            parscale_n=1,
            parscale_n_tokens=0,
            **config_params
        )
        gt_model = create_ground_truth_model(gt_config)

        # Sync weights (both should have identical parameters)
        standard_state = standard_model.state_dict()
        gt_state = gt_model.state_dict()
        
        # Check that they have the same parameter structure
        assert set(standard_state.keys()) == set(gt_state.keys()), (
            "Parameter structures should be identical for parscale_n=1, parscale_n_tokens=0"
        )

        # Sync weights
        for name, param in standard_state.items():
            if name in gt_state:
                param.data.copy_(gt_state[name].data)

        with torch.no_grad():
            standard_output = standard_model(input_ids)
            gt_output = gt_model(input_ids)

        # Outputs should be identical
        assert torch.allclose(
            standard_output.logits, gt_output.logits, atol=1e-6, rtol=1e-5
        ), (
            f"Single replica no prefix should equal standard: "
            f"max diff = {(standard_output.logits - gt_output.logits).abs().max()}"
        )

    def test_multi_replica_no_prefix_equals_standard(self):
        """Test that parscale_n>1, parscale_n_tokens=0 behaves like standard model."""
        config_params = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
        }

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config_params["vocab_size"], (batch_size, seq_len))

        # Create standard-like model (single replica, no prefix)
        torch.manual_seed(42)
        standard_config = Qwen2ParScaleConfig(
            parscale_n=1,
            parscale_n_tokens=0, 
            **config_params
        )
        standard_model = Qwen2ParScaleForCausalLM(standard_config)

        # Create multi-replica model with no prefix tokens (should be equivalent)
        torch.manual_seed(42)
        multi_config = Qwen2ParScaleConfig(
            parscale_n=4,
            parscale_n_tokens=0,  # No prefix tokens = no diversity
            **config_params
        )
        multi_model = Qwen2ParScaleForCausalLM(multi_config)

        # Sync the shared weights (non-ParScale parameters should be identical)
        standard_state = standard_model.state_dict()
        multi_state = multi_model.state_dict()
        
        for name, param in multi_state.items():
            if name in standard_state:
                # Copy from standard to multi-replica model
                param.data.copy_(standard_state[name].data)

        with torch.no_grad():
            standard_output = standard_model(input_ids)
            multi_output = multi_model(input_ids)

        # Outputs should be equivalent since all replicas are identical without prefix tokens
        assert torch.allclose(
            standard_output.logits, multi_output.logits, atol=1e-5, rtol=1e-4
        ), (
            f"Multi-replica with no prefix should equal standard: "
            f"max diff = {(standard_output.logits - multi_output.logits).abs().max()}"
        )

    def test_no_prefix_models_have_no_parscale_cache(self):
        """Test that models with parscale_n_tokens=0 don't create ParscaleCache."""
        config_params = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "parscale_n": 4,
            "parscale_n_tokens": 0,  # No prefix tokens
        }

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config_params["vocab_size"], (batch_size, seq_len))

        config = Qwen2ParScaleConfig(**config_params)
        model = Qwen2ParScaleForCausalLM(config)

        with torch.no_grad():
            output = model(input_ids, use_cache=True)

        # Should use DynamicCache, not ParscaleCache 
        from transformers.cache_utils import DynamicCache
        from parscale_xattn.modeling_base import ParscaleCache
        
        assert isinstance(output.past_key_values, DynamicCache), (
            "With parscale_n_tokens=0, should use DynamicCache"
        )
        assert not isinstance(output.past_key_values, ParscaleCache), (
            "With parscale_n_tokens=0, should NOT use ParscaleCache"
        )


if __name__ == "__main__":
    pytest.main([__file__])
