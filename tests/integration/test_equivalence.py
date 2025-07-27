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


class TestNumericalEquivalenceStandardMode:
    """Test numerical equivalence in standard Qwen2 mode (parscale_n=1)."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use small model for faster testing
        self.config_params = {
            "parscale_n": 1,
            "parscale_n_tokens": 0,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 1000,
            "max_position_embeddings": 128,
        }

        self.batch_size = 2
        self.seq_len = 8

        # Set seeds for reproducibility
        torch.manual_seed(42)
        self.new_config = Qwen2ParScaleConfig(**self.config_params)
        self.new_model = Qwen2ParScaleForCausalLM(self.new_config)

        torch.manual_seed(42)
        self.orig_config = create_ground_truth_config(**self.config_params)
        self.orig_model = create_ground_truth_model(self.orig_config)

        # Sync model weights
        self._sync_model_weights()

    def _sync_model_weights(self):
        """Synchronize weights between new and original models."""
        # Copy weights from original to new model to ensure identical starting point
        new_state = self.new_model.state_dict()
        orig_state = self.orig_model.state_dict()

        # Match parameters by name
        for name, param in new_state.items():
            if name in orig_state:
                param.data.copy_(orig_state[name].data)

    def test_forward_pass_equivalence(self):
        """Test that forward passes produce identical outputs."""
        input_ids = torch.randint(
            0, self.config_params["vocab_size"], (self.batch_size, self.seq_len)
        )

        with torch.no_grad():
            new_output = self.new_model(input_ids)
            orig_output = self.orig_model(input_ids)

        # Check logits are identical
        assert torch.allclose(
            new_output.logits, orig_output.logits, atol=1e-6, rtol=1e-5
        ), (
            f"Logits differ: max diff = {(new_output.logits - orig_output.logits).abs().max()}"
        )

        # Check hidden states if available
        if (
            new_output.hidden_states is not None
            and orig_output.hidden_states is not None
        ):
            for i, (new_h, orig_h) in enumerate(
                zip(new_output.hidden_states, orig_output.hidden_states)
            ):
                assert torch.allclose(new_h, orig_h, atol=1e-6, rtol=1e-5), (
                    f"Hidden states layer {i} differ: max diff = {(new_h - orig_h).abs().max()}"
                )

    def test_parameter_count_equivalence(self):
        """Test that both models have the same number of parameters."""
        new_params = sum(p.numel() for p in self.new_model.parameters())
        orig_params = sum(p.numel() for p in self.orig_model.parameters())

        assert new_params == orig_params, (
            f"Parameter count differs: new={new_params}, orig={orig_params}"
        )

    def test_gradient_equivalence(self):
        """Test that gradients are identical during backpropagation."""
        input_ids = torch.randint(
            0, self.config_params["vocab_size"], (self.batch_size, self.seq_len)
        )
        labels = input_ids.clone()

        # Forward and backward pass for new model
        new_output = self.new_model(input_ids, labels=labels)
        new_loss = new_output.loss
        new_loss.backward()

        # Store gradients
        new_grads = {}
        for name, param in self.new_model.named_parameters():
            if param.grad is not None:
                new_grads[name] = param.grad.clone()

        # Reset and do the same for original model
        self.orig_model.zero_grad()
        orig_output = self.orig_model(input_ids, labels=labels)
        orig_loss = orig_output.loss
        orig_loss.backward()

        # Compare losses
        assert torch.allclose(new_loss, orig_loss, atol=1e-6, rtol=1e-5), (
            f"Losses differ: new={new_loss}, orig={orig_loss}"
        )

        # Compare gradients
        for name, param in self.orig_model.named_parameters():
            if param.grad is not None and name in new_grads:
                assert torch.allclose(
                    new_grads[name], param.grad, atol=1e-5, rtol=1e-4
                ), (
                    f"Gradients for {name} differ: max diff = {(new_grads[name] - param.grad).abs().max()}"
                )


class TestNumericalEquivalenceParScaleMode:
    """Test numerical equivalence in ParScale mode (parscale_n > 1)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_params = {
            "parscale_n": 4,
            "parscale_n_tokens": 24,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 1000,
            "max_position_embeddings": 128,
        }

        self.batch_size = 2
        self.seq_len = 6

        # Set seeds for reproducibility
        torch.manual_seed(42)
        self.new_config = Qwen2ParScaleConfig(**self.config_params)
        self.new_model = Qwen2ParScaleForCausalLM(self.new_config)

        torch.manual_seed(42)
        self.orig_config = create_ground_truth_config(**self.config_params)
        self.orig_model = create_ground_truth_model(self.orig_config)

        # Sync model weights
        self._sync_model_weights()

    def _sync_model_weights(self):
        """Synchronize weights between new and original models."""
        new_state = self.new_model.state_dict()
        orig_state = self.orig_model.state_dict()

        for name, param in new_state.items():
            if name in orig_state:
                param.data.copy_(orig_state[name].data)

    def test_forward_pass_equivalence_parscale(self):
        """Test forward pass equivalence in ParScale mode."""
        input_ids = torch.randint(
            0, self.config_params["vocab_size"], (self.batch_size, self.seq_len)
        )

        with torch.no_grad():
            new_output = self.new_model(input_ids)
            orig_output = self.orig_model(input_ids)

        # Check logits are identical
        assert torch.allclose(
            new_output.logits, orig_output.logits, atol=1e-5, rtol=1e-4
        ), (
            f"ParScale logits differ: max diff = {(new_output.logits - orig_output.logits).abs().max()}"
        )

    def test_cache_behavior_equivalence(self):
        """Test that KV cache behavior is equivalent."""
        input_ids = torch.randint(
            0, self.config_params["vocab_size"], (self.batch_size, self.seq_len)
        )

        with torch.no_grad():
            new_output = self.new_model(input_ids, use_cache=True)
            orig_output = self.orig_model(input_ids, use_cache=True)

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

    def test_aggregation_equivalence(self):
        """Test that output aggregation produces identical results."""
        input_ids = torch.randint(
            0, self.config_params["vocab_size"], (self.batch_size, self.seq_len)
        )

        # Get intermediate hidden states before aggregation by modifying forward pass
        new_model_outputs = []
        orig_model_outputs = []

        def capture_before_aggregation(module, input, output):
            # Capture hidden states before final aggregation
            if hasattr(module, "parscale_n") and module.parscale_n > 1:
                # This is called after norm but before aggregation
                hidden_states = output.last_hidden_state
                if hidden_states.shape[0] == module.parscale_n * self.batch_size:
                    new_model_outputs.append(hidden_states.clone())

        def capture_before_aggregation_orig(module, input, output):
            if hasattr(module, "parscale_n") and module.parscale_n > 1:
                hidden_states = output.last_hidden_state
                if hidden_states.shape[0] == module.parscale_n * self.batch_size:
                    orig_model_outputs.append(hidden_states.clone())

        # Register hooks to capture pre-aggregation states
        new_hook = self.new_model.model.register_forward_hook(
            capture_before_aggregation
        )
        orig_hook = self.orig_model.model.register_forward_hook(
            capture_before_aggregation_orig
        )

        with torch.no_grad():
            new_output = self.new_model(input_ids)
            orig_output = self.orig_model(input_ids)

        # Remove hooks
        new_hook.remove()
        orig_hook.remove()

        # Final outputs should be identical
        assert torch.allclose(
            new_output.logits, orig_output.logits, atol=1e-5, rtol=1e-4
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


if __name__ == "__main__":
    pytest.main([__file__])
