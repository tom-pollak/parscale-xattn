"""Unit tests for ParScale output aggregation system."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from einops import rearrange

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from parscale_xattn import Qwen2ParScaleConfig
from parscale_xattn.modeling_base import ParScaleBaseModel
from parscale_xattn.modeling_cross_attn import ParScaleCrossAttnModel

# Import ground truth for comparison
sys.path.append(str(Path(__file__).parent.parent / "ground_truth"))
from config import create_ground_truth_base_model, create_ground_truth_config


@pytest.fixture(scope="class")
def model(small_config):
    """A small model for fast testing, scoped to the class."""
    return ParScaleCrossAttnModel(small_config)


class TestAggregationLayer:
    """Test the MLP aggregation layer structure."""

    def test_aggregation_layer_creation_standard_mode(self, small_config_no_replica):
        """Test that no aggregation layer is created in standard mode (parscale_n=1)."""
        model = ParScaleBaseModel(small_config_no_replica)
        assert not hasattr(model, "aggregate_layer") or model.aggregate_layer is None

    def test_aggregation_layer_creation_parscale_mode(self, small_config):
        """Test aggregation layer creation in ParScale mode."""
        model = ParScaleBaseModel(small_config)

        # Should have aggregation layer
        assert hasattr(model, "aggregate_layer")
        assert model.aggregate_layer is not None

        # Check structure matches research spec
        assert isinstance(model.aggregate_layer, nn.Sequential)
        assert len(model.aggregate_layer) == 3  # Linear -> SiLU -> Linear

        # Check first linear layer
        first_layer = model.aggregate_layer[0]
        assert isinstance(first_layer, nn.Linear)
        assert (
            first_layer.in_features
            == small_config.parscale_n * small_config.hidden_size
        )
        assert first_layer.out_features == small_config.hidden_size

        # Check activation
        activation = model.aggregate_layer[1]
        assert isinstance(activation, nn.SiLU)

        # Check second linear layer
        second_layer = model.aggregate_layer[2]
        assert isinstance(second_layer, nn.Linear)
        assert second_layer.in_features == small_config.hidden_size
        assert second_layer.out_features == small_config.parscale_n

    def test_aggregation_layer_different_configurations(self, small_config_dict):
        """Test aggregation layer for different ParScale configurations."""
        configs = [
            (2, 128),  # 2 replicas, hidden_size 128
            (4, 256),  # 4 replicas, hidden_size 256
            (8, 512),  # 8 replicas, hidden_size 512
        ]

        for parscale_n, hidden_size in configs:
            config_dict_copy = small_config_dict.copy()
            config_dict_copy.update(
                dict(parscale_n=parscale_n, hidden_size=hidden_size)
            )
            config = Qwen2ParScaleConfig(**config_dict_copy)
            model = ParScaleBaseModel(config)

            # Check dimensions
            first_layer = model.aggregate_layer[0]
            assert first_layer.in_features == parscale_n * hidden_size
            assert first_layer.out_features == hidden_size

            second_layer = model.aggregate_layer[2]
            assert second_layer.out_features == parscale_n


@pytest.mark.usefixtures("model")
class TestOutputAggregation:
    """Test the output aggregation mechanism."""

    batch_size = 2
    seq_len = 5

    def test_input_replication(self, model, small_config):
        """Test that inputs are correctly replicated across replicas."""
        input_ids = torch.randint(
            0, small_config.vocab_size, (self.batch_size, self.seq_len)
        )

        # Get input embeddings
        inputs_embeds = model.embed_tokens(input_ids)

        # Should be replicated to (parscale_n * batch_size, seq_len, hidden_size)
        replicated_embeds = repeat(
            inputs_embeds,
            "b s h -> (n_parscale b) s h",
            n_parscale=small_config.parscale_n,
        )

        expected_shape = (
            small_config.parscale_n * self.batch_size,
            self.seq_len,
            small_config.hidden_size,
        )
        assert replicated_embeds.shape == expected_shape

    def test_aggregation_attention_computation(self, model, small_config):
        """Test the dynamic weighted sum computation."""
        # Create dummy hidden states from replicas
        hidden_states = torch.randn(
            small_config.parscale_n * self.batch_size,
            self.seq_len,
            small_config.hidden_size,
        )

        # Compute aggregation attention as in research spec
        attn = torch.unsqueeze(
            torch.softmax(
                model.aggregate_layer(
                    rearrange(
                        hidden_states,
                        "(n_parscale b) s h -> b s (h n_parscale)",
                        n_parscale=small_config.parscale_n,
                    )
                ).float(),
                dim=-1,
            ),
            dim=-1,
        )  # [b s n_parscale 1]

        # Check attention shape and properties
        expected_attn_shape = (
            self.batch_size,
            self.seq_len,
            small_config.parscale_n,
            1,
        )
        assert attn.shape == expected_attn_shape

        # Attention weights should sum to 1 across replicas
        attn_sum = attn.squeeze(-1).sum(dim=-1)  # Sum across parscale_n dimension
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6)

    def test_attention_smoothing(self, model, small_config):
        """Test attention smoothing mechanism."""
        # Test with smoothing enabled
        model.parscale_aggregate_attn_smoothing = 0.1

        hidden_states = torch.randn(
            small_config.parscale_n * self.batch_size,
            self.seq_len,
            small_config.hidden_size,
        )

        # Compute base attention
        attn = torch.unsqueeze(
            torch.softmax(
                model.aggregate_layer(
                    rearrange(
                        hidden_states,
                        "(n_parscale b) s h -> b s (h n_parscale)",
                        n_parscale=small_config.parscale_n,
                    )
                ).float(),
                dim=-1,
            ),
            dim=-1,
        )

        # Apply smoothing as in research spec
        smoothing = model.parscale_aggregate_attn_smoothing
        smoothed_attn = attn * (1 - smoothing) + (smoothing / small_config.parscale_n)

        # Smoothed attention should still sum to 1
        smoothed_sum = smoothed_attn.squeeze(-1).sum(dim=-1)
        assert torch.allclose(smoothed_sum, torch.ones_like(smoothed_sum), atol=1e-6)

        # Smoothed attention should be more uniform than original
        attn_var = attn.squeeze(-1).var(dim=-1)
        smoothed_var = smoothed_attn.squeeze(-1).var(dim=-1)
        assert torch.all(smoothed_var <= attn_var)

        # Reset smoothing
        model.parscale_aggregate_attn_smoothing = small_config.parscale_attn_smooth

    def test_final_weighted_sum(self, model, small_config):
        """Test the final weighted sum aggregation."""
        hidden_states = torch.randn(
            small_config.parscale_n * self.batch_size,
            self.seq_len,
            small_config.hidden_size,
        )

        # Compute attention weights
        attn = torch.unsqueeze(
            torch.softmax(
                model.aggregate_layer(
                    rearrange(
                        hidden_states,
                        "(n_parscale b) s h -> b s (h n_parscale)",
                        n_parscale=small_config.parscale_n,
                    )
                ).float(),
                dim=-1,
            ),
            dim=-1,
        )

        # Apply weighted sum as in research spec
        aggregated = torch.sum(
            rearrange(
                hidden_states,
                "(n_parscale b) s h -> b s n_parscale h",
                n_parscale=small_config.parscale_n,
            )
            * attn,
            dim=2,
            keepdim=False,
        ).to(hidden_states.dtype)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, small_config.hidden_size)
        assert aggregated.shape == expected_shape

    def test_end_to_end_aggregation(self, model, small_config):
        """Test end-to-end output aggregation in model forward pass."""
        input_ids = torch.randint(
            0, small_config.vocab_size, (self.batch_size, self.seq_len)
        )

        # Forward pass
        output = model(input_ids)

        # Output should have correct shape (batch_size, seq_len, hidden_size)
        # Not (parscale_n * batch_size, seq_len, hidden_size)
        expected_shape = (self.batch_size, self.seq_len, small_config.hidden_size)
        assert output.last_hidden_state.shape == expected_shape


class TestAggregationCompatibility:
    """Test aggregation compatibility with ground truth."""

    def test_aggregation_layer_structure_compatibility(self, small_config_dict):
        """Test that aggregation layer structure matches ground truth."""
        new_config = Qwen2ParScaleConfig(**small_config_dict)
        orig_config = create_ground_truth_config(**small_config_dict)

        new_model = ParScaleCrossAttnModel(new_config)
        orig_model = create_ground_truth_base_model(orig_config)

        # Both should have aggregation layers
        assert hasattr(new_model, "aggregate_layer")
        assert hasattr(orig_model, "aggregate_layer")

        # Structure should match
        assert len(new_model.aggregate_layer) == len(orig_model.aggregate_layer)

        # Dimensions should match
        new_first = new_model.aggregate_layer[0]
        orig_first = orig_model.aggregate_layer[0]
        assert new_first.in_features == orig_first.in_features
        assert new_first.out_features == orig_first.out_features

        new_second = new_model.aggregate_layer[2]
        orig_second = orig_model.aggregate_layer[2]
        assert new_second.in_features == orig_second.in_features
        assert new_second.out_features == orig_second.out_features

    def test_smoothing_parameter_compatibility(self, small_config_dict):
        """Test that smoothing parameters match ground truth."""
        config_params = {**small_config_dict, "parscale_attn_smooth": 0.05}

        new_config = Qwen2ParScaleConfig(**config_params)
        orig_config = create_ground_truth_config(**config_params)

        new_model = ParScaleCrossAttnModel(new_config)
        orig_model = create_ground_truth_base_model(orig_config)

        assert (
            new_model.parscale_aggregate_attn_smoothing
            == orig_model.parscale_aggregate_attn_smoothing
        )


class TestAggregationMathematicalProperties:
    """Test mathematical properties of the aggregation system."""

    def test_aggregation_preserves_batch_dimension(self, small_config):
        """Test that aggregation correctly handles batch dimension."""
        model = ParScaleCrossAttnModel(small_config)

        batch_sizes = [1, 2, 4, 8]
        seq_len = 3

        for batch_size in batch_sizes:
            input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
            output = model(input_ids)

            # Output batch size should match input
            assert output.last_hidden_state.shape[0] == batch_size

    def test_aggregation_deterministic(self, small_config):
        """Test that aggregation is deterministic for same inputs."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        model1 = ParScaleCrossAttnModel(small_config)

        torch.manual_seed(42)
        model2 = ParScaleCrossAttnModel(small_config)

        # Same input
        input_ids = torch.randint(0, small_config.vocab_size, (1, 3))

        with torch.no_grad():
            output1 = model1(input_ids)
            output2 = model2(input_ids)

        # Outputs should be identical
        assert torch.allclose(output1.last_hidden_state, output2.last_hidden_state)


if __name__ == "__main__":
    pytest.main([__file__])
