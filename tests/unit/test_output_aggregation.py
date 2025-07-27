"""Unit tests for ParScale output aggregation system."""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path
from einops import rearrange

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from parscale_xattn import Qwen2ParScaleConfig
from parscale_xattn.modeling_base import ParScaleBaseModel
from parscale_xattn.modeling_cross_attn import ParScaleCrossAttnModel

# Import ground truth for comparison
sys.path.append(str(Path(__file__).parent.parent / "ground_truth"))
from config import create_ground_truth_config, create_ground_truth_base_model


class TestAggregationLayer:
    """Test the MLP aggregation layer structure."""
    
    def test_aggregation_layer_creation_standard_mode(self):
        """Test that no aggregation layer is created in standard mode (parscale_n=1)."""
        config = Qwen2ParScaleConfig(parscale_n=1, parscale_n_tokens=0)
        model = ParScaleBaseModel(config)
        
        # Should not have aggregation layer
        assert not hasattr(model, 'aggregate_layer') or model.aggregate_layer is None
    
    def test_aggregation_layer_creation_parscale_mode(self):
        """Test aggregation layer creation in ParScale mode."""
        config = Qwen2ParScaleConfig(parscale_n=4, parscale_n_tokens=48)
        model = ParScaleBaseModel(config)
        
        # Should have aggregation layer
        assert hasattr(model, 'aggregate_layer')
        assert model.aggregate_layer is not None
        
        # Check structure matches research spec
        assert isinstance(model.aggregate_layer, nn.Sequential)
        assert len(model.aggregate_layer) == 3  # Linear -> SiLU -> Linear
        
        # Check first linear layer
        first_layer = model.aggregate_layer[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == config.parscale_n * config.hidden_size
        assert first_layer.out_features == config.hidden_size
        
        # Check activation
        activation = model.aggregate_layer[1]
        assert isinstance(activation, nn.SiLU)
        
        # Check second linear layer
        second_layer = model.aggregate_layer[2]
        assert isinstance(second_layer, nn.Linear)
        assert second_layer.in_features == config.hidden_size
        assert second_layer.out_features == config.parscale_n
    
    def test_aggregation_layer_different_configurations(self):
        """Test aggregation layer for different ParScale configurations."""
        configs = [
            (2, 128),   # 2 replicas, hidden_size 128
            (4, 256),   # 4 replicas, hidden_size 256
            (8, 512),   # 8 replicas, hidden_size 512
        ]
        
        for parscale_n, hidden_size in configs:
            config = Qwen2ParScaleConfig(
                parscale_n=parscale_n,
                parscale_n_tokens=48,
                hidden_size=hidden_size
            )
            model = ParScaleBaseModel(config)
            
            # Check dimensions
            first_layer = model.aggregate_layer[0]
            assert first_layer.in_features == parscale_n * hidden_size
            assert first_layer.out_features == hidden_size
            
            second_layer = model.aggregate_layer[2]
            assert second_layer.out_features == parscale_n


class TestOutputAggregation:
    """Test the output aggregation mechanism."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Qwen2ParScaleConfig(
            parscale_n=4,
            parscale_n_tokens=48,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4
        )
        self.model = ParScaleCrossAttnModel(self.config)
        self.batch_size = 2
        self.seq_len = 5
    
    def test_input_replication(self):
        """Test that inputs are correctly replicated across replicas."""
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        
        # Get input embeddings 
        inputs_embeds = self.model.embed_tokens(input_ids)
        original_shape = inputs_embeds.shape
        
        # Should be replicated to (parscale_n * batch_size, seq_len, hidden_size)
        replicated_embeds = rearrange(
            inputs_embeds, "b s h -> (n_parscale b) s h", n_parscale=self.config.parscale_n
        )
        
        expected_shape = (
            self.config.parscale_n * self.batch_size,
            self.seq_len,
            self.config.hidden_size
        )
        assert replicated_embeds.shape == expected_shape
    
    def test_aggregation_attention_computation(self):
        """Test the dynamic weighted sum computation."""
        # Create dummy hidden states from replicas
        hidden_states = torch.randn(
            self.config.parscale_n * self.batch_size,
            self.seq_len,
            self.config.hidden_size
        )
        
        # Compute aggregation attention as in research spec
        attn = torch.unsqueeze(
            torch.softmax(
                self.model.aggregate_layer(
                    rearrange(
                        hidden_states,
                        "(n_parscale b) s h -> b s (h n_parscale)",
                        n_parscale=self.config.parscale_n,
                    )
                ).float(),
                dim=-1,
            ),
            dim=-1,
        )  # [b s n_parscale 1]
        
        # Check attention shape and properties
        expected_attn_shape = (self.batch_size, self.seq_len, self.config.parscale_n, 1)
        assert attn.shape == expected_attn_shape
        
        # Attention weights should sum to 1 across replicas
        attn_sum = attn.squeeze(-1).sum(dim=-1)  # Sum across parscale_n dimension
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6)
    
    def test_attention_smoothing(self):
        """Test attention smoothing mechanism."""
        # Test with smoothing enabled
        self.model.parscale_aggregate_attn_smoothing = 0.1
        
        hidden_states = torch.randn(
            self.config.parscale_n * self.batch_size,
            self.seq_len,
            self.config.hidden_size
        )
        
        # Compute base attention
        attn = torch.unsqueeze(
            torch.softmax(
                self.model.aggregate_layer(
                    rearrange(
                        hidden_states,
                        "(n_parscale b) s h -> b s (h n_parscale)",
                        n_parscale=self.config.parscale_n,
                    )
                ).float(),
                dim=-1,
            ),
            dim=-1,
        )
        
        # Apply smoothing as in research spec
        smoothing = self.model.parscale_aggregate_attn_smoothing
        smoothed_attn = attn * (1 - smoothing) + (smoothing / self.config.parscale_n)
        
        # Smoothed attention should still sum to 1
        smoothed_sum = smoothed_attn.squeeze(-1).sum(dim=-1)
        assert torch.allclose(smoothed_sum, torch.ones_like(smoothed_sum), atol=1e-6)
        
        # Smoothed attention should be more uniform than original
        attn_var = attn.squeeze(-1).var(dim=-1)
        smoothed_var = smoothed_attn.squeeze(-1).var(dim=-1)
        assert torch.all(smoothed_var <= attn_var)
    
    def test_final_weighted_sum(self):
        """Test the final weighted sum aggregation."""
        hidden_states = torch.randn(
            self.config.parscale_n * self.batch_size,
            self.seq_len,
            self.config.hidden_size
        )
        
        # Compute attention weights
        attn = torch.unsqueeze(
            torch.softmax(
                self.model.aggregate_layer(
                    rearrange(
                        hidden_states,
                        "(n_parscale b) s h -> b s (h n_parscale)",
                        n_parscale=self.config.parscale_n,
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
                n_parscale=self.config.parscale_n,
            ) * attn,
            dim=2,
            keepdim=False,
        ).to(hidden_states.dtype)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        assert aggregated.shape == expected_shape
    
    def test_end_to_end_aggregation(self):
        """Test end-to-end output aggregation in model forward pass."""
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward pass
        output = self.model(input_ids)
        
        # Output should have correct shape (batch_size, seq_len, hidden_size)
        # Not (parscale_n * batch_size, seq_len, hidden_size)
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        assert output.last_hidden_state.shape == expected_shape


class TestAggregationCompatibility:
    """Test aggregation compatibility with ground truth."""
    
    def test_aggregation_layer_structure_compatibility(self):
        """Test that aggregation layer structure matches ground truth."""
        config_params = {
            "parscale_n": 4,
            "parscale_n_tokens": 48,
            "hidden_size": 128,
            "num_hidden_layers": 2,
        }
        
        new_config = Qwen2ParScaleConfig(**config_params)
        orig_config = create_ground_truth_config(**config_params)
        
        new_model = ParScaleCrossAttnModel(new_config)
        orig_model = create_ground_truth_base_model(orig_config)
        
        # Both should have aggregation layers
        assert hasattr(new_model, 'aggregate_layer')
        assert hasattr(orig_model, 'aggregate_layer')
        
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
    
    def test_smoothing_parameter_compatibility(self):
        """Test that smoothing parameters match ground truth."""
        config_params = {
            "parscale_n": 4,
            "parscale_attn_smooth": 0.05,
        }
        
        new_config = Qwen2ParScaleConfig(**config_params)
        orig_config = create_ground_truth_config(**config_params)
        
        new_model = ParScaleCrossAttnModel(new_config)
        orig_model = create_ground_truth_base_model(orig_config)
        
        assert new_model.parscale_aggregate_attn_smoothing == orig_model.parscale_aggregate_attn_smoothing


class TestAggregationMathematicalProperties:
    """Test mathematical properties of the aggregation system."""
    
    def test_aggregation_preserves_batch_dimension(self):
        """Test that aggregation correctly handles batch dimension."""
        config = Qwen2ParScaleConfig(parscale_n=4, hidden_size=64, num_hidden_layers=1)
        model = ParScaleCrossAttnModel(config)
        
        batch_sizes = [1, 2, 4, 8]
        seq_len = 3
        
        for batch_size in batch_sizes:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            output = model(input_ids)
            
            # Output batch size should match input
            assert output.last_hidden_state.shape[0] == batch_size
    
    def test_aggregation_deterministic(self):
        """Test that aggregation is deterministic for same inputs."""
        config = Qwen2ParScaleConfig(parscale_n=2, hidden_size=32, num_hidden_layers=1)
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        model1 = ParScaleCrossAttnModel(config)
        
        torch.manual_seed(42)
        model2 = ParScaleCrossAttnModel(config)
        
        # Same input
        input_ids = torch.randint(0, config.vocab_size, (1, 3))
        
        with torch.no_grad():
            output1 = model1(input_ids)
            output2 = model2(input_ids)
        
        # Outputs should be identical
        assert torch.allclose(output1.last_hidden_state, output2.last_hidden_state)


if __name__ == "__main__":
    pytest.main([__file__])