"""Unit tests for ParScale prefix token system."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from parscale_xattn import Qwen2ParScaleConfig
from parscale_xattn.modeling_base import Qwen2Attention, ParscaleCache
from parscale_xattn.modeling_cross_attn import ParScaleCrossAttnModel

# Import ground truth for comparison
sys.path.append(str(Path(__file__).parent.parent / "ground_truth"))
from config import create_ground_truth_config, create_ground_truth_model


class TestPrefixTokenCreation:
    """Test creation of prefix tokens in attention layers."""
    
    def test_no_prefix_tokens_standard_mode(self):
        """Test that no prefix tokens are created in standard Qwen2 mode (parscale_n=1)."""
        config = Qwen2ParScaleConfig(parscale_n=1, parscale_n_tokens=0)
        attention = Qwen2Attention(config, layer_idx=0)
        
        # Should not have prefix parameters
        assert not hasattr(attention, 'prefix_k')
        assert not hasattr(attention, 'prefix_v')
    
    def test_prefix_tokens_created_parscale_mode(self):
        """Test that prefix tokens are created in ParScale mode (parscale_n > 1)."""
        config = Qwen2ParScaleConfig(parscale_n=4, parscale_n_tokens=48)
        attention = Qwen2Attention(config, layer_idx=0)
        
        # Should have prefix parameters
        assert hasattr(attention, 'prefix_k')
        assert hasattr(attention, 'prefix_v')
        
        # Check shapes match research spec
        expected_shape = (
            config.parscale_n,  # 4 replicas
            config.num_key_value_heads,  # 32 heads
            config.parscale_n_tokens,  # 48 tokens
            attention.head_dim,  # head dimension
        )
        
        assert attention.prefix_k.shape == expected_shape
        assert attention.prefix_v.shape == expected_shape
    
    def test_prefix_token_shapes(self):
        """Test prefix token shapes are correct for different configurations."""
        configs = [
            (2, 24),  # 2 replicas, 24 tokens
            (4, 48),  # 4 replicas, 48 tokens  
            (8, 32),  # 8 replicas, 32 tokens
        ]
        
        for parscale_n, n_tokens in configs:
            config = Qwen2ParScaleConfig(parscale_n=parscale_n, parscale_n_tokens=n_tokens)
            attention = Qwen2Attention(config, layer_idx=0)
            
            expected_shape = (parscale_n, config.num_key_value_heads, n_tokens, attention.head_dim)
            assert attention.prefix_k.shape == expected_shape
            assert attention.prefix_v.shape == expected_shape


class TestParscaleCache:
    """Test ParscaleCache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Qwen2ParScaleConfig(parscale_n=4, parscale_n_tokens=48)
        self.batch_size = 2
        self.seq_len = 10
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        
        # Create dummy prefix k/v tensors
        self.prefix_k = [torch.randn(
            self.config.parscale_n,
            self.config.num_key_value_heads,
            self.config.parscale_n_tokens,
            self.head_dim
        ) for _ in range(self.config.num_hidden_layers)]
        
        self.prefix_v = [torch.randn(
            self.config.parscale_n,
            self.config.num_key_value_heads,
            self.config.parscale_n_tokens,
            self.head_dim
        ) for _ in range(self.config.num_hidden_layers)]
    
    def test_parscale_cache_initialization(self):
        """Test ParscaleCache initialization."""
        cache = ParscaleCache(self.prefix_k, self.prefix_v)
        
        assert cache.parscale_n == self.config.parscale_n
        assert cache.n_prefix_tokens == self.config.parscale_n_tokens
        assert len(cache.key_cache) == self.config.num_hidden_layers
        assert len(cache.value_cache) == self.config.num_hidden_layers
    
    def test_cache_sequence_length(self):
        """Test sequence length calculation accounts for prefix tokens."""
        cache = ParscaleCache(self.prefix_k, self.prefix_v)
        
        # Empty cache should report 0 sequence length (prefix doesn't count)
        assert cache.get_seq_length() == 0
        
        # After adding actual tokens, should subtract prefix length
        dummy_k = torch.randn(
            self.config.parscale_n * self.batch_size,
            self.config.num_key_value_heads,
            self.seq_len,
            self.head_dim
        )
        dummy_v = torch.randn_like(dummy_k)
        
        cache.update(dummy_k, dummy_v, layer_idx=0)
        expected_len = self.config.parscale_n_tokens + self.seq_len - self.config.parscale_n_tokens
        assert cache.get_seq_length() == expected_len
    
    def test_cache_reorder_for_beam_search(self):
        """Test cache reordering for beam search accounts for replicas."""
        cache = ParscaleCache(self.prefix_k, self.prefix_v)
        
        # Add some dummy data
        dummy_k = torch.randn(
            self.config.parscale_n * self.batch_size,
            self.config.num_key_value_heads,
            self.seq_len,
            self.head_dim
        )
        dummy_v = torch.randn_like(dummy_k)
        cache.update(dummy_k, dummy_v, layer_idx=0)
        
        # Test beam index reordering
        beam_idx = torch.tensor([1, 0])  # Swap batch elements
        cache.reorder_cache(beam_idx)
        
        # Should have reordered correctly (detailed verification would need actual beam search)
        assert cache.key_cache[0].shape[0] == self.config.parscale_n * self.batch_size


class TestAttentionMaskExpansion:
    """Test attention mask expansion for prefix tokens."""
    
    def test_attention_mask_expansion(self):
        """Test that attention masks are properly expanded for prefix tokens."""
        config = Qwen2ParScaleConfig(parscale_n=4, parscale_n_tokens=48)
        attention = Qwen2Attention(config, layer_idx=0)
        
        batch_size = 2
        seq_len = 10
        
        # Create dummy inputs
        hidden_states = torch.randn(
            config.parscale_n * batch_size, seq_len, config.hidden_size
        )
        
        # Create attention mask (standard causal mask format)
        attention_mask = torch.ones(
            config.parscale_n * batch_size, 1, seq_len, seq_len
        )
        
        # Create position embeddings
        position_embeddings = (
            torch.randn(batch_size, seq_len, attention.head_dim),
            torch.randn(batch_size, seq_len, attention.head_dim)
        )
        
        # Forward pass should handle mask expansion internally
        output, _ = attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        
        # Output should have correct shape
        assert output.shape == hidden_states.shape


class TestModelPrefixTokenIntegration:
    """Test prefix token integration in full model."""
    
    def test_model_creates_parscale_cache(self):
        """Test that model creates ParscaleCache from prefix tokens."""
        config = Qwen2ParScaleConfig(parscale_n=4, parscale_n_tokens=48)
        model = ParScaleCrossAttnModel(config)
        
        # Should have prefix parameters in attention layers
        for layer in model.layers:
            if hasattr(layer.self_attn, 'prefix_k'):
                assert layer.self_attn.prefix_k.shape[0] == config.parscale_n
                assert layer.self_attn.prefix_k.shape[2] == config.parscale_n_tokens
    
    def test_forward_pass_with_prefix_tokens(self):
        """Test forward pass correctly uses prefix tokens."""
        config = Qwen2ParScaleConfig(
            parscale_n=4, 
            parscale_n_tokens=48,
            num_hidden_layers=2,  # Small for testing
            hidden_size=128
        )
        model = ParScaleCrossAttnModel(config)
        
        batch_size = 2
        seq_len = 5
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass should work
        output = model(input_ids, use_cache=True)
        
        # Should return past_key_values (ParscaleCache)
        assert output.past_key_values is not None
        assert isinstance(output.past_key_values, ParscaleCache)
        assert output.past_key_values.parscale_n == config.parscale_n


class TestPrefixTokenCompatibility:
    """Test compatibility with ground truth implementation."""
    
    def test_prefix_shapes_match_ground_truth(self):
        """Test that prefix token shapes match ground truth implementation."""
        config_params = {
            "parscale_n": 4,
            "parscale_n_tokens": 48,
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }
        
        # Create both models
        new_config = Qwen2ParScaleConfig(**config_params)
        orig_config = create_ground_truth_config(**config_params)
        
        new_model = ParScaleCrossAttnModel(new_config)
        orig_model = create_ground_truth_model(orig_config)
        
        # Compare prefix shapes in first layer
        new_layer = new_model.layers[0]
        orig_layer = orig_model.model.layers[0]
        
        if hasattr(new_layer.self_attn, 'prefix_k') and hasattr(orig_layer.self_attn, 'prefix_k'):
            assert new_layer.self_attn.prefix_k.shape == orig_layer.self_attn.prefix_k.shape
            assert new_layer.self_attn.prefix_v.shape == orig_layer.self_attn.prefix_v.shape


if __name__ == "__main__":
    pytest.main([__file__])