"""Unit tests for ParScale prefix token system."""

import pytest
import torch


from parscale_xattn import ParScaleModel, ParScaleConfig
from parscale_xattn.models.base_model import Qwen2Attention, ParscaleCache


class TestPrefixTokenCreation:
    """Test creation of prefix tokens in attention layers."""

    def test_no_prefix_tokens_standard_mode(self, small_config_no_replica):
        """Test that no prefix tokens are created in standard Qwen2 mode (parscale_n=1)."""
        attention = Qwen2Attention(small_config_no_replica, layer_idx=0)

        # Should not have prefix parameters
        assert not hasattr(attention, "prefix_k")
        assert not hasattr(attention, "prefix_v")

    def test_prefix_tokens_created_parscale_mode(self, small_config):
        """Test that prefix tokens are created in ParScale mode (parscale_n > 1)."""
        attention = Qwen2Attention(small_config, layer_idx=0)

        # Should have prefix parameters
        assert hasattr(attention, "prefix_k")
        assert hasattr(attention, "prefix_v")

        # Check shapes match research spec
        expected_shape = (
            small_config.parscale_n,
            small_config.num_key_value_heads,
            small_config.parscale_n_tokens,
            attention.head_dim,
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
            config = ParScaleConfig(parscale_n=parscale_n, parscale_n_tokens=n_tokens)
            attention = Qwen2Attention(config, layer_idx=0)

            expected_shape = (
                parscale_n,
                config.num_key_value_heads,
                n_tokens,
                attention.head_dim,
            )
            assert attention.prefix_k.shape == expected_shape
            assert attention.prefix_v.shape == expected_shape


@pytest.fixture(scope="class")
def parscale_cache(small_config):
    """A ParscaleCache instance for testing."""
    head_dim = small_config.hidden_size // small_config.num_attention_heads
    prefix_k = [
        torch.randn(
            small_config.parscale_n,
            small_config.num_key_value_heads,
            small_config.parscale_n_tokens,
            head_dim,
        )
        for _ in range(small_config.num_hidden_layers)
    ]
    prefix_v = [torch.randn_like(k) for k in prefix_k]
    return ParscaleCache(prefix_k, prefix_v)


@pytest.mark.usefixtures("parscale_cache")
class TestParscaleCache:
    """Test ParscaleCache functionality."""

    batch_size = 2
    seq_len = 10

    def test_parscale_cache_initialization(self, parscale_cache, small_config):
        """Test ParscaleCache initialization."""
        assert parscale_cache.parscale_n == small_config.parscale_n
        assert parscale_cache.n_prefix_tokens == small_config.parscale_n_tokens
        assert len(parscale_cache.key_cache) == small_config.num_hidden_layers
        assert len(parscale_cache.value_cache) == small_config.num_hidden_layers

    def test_cache_sequence_length(self, parscale_cache, small_config):
        """Test sequence length calculation accounts for prefix tokens."""
        assert parscale_cache.get_seq_length() == 0

        head_dim = small_config.hidden_size // small_config.num_attention_heads
        dummy_k = torch.randn(
            small_config.parscale_n * self.batch_size,
            small_config.num_key_value_heads,
            self.seq_len,
            head_dim,
        )
        dummy_v = torch.randn_like(dummy_k)

        parscale_cache.update(dummy_k, dummy_v, layer_idx=0)
        expected_len = self.seq_len
        assert parscale_cache.get_seq_length() == expected_len


class TestAttentionMaskExpansion:
    """Test attention mask expansion for prefix tokens."""

    def test_attention_mask_expansion(self, small_config):
        """Test that attention masks are properly expanded for prefix tokens."""
        attention = Qwen2Attention(small_config, layer_idx=0)
        batch_size = 2
        seq_len = 10

        hidden_states = torch.randn(
            small_config.parscale_n * batch_size, seq_len, small_config.hidden_size
        )
        attention_mask = torch.ones(
            small_config.parscale_n * batch_size, 1, seq_len, seq_len
        )
        position_embeddings = (
            torch.randn(
                small_config.parscale_n * batch_size, seq_len, attention.head_dim
            ),
            torch.randn(
                small_config.parscale_n * batch_size, seq_len, attention.head_dim
            ),
        )

        output, _ = attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        assert output.shape == hidden_states.shape


class TestModelPrefixTokenIntegration:
    """Test prefix token integration in full model."""

    def test_model_creates_parscale_cache(self, small_config):
        """Test that model creates ParscaleCache from prefix tokens."""
        model = ParScaleModel(small_config)

        for layer in model.layers:
            if hasattr(layer.self_attn, "prefix_k"):
                assert layer.self_attn.prefix_k.shape[0] == small_config.parscale_n  # type: ignore
                assert (
                    layer.self_attn.prefix_k.shape[2] == small_config.parscale_n_tokens  # type: ignore
                )

    def test_forward_pass_with_prefix_tokens(self, small_config):
        """Test forward pass correctly uses prefix tokens."""
        model = ParScaleModel(small_config)
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        output = model(input_ids, use_cache=True)

        assert output.past_key_values is not None
        assert isinstance(output.past_key_values, ParscaleCache)
        assert output.past_key_values.parscale_n == small_config.parscale_n
