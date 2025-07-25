#!/usr/bin/env python3
"""Test script to verify parscale_n_tokens=0 disables learnt prefix correctly."""

import torch
from src.parscale_xattn import Qwen2ParScaleForCausalLM, Qwen2ParScaleConfig

def test_disable_learnt_prefix():
    """Test that parscale_n_tokens=0 disables learnt prefix as expected."""
    
    # Test with parscale_n_tokens > 0 (default behavior)
    print("Testing with parscale_n_tokens=8 (prefix enabled)...")
    config_with_prefix = Qwen2ParScaleConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        parscale_n=2,
        parscale_n_tokens=8,
    )
    
    model_with_prefix = Qwen2ParScaleForCausalLM(config_with_prefix)
    
    # Check that prefix parameters exist
    for layer in model_with_prefix.model.layers:
        assert hasattr(layer.self_attn, 'prefix_k'), "prefix_k should exist when parscale_n_tokens > 0"
        assert hasattr(layer.self_attn, 'prefix_v'), "prefix_v should exist when parscale_n_tokens > 0"
        assert layer.self_attn.prefix_k.shape[2] == 8, f"prefix_k should have 8 tokens, got {layer.self_attn.prefix_k.shape[2]}"
    
    print("✓ Prefix parameters exist when parscale_n_tokens > 0")
    
    # Test with parscale_n_tokens=0 (prefix disabled)
    print("\nTesting with parscale_n_tokens=0 (prefix disabled)...")
    config_no_prefix = Qwen2ParScaleConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        parscale_n=2,
        parscale_n_tokens=0,
    )
    
    model_no_prefix = Qwen2ParScaleForCausalLM(config_no_prefix)
    
    # Check that prefix parameters don't exist
    for layer in model_no_prefix.model.layers:
        assert not hasattr(layer.self_attn, 'prefix_k'), "prefix_k should not exist when parscale_n_tokens=0"
        assert not hasattr(layer.self_attn, 'prefix_v'), "prefix_v should not exist when parscale_n_tokens=0"
    
    print("✓ Prefix parameters don't exist when parscale_n_tokens=0")
    
    # Test forward pass with both configurations
    print("\nTesting forward pass...")
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Test with prefix enabled
    with torch.no_grad():
        output_with_prefix = model_with_prefix(input_ids)
        print(f"✓ Forward pass successful with prefix enabled, output shape: {output_with_prefix.logits.shape}")
    
    # Test with prefix disabled
    with torch.no_grad():
        output_no_prefix = model_no_prefix(input_ids)
        print(f"✓ Forward pass successful with prefix disabled, output shape: {output_no_prefix.logits.shape}")
    
    print("\n✅ All tests passed! parscale_n_tokens=0 correctly disables learnt prefix.")

if __name__ == "__main__":
    test_disable_learnt_prefix()