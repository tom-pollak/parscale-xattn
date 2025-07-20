#!/usr/bin/env python3
"""Test script to verify ParScale cross-attention functionality."""

from src.parscale_xattn import Qwen2ParScaleForCausalLM, Qwen2ParScaleConfig
import torch

def test_configurations():
    """Test different configuration options."""
    print("Testing ParScale Cross-Attention Extension")
    print("=" * 50)
    
    # Test 1: Standard Qwen2 behavior (parscale_n=1)
    print("1. Testing standard Qwen2 behavior (parscale_n=1)...")
    config = Qwen2ParScaleConfig(parscale_n=1)
    model = Qwen2ParScaleForCausalLM(config)
    print(f"   ✓ Config: parscale_n={config.parscale_n}, cross_attn={config.parscale_enable_cross_attn}")
    
    # Test 2: Original ParScale behavior (cross-attention disabled)
    print("2. Testing original ParScale behavior (cross-attention disabled)...")
    config = Qwen2ParScaleConfig(
        parscale_n=4, 
        parscale_n_tokens=48,
        parscale_enable_cross_attn=False
    )
    model = Qwen2ParScaleForCausalLM(config)
    print(f"   ✓ Config: parscale_n={config.parscale_n}, cross_attn={config.parscale_enable_cross_attn}")
    
    # Test 3: New cross-attention extension (all layers)
    print("3. Testing cross-attention extension (all layers)...")
    config = Qwen2ParScaleConfig(
        parscale_n=4, 
        parscale_n_tokens=48,
        parscale_enable_cross_attn=True,
        parscale_cross_attn_layers=None
    )
    model = Qwen2ParScaleForCausalLM(config)
    print(f"   ✓ Config: parscale_n={config.parscale_n}, cross_attn={config.parscale_enable_cross_attn}")
    print(f"   ✓ Cross-attn layers: {config.parscale_cross_attn_layers}")
    
    # Test 4: Cross-attention on specific layers only
    print("4. Testing cross-attention on specific layers...")
    config = Qwen2ParScaleConfig(
        parscale_n=4, 
        parscale_n_tokens=48,
        parscale_enable_cross_attn=True,
        parscale_cross_attn_layers=[0, 4, 8, 12]
    )
    model = Qwen2ParScaleForCausalLM(config)
    print(f"   ✓ Config: parscale_n={config.parscale_n}, cross_attn={config.parscale_enable_cross_attn}")
    print(f"   ✓ Cross-attn layers: {config.parscale_cross_attn_layers}")
    
    print("\nAll configuration tests passed!")
    
def test_forward_pass():
    """Test basic forward pass functionality."""
    print("\nTesting forward pass functionality")
    print("-" * 40)
    
    # Test with a small model and simple input
    config = Qwen2ParScaleConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        parscale_n=2,
        parscale_n_tokens=4,
        parscale_enable_cross_attn=True
    )
    
    model = Qwen2ParScaleForCausalLM(config)
    model.eval()
    
    # Create simple input
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        print(f"Output logits shape: {logits.shape}")
        print("   ✓ Forward pass completed successfully")
    
    print("Forward pass test passed!")

if __name__ == "__main__":
    test_configurations()
    test_forward_pass()
    print("\n🎉 All tests completed successfully!")