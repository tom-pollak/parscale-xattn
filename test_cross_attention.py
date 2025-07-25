#!/usr/bin/env python3
"""
Simple test script to reproduce and debug cross-attention shape mismatch issue.
Allows rapid iteration on the fix without running full training.
"""

import torch
from src.parscale_xattn import Qwen2ParScaleConfig, Qwen2ParScaleForCausalLM


def test_cross_attention_shapes(parscale_n=4, seq_len=32, batch_size=2):
    """Test cross-attention with specified configuration."""
    print(f"Testing cross-attention with parscale_n={parscale_n}, seq_len={seq_len}, batch_size={batch_size}")
    
    # Create config with cross-attention enabled
    config = Qwen2ParScaleConfig(
        vocab_size=1000,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=4,  # Small for quick testing
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        parscale_n=parscale_n,
        enable_cross_attn=True,
        parscale_cross_attn_layers=[0, 2],  # Enable on first and third layers
        parscale_n_tokens=48,
    )
    
    print(f"Config: hidden_size={config.hidden_size}, num_attention_heads={config.num_attention_heads}")
    print(f"Cross-attention enabled on layers: {config.parscale_cross_attn_layers}")
    
    # Create model with small config for quick testing
    model = Qwen2ParScaleForCausalLM(config)
    model.eval()
    
    # Create input tensors
    # For parscale_n > 1, input should have shape (parscale_n * batch_size, seq_len)
    input_ids = torch.randint(0, config.vocab_size, (parscale_n * batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    try:
        # Forward pass - this should reproduce the error
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print("✅ Forward pass successful!")
        print(f"Output logits shape: {outputs.logits.shape}")
        return True
        
    except RuntimeError as e:
        print(f"❌ RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_standard_attention(seq_len=32, batch_size=2):
    """Test standard attention (parscale_n=1) for comparison."""
    print(f"\nTesting standard attention (parscale_n=1)")
    
    # Create config without cross-attention (standard mode)
    config = Qwen2ParScaleConfig(
        vocab_size=1000,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=4,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        parscale_n=1,  # Standard mode
        enable_cross_attn=False,
        parscale_n_tokens=0,
    )
    
    model = Qwen2ParScaleForCausalLM(config)
    model.eval()
    
    # Create input tensors (no parscale_n multiplication for standard mode)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print("✅ Standard attention successful!")
        print(f"Output logits shape: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Standard attention failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Attention Refactored Implementation Test")
    print("=" * 60)
    
    # Test standard attention first (should work)
    standard_success = test_standard_attention()
    
    # Test cross-attention with refactored implementation
    cross_success = test_cross_attention_shapes(parscale_n=4)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Standard attention (parscale_n=1): {'✅ PASS' if standard_success else '❌ FAIL'}")
    print(f"Cross attention (parscale_n=4): {'✅ PASS' if cross_success else '❌ FAIL'}")
    
    if cross_success and standard_success:
        print("\n🎉 Both standard and cross-attention work correctly!")
        print("✅ Refactored cross-attention implementation is successful.")
    elif not cross_success and standard_success:
        print("\n🔍 Cross-attention still has issues - needs further debugging.")
    else:
        print("\n❌ Both implementations have issues - check basic model setup.")