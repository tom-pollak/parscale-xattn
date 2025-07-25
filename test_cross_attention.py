#!/usr/bin/env python3
"""
Simple test script to reproduce and debug cross-attention shape mismatch issue.
Allows rapid iteration on the fix without running full training.
"""

import torch
from src.parscale_xattn import Qwen2ParScaleConfig, Qwen2ParScaleForCausalLM


def test_cross_attention_shapes(parscale_n=4, seq_len=32, batch_size=2):
    """Test cross-attention with specified configuration."""
    print(
        f"Testing cross-attention with parscale_n={parscale_n}, seq_len={seq_len}, batch_size={batch_size}"
    )

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

    print(
        f"Config: hidden_size={config.hidden_size}, num_attention_heads={config.num_attention_heads}"
    )
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


def test_invalid_configurations():
    """Test that invalid configurations are caught by validation."""
    print(f"\nTesting invalid configuration validation")

    test_cases = [
        {
            "name": "Cross-attention with parscale_n=1",
            "config": {"parscale_n": 1, "enable_cross_attn": True},
            "should_fail": True,
        },
        {
            "name": "Prefix tokens with parscale_n=1",
            "config": {"parscale_n": 1, "parscale_n_tokens": 48},
            "should_fail": True,
        },
        {
            "name": "Cross-attention layers without enabling cross-attention",
            "config": {
                "parscale_n": 4,
                "enable_cross_attn": False,
                "parscale_cross_attn_layers": [0, 2],
            },
            "should_fail": True,
        },
        {
            "name": "Valid parscale_n=1 (standard Qwen2)",
            "config": {
                "parscale_n": 1,
                "enable_cross_attn": False,
                "parscale_n_tokens": 0,
            },
            "should_fail": False,
        },
    ]

    results = []
    for test_case in test_cases:
        try:
            config = Qwen2ParScaleConfig(
                vocab_size=1000,
                hidden_size=1536,
                num_hidden_layers=4,
                num_attention_heads=12,
                num_key_value_heads=2,
                **test_case["config"],
            )
            # If we get here, config creation succeeded
            if test_case["should_fail"]:
                print(
                    f"❌ {test_case['name']}: Expected validation error but config was created"
                )
                results.append(False)
            else:
                print(f"✅ {test_case['name']}: Valid config created successfully")
                results.append(True)

        except ValueError as e:
            if test_case["should_fail"]:
                print(
                    f"✅ {test_case['name']}: Caught expected validation error: {str(e)[:80]}..."
                )
                results.append(True)
            else:
                print(f"❌ {test_case['name']}: Unexpected validation error: {e}")
                results.append(False)
        except Exception as e:
            print(f"❌ {test_case['name']}: Unexpected error: {e}")
            results.append(False)

    return all(results)


if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Attention Refactored Implementation Test")
    print("=" * 60)

    # Test configuration validation first
    config_validation_success = test_invalid_configurations()

    # Test standard attention (should work)
    standard_success = test_standard_attention()

    # Test cross-attention with refactored implementation
    cross_success = test_cross_attention_shapes(parscale_n=4)

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(
        f"Configuration validation: {'✅ PASS' if config_validation_success else '❌ FAIL'}"
    )
    print(
        f"Standard attention (parscale_n=1): {'✅ PASS' if standard_success else '❌ FAIL'}"
    )
    print(
        f"Cross attention (parscale_n=4): {'✅ PASS' if cross_success else '❌ FAIL'}"
    )

    if config_validation_success and cross_success and standard_success:
        print("\n🎉 All tests passed!")
        print("✅ Configuration validation prevents invalid states")
        print("✅ Refactored cross-attention implementation works correctly")
        print("✅ Tensor dimension assertions will catch runtime inconsistencies")
    elif not config_validation_success:
        print("\n❌ Configuration validation failed - invalid states not being caught")
    elif not cross_success and standard_success:
        print("\n🔍 Cross-attention still has issues - needs further debugging.")
    else:
        print("\n❌ Multiple issues detected - check implementation.")
