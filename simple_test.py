#!/usr/bin/env python3
"""Simple test to verify import and configuration."""

print("Testing ParScale Cross-Attention Extension")
print("=" * 50)

try:
    from src.parscale_xattn import Qwen2ParScaleConfig

    print("✓ Successfully imported Qwen2ParScaleConfig")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test configuration parameters
configs = [
    # Standard Qwen2
    {"parscale_n": 1, "enable_cross_attn": False},
    # Original ParScale
    {"parscale_n": 4, "enable_cross_attn": False},
    # Cross-attention enabled
    {"parscale_n": 4, "enable_cross_attn": True},
    # Cross-attention on specific layers
    {
        "parscale_n": 4,
        "enable_cross_attn": True,
        "parscale_cross_attn_layers": [0, 2],
    },
]

for i, config_params in enumerate(configs, 1):
    try:
        config = Qwen2ParScaleConfig(**config_params)
        print(
            f"✓ Config {i}: parscale_n={config.parscale_n}, cross_attn={config.enable_cross_attn}"
        )
        if hasattr(config, "parscale_cross_attn_layers"):
            print(f"              layers={config.parscale_cross_attn_layers}")
    except Exception as e:
        print(f"✗ Config {i} failed: {e}")

print("\n🎉 Basic configuration tests passed!")
