# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ParScale-XAttn extension project that implements cross-attention across model replicas for the ParScale paradigm. ParScale is the third scaling paradigm for Large Language Models that leverages parallel computation during both training and inference time.

## Architecture

### Key Configuration Parameters

- `parscale_n`: Number of parallel replicas (default: 1)
- `parscale_n_tokens`: Number of prefix tokens for cross-replica communication (default: 48)
- `enable_cross_attn`: Enable cross-attention between same-position tokens across replicas (default: False)
- `parscale_cross_attn_layers`: List of layer indices where cross-attention is enabled (default: None)

### ParScale Overview

ParScale operates with multiple parallel replicas when `parscale_n > 1`. This extension adds cross-attention between same-position tokens across replicas, complementing the existing prefix token communication mechanism.

### Model Usage

```python
from parscale_xattn import ParScaleForCausalLM, ParScaleConfig

# Standard usage (parscale_n=1 behaves like Qwen2)
config = ParScaleConfig()
model = ParScaleForCausalLM(config)

# ParScale with cross-attention on specific layers
config = ParScaleConfig(
    parscale_n=4,
    enable_cross_attn=True,
    parscale_cross_attn_layers=[0, 4, 8, 12]
)
model = ParScaleForCausalLM(config)
```
