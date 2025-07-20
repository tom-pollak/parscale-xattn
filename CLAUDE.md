# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ParScale-XAttn extension project that implements cross-attention across model replicas for the ParScale paradigm. ParScale is the third scaling paradigm for Large Language Models that leverages parallel computation during both training and inference time.

## Architecture

### Core Components

- **Configuration**: `Qwen2ParScaleConfig` extends standard Qwen2 configuration with ParScale-specific parameters:
  - `parscale_n`: Number of parallel replicas (default: 1)
  - `parscale_n_tokens`: Number of prefix tokens for cross-replica communication (default: 48)
  - `parscale_attn_smooth`: Attention smoothing parameter (default: 0.01)

- **Model Classes**:
  - `Qwen2ParScaleForCausalLM`: Main causal language model with ParScale support
  - `Qwen2Model`: Core transformer model with replica aggregation
  - `Qwen2Attention`: Attention mechanism with prefix tokens for cross-replica communication
  - `ParscaleCache`: Custom cache implementation handling prefix tokens across replicas

### ParScale Implementation Details

When `parscale_n > 1`, the model operates with multiple parallel replicas:

1. **Input Replication**: Input embeddings are replicated across `parscale_n` replicas
2. **Prefix Tokens**: Each attention layer uses learnable prefix tokens (`prefix_k`, `prefix_v`) stored as model parameters
3. **Cross-Replica Communication**: Attention mechanisms can attend to prefix tokens from all replicas
4. **Output Aggregation**: A learned aggregation layer dynamically weights outputs from different replicas

## Development Commands

### Environment Setup
```bash
# Use uv for Python environment management
uv run python <script.py>
uv run pytest <test_file>
```

### Code Formatting
```bash
# Format code using ruff
ruff format
```

### Model Usage
```python
from parscale_xattn import Qwen2ParScaleForCausalLM, Qwen2ParScaleConfig

# Standard Qwen2 usage (parscale_n=1)
config = Qwen2ParScaleConfig()
model = Qwen2ParScaleForCausalLM(config)

# ParScale with cross-attention (parscale_n>1)
config = Qwen2ParScaleConfig(parscale_n=4, parscale_n_tokens=48)
model = Qwen2ParScaleForCausalLM(config)
```

## Key Implementation Notes

- All ParScale modifications are conditionally wrapped with `if config.parscale_n > 1`
- The model maintains backward compatibility with standard Qwen2 when `parscale_n=1`
- Cross-replica attention is implemented through prefix tokens rather than explicit cross-attention layers
- The aggregation mechanism uses learned attention weights with optional smoothing

## File Structure

- `src/parscale_xattn/configuration_qwen2_parscale.py`: Model configuration with ParScale parameters
- `src/parscale_xattn/modeling_qwen2_parscale.py`: Core model implementation with ParScale extensions
- `src/parscale_xattn/__init__.py`: Package exports and public API