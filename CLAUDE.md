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

## Development Commands

### Environment Setup
```bash
# Use uv for Python environment management
uv run python <script.py>
```

### Code Formatting
```bash
# Format code using ruff
ruff format
```

### Training Commands
```bash
# Single GPU training - Basic ParScale (parscale_n=1, like standard Qwen2)
uv run python train.py --config-path=configs --config-name=basic

# ParScale with 4 replicas
uv run python train.py --config-path=configs --config-name=basic parscale.parscale_n=4

# Cross-attention enabled with 4 replicas
uv run python train.py --config-path=configs --config-name=cross_attn parscale.parscale_n=4

# Multi-GPU training (8 GPUs with FSDP)
CONFIG_FILE=configs/basic.yaml torchrun --nproc_per_node=8 train.py \
  parscale.parscale_n=4 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=16
```

### Hyperparameter Sweeps
```bash
# Create and run wandb sweeps for systematic experiments
python wandb_sweep.py create lr_verification
wandb agent <sweep_id>

# Available sweep types: lr_verification, parscale_scaling, xattn_all_layers, xattn_preset_layers
```

### Model Usage
```python
from parscale_xattn import Qwen2ParScaleForCausalLM, Qwen2ParScaleConfig

# Standard usage (parscale_n=1 behaves like Qwen2)
config = Qwen2ParScaleConfig()
model = Qwen2ParScaleForCausalLM(config)

# ParScale with cross-attention on specific layers
config = Qwen2ParScaleConfig(
    parscale_n=4, 
    enable_cross_attn=True,
    parscale_cross_attn_layers=[0, 4, 8, 12]
)
model = Qwen2ParScaleForCausalLM(config)
```

## Key Implementation Notes

- All ParScale modifications are conditionally wrapped with `if config.parscale_n > 1`
- Maintains backward compatibility with standard Qwen2 when `parscale_n=1`
- Two communication mechanisms: prefix tokens (always enabled) + optional cross-attention
- Training follows two-stage approach: convert existing Qwen2 → continue training with ParScale parameters