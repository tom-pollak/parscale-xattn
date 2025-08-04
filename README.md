# ParScale Cross-Attention Extension

PARSCALE introduces the third scaling paradigm for scaling LLMs: leverages parallel computation during both training and inference time (Parallel Scaling, or ParScale).

This extension adds **cross-replica attention** to the ParScale paradigm, enabling more flexible data-dependent communication between replicas beyond the existing prefix token mechanism.

## Quick Start

### Training

```bash
# Basic ParScale training with 8 GPUs
CONFIG_FILE=configs/parscale.yaml uv run accelerate launch train.py

# Cross replica with 8 GPUs
CONFIG_FILE=configs/cross_attn.yaml uv run accelerate launch train.py
```

### Hyperparameter Sweeps with Wandb

For systematic experimentation, use the wandb sweep script to replicate original paper results:

```bash
# 1. Learning rate verification (P=1,4 × 4 learning rates = 8 runs)
python wandb_sweep.py create lr_verification
wandb agent <sweep_id>

# 2. Original paper replication: P=1,2,4,8 with fixed LR (4 runs)
python wandb_sweep.py create parscale_scaling
wandb agent <sweep_id>

# 3. Cross-attention on all layers with P=1,2,4,8 (4 runs)
python wandb_sweep.py create xattn_all_layers
wandb agent <sweep_id>

# 4. Cross-attention on preset layers [0,6,12,18] with P=1,2,4,8 (4 runs)
python wandb_sweep.py create xattn_preset_layers
wandb agent <sweep_id>
```

## Overview

The original ParScale implementation uses:

- **Input Replication**: Input embeddings replicated across `parscale_n` replicas
- **Prefix Tokens**: Learnable prefix tokens for cross-replica communication
- **Output Aggregation**: Learned attention-based aggregation of replica outputs

This extension adds:

- **Cross-Replica Attention**: Same-position tokens across replicas can directly attend to each other
- **Data-Dependent Communication**: Unlike fixed prefix tokens, cross-attention provides adaptive information exchange
- **Configurable Layers**: Option to enable cross-attention on specific layers only

## Cross-Attention Mechanism

When enabled, the cross-attention mechanism works as follows:

1. **Token Alignment**: The first token in replica 1 can attend to the first token in all other replicas
2. **Position-Wise Communication**: Each sequence position enables communication across all replicas
3. **Maintaining Causality**: Causal masking is preserved within each replica while enabling cross-replica attention
4. **Complementary to Prefix Tokens**: Works alongside the existing prefix token mechanism

## Configuration

### Training Configuration

The training script follows the original ParScale paper's hyperparameters for continual pre-training (Stage 2):

- `configs/basic.yaml`: Standard ParScale training
- `configs/cross_attn.yaml`: Cross-replica enabled

### Model Configuration Parameters

- `parscale_n` (int, default: 1): Number of parallel replicas
- `parscale_n_tokens` (int, default: 48): Number of prefix tokens for cross-replica communication
- `enable_cross_attn` (bool, default: False): Enable cross-attention between same-position tokens across replicas
- `parscale_cross_attn_layers` (list[int], default: None): Layer indices where cross-attention is enabled. If None, applies to all layers when cross-attention is enabled

### Direct Model Usage

```python
from parscale_xattn import ParScaleForCausalLM, ParScaleConfig
config = ParScaleConfig(parscale_n=4, enable_cross_attn=True)
model = ParScaleForCausalLM(config)
```

## Implementation Details

### Attention Mechanism Changes

When cross-attention is enabled:

1. **Key/Value Expansion**: Keys and values from all replicas are concatenated for each sequence position
2. **Cross-Replica Queries**: Each replica's queries can attend to keys/values from all replicas at the same position
3. **Output Projection**: Specialized projection layer handles the expanded attention output dimensions

# ParScale Cross-Replica Research Direction

## Problem Statement

Currently in ParScale, the only way replicas know what to do is based on the initial learnt prefix. They have no way of communicating with each other during the forward pass, which seems like a waste. Each replica processes independently and only gets aggregated at the very end.

## Proposed Solution: Cross-Replica Attention

Add a cross-attention layer interspersed throughout the ParScale models that works **between the replicas**. In this layer, each token can talk to the tokens from other replicas with the same **sequence position** as itself.

### Core Mechanism

- Token 12 from replica 1 can attend to all token 12s from other replicas
- Token 5 from replica 3 can attend to all token 5s from other replicas
- No cross-position communication (token 12 cannot attend to token 5)

This same-position constraint maintains the causal structure while enabling replica coordination.

### Expected Benefits

- Each replica can specialize based on what other replicas are doing
- Better coordination and division of labor between replicas
- More sophisticated communication than just learned prefix tokens

## Implementation Details

### Cross-Attention Layer Placement

```python
# New config parameters
enable_cross_attn: bool = False
parscale_cross_attn_layers: list[int] = None  # Which layers get cross-attention
```

### Attention Mechanism

For each sequence position `i`, gather hidden states from all replicas:

```python
# Shape: (parscale_n, batch_size, hidden_size)
cross_replica_states = rearrange(
    hidden_states[:, i, :],
    "(n_parscale b) h -> n_parscale b h",
    n_parscale=self.parscale_n
)
```

Then apply attention across the replica dimension while keeping batch and position separate.

## Extension: RoPE for Replica Positioning

### Motivation

Currently, replicas only know their identity through learned prefix tokens. We can add RoPE to the cross-replica attention based on the current **replica ID** (which acts as a position).

### Implementation

```python
# Replica positions: [0, 1, 2, ..., parscale_n-1]
replica_positions = torch.arange(self.parscale_n, device=device)

# Apply RoPE to cross-replica attention
cos_replica, sin_replica = self.replica_rotary_emb(hidden_states, replica_positions)
q_replica, k_replica = apply_rotary_pos_emb(q_cross, k_cross, cos_replica, sin_replica)
```
