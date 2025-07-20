
# ParScale Cross-Attention Extension

PARSCALE introduces the third scaling paradigm for scaling LLMs: leverages parallel computation during both training and inference time (Parallel Scaling, or ParScale).

This extension adds **cross-attention across model replicas** to the ParScale paradigm, enabling more flexible data-dependent communication between replicas beyond the existing prefix token mechanism.

## Quick Start

### Installation
```bash
pip install -e .
```

### Training
```bash
# Basic ParScale training (parscale_n=1, like standard Qwen2)
uv run python train.py --config-path=configs --config-name=basic

# ParScale with 4 replicas
uv run python train.py --config-path=configs --config-name=basic parscale.parscale_n=4

# Cross-attention enabled with 4 replicas
uv run python train.py --config-path=configs --config-name=cross_attn parscale.parscale_n=4

# Cross-attention on specific layers only
uv run python train.py --config-path=configs --config-name=cross_attn \
  parscale.parscale_n=4 \
  parscale.parscale_cross_attn_layers=[0,4,8,12]

# Custom output directory
uv run python train.py --config-path=configs --config-name=basic \
  training.output_dir=./my-model \
  parscale.parscale_n=2
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

The training script uses YAML configs with OmegaConf CLI overrides. See `configs/` for examples:

- `configs/basic.yaml`: Standard ParScale training 
- `configs/cross_attn.yaml`: Cross-attention enabled

### Model Configuration Parameters

- `parscale_n` (int, default: 1): Number of parallel replicas
- `parscale_n_tokens` (int, default: 48): Number of prefix tokens for cross-replica communication  
- `parscale_enable_cross_attn` (bool, default: False): Enable cross-attention between same-position tokens across replicas
- `parscale_cross_attn_layers` (list[int], default: None): Layer indices where cross-attention is enabled. If None, applies to all layers when cross-attention is enabled

### Direct Model Usage

```python
from parscale_xattn import Qwen2ParScaleForCausalLM, Qwen2ParScaleConfig

# Standard ParScale without cross-attention (original behavior)
config = Qwen2ParScaleConfig(parscale_n=4, parscale_enable_cross_attn=False)
model = Qwen2ParScaleForCausalLM(config)

# ParScale with cross-attention on specific layers only  
config = Qwen2ParScaleConfig(
    parscale_n=4,
    parscale_enable_cross_attn=True,
    parscale_cross_attn_layers=[0, 4, 8, 12]
)
model = Qwen2ParScaleForCausalLM(config)
```

## Implementation Details

### Attention Mechanism Changes

When cross-attention is enabled:

1. **Key/Value Expansion**: Keys and values from all replicas are concatenated for each sequence position
2. **Cross-Replica Queries**: Each replica's queries can attend to keys/values from all replicas at the same position
3. **Output Projection**: Specialized projection layer handles the expanded attention output dimensions

### Backward Compatibility

- **Default Behavior**: `parscale_enable_cross_attn=False` ensures no behavior change from original ParScale
- **Existing Models**: All existing ParScale functionality is preserved
- **Gradual Adoption**: Can be enabled on specific layers for controlled experimentation

## Comparison: Prefix Tokens vs Cross-Attention

| Aspect | Prefix Tokens | Cross-Attention |
|--------|---------------|-----------------|
| **Communication** | Fixed learnable tokens | Data-dependent attention |
| **Flexibility** | Static cross-replica information | Dynamic based on input content |
| **Parameters** | `parscale_n_tokens` learnable tokens per layer | Linear projection layers |
| **Computational Cost** | Low (fixed tokens) | Higher (expanded attention) |
| **Use Case** | General cross-replica communication | Content-specific information exchange |

## When to Use Cross-Attention

**Enable cross-attention when:**
- Input sequences have strong positional dependencies
- Different replicas need to share position-specific information
- Dynamic communication patterns are more important than efficiency

**Stick with prefix tokens when:**
- Computational efficiency is critical
- General cross-replica communication is sufficient
- Working with very long sequences where cross-attention becomes expensive

## Performance Considerations

- **Memory Usage**: Cross-attention increases memory usage by approximately `parscale_n` factor for attention computations
- **Computation**: Additional linear projections for handling expanded attention outputs
- **Layer Selection**: Use `parscale_cross_attn_layers` to enable cross-attention only where most beneficial

## Example Use Cases

1. **Multi-Document QA**: Different replicas process different documents, cross-attention enables position-wise information sharing
2. **Code Generation**: Different replicas handle different aspects (syntax, semantics), position-wise communication improves coherence
3. **Long Sequence Processing**: Cross-attention at specific positions can help maintain global context across replicas
