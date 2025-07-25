```
[rank0]: torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <function repeat at 0x7284163e3560>(*(DTensor(local_tensor=FakeTensor(..., device='cuda:0',
size=(1, 2, 48, 128)), device_mesh=DeviceMesh('cuda', [0, 1, 2, 3, 4, 5, 6, 7]), placements=(Shard(dim=0),)), 'n_parscale ... -> (n_parscale b) ...'), **{'b': 4}): got RuntimeError("shape '[2, 2, 48,
128]' is invalid for input of size 49152")

[rank0]: from user code:
[rank0]:    File "/lambda/nfs/nethome-us-east-1/tomp/parscale-xattn/src/parscale_xattn/modeling_qwen2_parscale.py", line 500, in forward
[rank0]:     hidden_states, self_attn_weights = self.self_attn(
[rank0]:   File "/lambda/nfs/nethome-us-east-1/tomp/parscale-xattn/.venv/lib/python3.12/site-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py", line 171, in forward
[rank0]:     return self.checkpoint_fn(  # type: ignore[misc]
[rank0]:   File "/lambda/nfs/nethome-us-east-1/tomp/parscale-xattn/src/parscale_xattn/modeling_qwen2_parscale.py", line 267, in forward
[rank0]:     key_states, value_states = past_key_value.update(
[rank0]:   File "/lambda/nfs/nethome-us-east-1/tomp/parscale-xattn/src/parscale_xattn/modeling_qwen2_parscale.py", line 168, in update
[rank0]:     self.key_cache[layer_idx] = repeat(

[rank0]: Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```

# ParScale Cross-Attention Extension

PARSCALE introduces the third scaling paradigm for scaling LLMs: leverages parallel computation during both training and inference time (Parallel Scaling, or ParScale).

This extension adds **cross-attention across model replicas** to the ParScale paradigm, enabling more flexible data-dependent communication between replicas beyond the existing prefix token mechanism.

## Quick Start

### Installation
```bash
pip install -e .
```

### Training

#### Single GPU
```bash
# Basic ParScale training (parscale_n=1, like standard Qwen2)
CONFIG_FILE=configs/basic.yaml uv run accelerate launch train.py

# ParScale with 4 replicas
CONFIG_FILE=configs/basic.yaml uv run accelerate launch  train.py --parscale.parscale_n=4

# Cross-attention enabled with 4 replicas
CONFIG_FILE=configs/cross_attn.yaml uv run accelerate launch  train.py --parscale.parscale_n=4
```

#### Multi-GPU (8 GPUs with FSDP)
```bash
# Basic ParScale training with 8 GPUs
CONFIG_FILE=configs/basic.yaml torchrun --nproc_per_node=8 train.py \
  parscale.parscale_n=4 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=16

# Cross-attention with 8 GPUs
CONFIG_FILE=configs/cross_attn.yaml torchrun --nproc_per_node=8 train.py \
  parscale.parscale_n=4 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=16

# Larger model (Qwen2-7B) with selective cross-attention
CONFIG_FILE=configs/cross_attn.yaml torchrun --nproc_per_node=8 train.py \
  training.base_model=Qwen/Qwen2-7B \
  parscale.parscale_n=8 \
  parscale.parscale_cross_attn_layers=[0,4,8,12,16,20,24,28] \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=32
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

**Sweep Descriptions:**
- `lr_verification`: Tests learning rates [1e-4, 3e-4, 5e-4, 1e-3] with P=1 and P=4 to verify optimal LR
- `parscale_scaling`: Replicates original paper's P=1,2,4,8 scaling experiments
- `xattn_all_layers`: Same scaling but with cross-attention enabled on all layers
- `xattn_preset_layers`: Same scaling but with cross-attention on layers [0,6,12,18] only

Total: 20 focused runs across all sweeps.

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

**Default Hyperparameters:**
- **Learning Rate**: 3e-4 (Stage 2 from paper)
- **Schedule**: Constant with warmup (WSD-style)
- **Warmup Steps**: 2000 (2K from paper)
- **Max Steps**: 76,294 (~20B tokens as in paper's Stage 2)
- **Batch Size**: 4 per device × 4 gradient accumulation = 16 effective batch size
- **Model**: Qwen2-1.5B (similar to paper's 1.8B model)
- **Checkpointing**: Save only final checkpoint to save disk space

The training approach converts an existing Qwen2 model to ParScale format and continues training with newly initialized ParScale parameters (prefix tokens + optional cross-attention), similar to the paper's two-stage strategy where Stage 2 adds ParScale to an already-trained model.

**Configuration Files:**
- `configs/basic.yaml`: Standard ParScale training
- `configs/cross_attn.yaml`: Cross-attention enabled

### Model Configuration Parameters

- `parscale_n` (int, default: 1): Number of parallel replicas
- `parscale_n_tokens` (int, default: 48): Number of prefix tokens for cross-replica communication
- `enable_cross_attn` (bool, default: False): Enable cross-attention between same-position tokens across replicas
- `parscale_cross_attn_layers` (list[int], default: None): Layer indices where cross-attention is enabled. If None, applies to all layers when cross-attention is enabled

### Direct Model Usage

```python
from parscale_xattn import Qwen2ParScaleForCausalLM, Qwen2ParScaleConfig

# Standard ParScale without cross-attention (original behavior)
config = Qwen2ParScaleConfig(parscale_n=4, enable_cross_attn=False)
model = Qwen2ParScaleForCausalLM(config)

# ParScale with cross-attention on specific layers only
config = Qwen2ParScaleConfig(
    parscale_n=4,
    enable_cross_attn=True,
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

- **Default Behavior**: `enable_cross_attn=False` ensures no behavior change from original ParScale
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
