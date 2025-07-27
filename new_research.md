# ParScale Cross-Attention Research Direction

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

### Benefits of Replica RoPE
- More principled way of encoding replica identity vs learned prefixes
- Leverages the proven RoPE mechanism for positional understanding
- Could enable better generalization to different numbers of replicas
- Cleaner separation between sequence position (standard RoPE) and replica position (replica RoPE)

## Research Questions

1. **Layer Selection**: Which layers benefit most from cross-replica attention? Early, middle, or late layers?

2. **Attention Pattern**: Should it be full attention across replicas or some structured pattern?

3. **RoPE Integration**: How does replica RoPE interact with standard sequence RoPE? Additive or separate?

4. **Training Strategy**: How to initialize and train the cross-attention components?

5. **Computational Cost**: What's the overhead vs. benefit trade-off?

## Expected Architecture

```
Input → Replicate → 
Layer 1 → Cross-Attn (if enabled) → 
Layer 2 → 
Layer 3 → Cross-Attn (if enabled) →
...
Final Layer → Aggregate → Output
```

Where cross-attention layers enable same-position communication between replicas before continuing with standard transformer processing.