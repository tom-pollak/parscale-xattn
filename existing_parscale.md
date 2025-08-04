# ParScale Research Specification

## Background

There was initial research by Qwen lab who created a technique called PARSCALE. The idea is fairly simple: When running inference with low batch sizes, we are often memory constrained, not compute constrained (e.g. batch size 1). We can increase the batch size to e.g. 8 without losing performance since we are not bottlenecked on compute and can reuse the weights.

How can we use the increase in batch size to improve our model? They introduce a new dimension to the scaling -- on the batch dimension. Side tangent: they also argue that CFG (classifier free guidance) works so well is that it essentially allows you to double the amount of FLOPs we give to each prompt.

## Core Implementation

The way they did this in LLMs was to run the same model N times (called replicas) with different learnt prefixes, then aggregate the results from each of the replicas in a final MLP.

### Key Configuration Parameters
- `parscale_n`: Number of parallel replicas (default: 1)
- `parscale_n_tokens`: Number of prefix tokens for cross-replica communication (default: 48)

### Input Replication
```python
# Input transformation: we directly copy the input for n_parscale times.
# The transformation is implemented through KVCache (ParscaleCache).
inputs_embeds = repeat(
    inputs_embeds, "b s h -> (n_parscale b) s h", n_parscale=self.parscale_n
)
```

### Prefix Token System
Each attention layer learns prefix key-value pairs for each replica:

```python
if config.parscale_n > 1:
    self.prefix_k = nn.Parameter(
        torch.empty((
            config.parscale_n,
            config.num_key_value_heads,
            config.parscale_n_tokens,
            self.head_dim,
        ))
    )
    self.prefix_v = nn.Parameter(
        torch.empty((
            config.parscale_n,
            config.num_key_value_heads,
            config.parscale_n_tokens,
            self.head_dim,
        ))
    )
```

The trained prefix is saved in `layer.self_attn.prefix_k` / `layer.self_attn.prefix_v`. They extract them to construct `ParscaleCache`.

### Attention Mask Handling
```python
# Expand attention mask to contain the prefix tokens
n_virtual_tokens = self.config.parscale_n_tokens

if attention_mask is not None:
    attention_mask = torch.cat([
        torch.zeros((
            attention_mask.shape[0],
            attention_mask.shape[1],
            attention_mask.shape[2],
            self.config.parscale_n_tokens,
        ), dtype=attention_mask.dtype, device=attention_mask.device),
        attention_mask,
    ], dim=3)
```

### Output Aggregation
The final step is output aggregation based on dynamic weighted sum:

```python
# output aggregation, based on dynamic weighted sum.
attn = torch.unsqueeze(
    torch.softmax(
        self.aggregate_layer(
            rearrange(
                hidden_states,
                "(n_parscale b) s h -> b s (h n_parscale)",
                n_parscale=self.parscale_n,
            )
        ).float(),
        dim=-1,
    ),
    dim=-1,
)  # [b s n_parscale 1]
```

The aggregation layer is a simple MLP:
```python
self.aggregate_layer = torch.nn.Sequential(
    torch.nn.Linear(config.parscale_n * config.hidden_size, config.hidden_size),
    torch.nn.SiLU(),
    torch.nn.Linear(config.hidden_size, config.parscale_n),
)
```

### Attention Smoothing
Optional smoothing prevents over-concentration on single replicas:
```python
if self.parscale_aggregate_attn_smoothing != 0.0:
    attn = attn * (1 - self.parscale_aggregate_attn_smoothing) + (
        self.parscale_aggregate_attn_smoothing / self.parscale_n
    )
```

Final weighted sum:
```python
hidden_states = torch.sum(
    rearrange(
        hidden_states,
        "(n_parscale b) s h -> b s n_parscale h",
        n_parscale=self.parscale_n,
    ) * attn,
    dim=2,
    keepdim=False,
).to(hidden_states.dtype)
```


## Key Implementation Details

- All modifications are wrapped within `if config.parscale_n > 1` for backward compatibility
- When `parscale_n = 1`, behaves like standard Qwen2
- Uses shared weights across replicas - diversity comes from learned prefix tokens
- Custom `ParscaleCache` handles prefix token management during generation