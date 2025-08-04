"""Cross-replica attention module for ParScale models."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ...configs import ParScaleConfig


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from
    (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class CrossReplicaAttention(nn.Module):
    """
    Cross-replica attention module for ParScale models.

    This module enables attention between same-position tokens across different replicas,
    allowing replicas to coordinate and specialize based on what other replicas are doing.

    Key features:
    - Same-position constraint: token i in replica j can only attend to token i in other replicas
    - Optional replica-specific RoPE embeddings for positional awareness
    - Supports Grouped Query Attention (GQA)
    """

    def __init__(self, config: ParScaleConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.parscale_n = config.parscale_n
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5

        if (self.hidden_size % self.num_heads) != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, but got hidden_size={self.hidden_size} and num_heads={self.num_heads}"
            )

        # Projection layers for cross-replica attention
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        replica_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-replica attention.

        Args:
            hidden_states: Input tensor of shape (p*b, s, d) where:
                - p = parscale_n (number of replicas)
                - b = batch_size
                - s = sequence_length
                - d = hidden_size
            replica_position_embeddings: Optional tuple of (cos, sin) tensors for replica RoPE

        Returns:
            torch.Tensor: Output tensor of shape (p*b, s, d)
        """
        # hidden_states: (p*b, s, d)
        p = self.parscale_n
        b, s, d = (
            hidden_states.size(0) // p,
            hidden_states.size(1),
            hidden_states.size(2),
        )

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for cross-replica attention: (p*b, s, d) -> (b, s, h, p, d_h)
        # This keeps sequence positions separate to prevent information leakage
        q = rearrange(q, "(p b) s (h d_h) -> b s h p d_h", p=p, h=self.num_heads)
        k = rearrange(
            k,
            "(p b) s (h d_h) -> b s h p d_h",
            p=p,
            h=self.num_key_value_heads,
        )
        v = rearrange(
            v,
            "(p b) s (h d_h) -> b s h p d_h",
            p=p,
            h=self.num_key_value_heads,
        )

        # Apply replica-specific RoPE if provided
        if replica_position_embeddings is not None:
            cos, sin = replica_position_embeddings
            # cos/sin are (b, p, head_dim). We need to expand for (b, s, p, head_dim)
            # Expand across sequence length: (b, p, head_dim) -> (b, s, p, head_dim)
            cos = cos.unsqueeze(1).expand(b, s, p, -1)
            sin = sin.unsqueeze(1).expand(b, s, p, -1)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # Repeat K,V for Grouped Query Attention (GQA) with new tensor shape
        # Shape: (b, s, h, p, d_h) where h is num_key_value_heads -> (b, s, num_heads, p, d_h)
        if self.num_key_value_groups > 1:
            k = repeat(k, "b s h p d_h -> b s (g h) p d_h", g=self.num_key_value_groups)
            v = repeat(v, "b s h p d_h -> b s (g h) p d_h", g=self.num_key_value_groups)

        # Apply scaled dot-product attention position by position
        # Shape: (b, s, h, p, d_h) - each position attends to same position across replicas
        # This prevents information leakage since positions are processed separately
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,  # No causal constraint needed within same position across replicas
        )

        # Reshape back to original format: (b, s, h, p, d_h) -> (p*b, s, h*d_h)
        attn_output = rearrange(attn_output, "b s h p d_h -> (p b) s (h d_h)", p=p)

        # Apply output projection
        attn_output = self.o_proj(attn_output)
        return attn_output
