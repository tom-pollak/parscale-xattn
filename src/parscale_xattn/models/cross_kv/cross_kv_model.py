from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from einops import repeat

from .cross_kv_attn import CrossKVAttention
from ...configs import ParScaleConfig
from ..base_model import (
    ParScaleBaseModel,
    ParScaleBaseDecoderLayer,
    ParScaleBaseForCausalLM,
)


class ParScaleCrossKVDecoderLayer(ParScaleBaseDecoderLayer):
    """
    ParScale decoder layer with cross-replica attention capabilities.

    Inherits from the base ParScale decoder layer and adds:
    - Cross-replica attention for same-position token communication
    - Layer normalization for cross-attention
    - Conditional cross-attention based on configuration
    """

    def __init__(self, config: ParScaleConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        # Add cross-attention if enabled for this layer
        self.enable_cross_attn = config.enable_cross_attn and (
            self.config.parscale_cross_attn_layers is None
            or self.layer_idx in self.config.parscale_cross_attn_layers
        )

        if self.enable_cross_attn:
            self.self_attn = CrossKVAttention(config)


class ParScaleCrossKVModel(ParScaleBaseModel):
    """
    ParScale model with cross-replica attention capabilities.

    Extends the base ParScale model to support:
    - Cross-attention between same-position tokens across replicas
    - Replica-specific RoPE embeddings for cross-attention
    - Layer-specific cross-attention configuration
    """

    def __init__(self, config: ParScaleConfig):
        # Initialize with base config, but we'll override the layers
        super().__init__(config)

        # Replace layers with cross-attention enabled layers
        self.layers = nn.ModuleList(
            [
                ParScaleCrossKVDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # Re-initialize weights to ensure cross-attention layers are initialized
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # type: ignore
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        inputs_embeds = inputs_embeds or self.embed_tokens(input_ids)
        if self.config.enable_cross_attn and self.config.enable_replica_rope:
            p = self.parscale_n
            b, s, d = (
                inputs_embeds.size(0) // p,
                inputs_embeds.size(1),
                inputs_embeds.size(2),
            )
            cos, sin = self.rotarty_embedding(inputs_embeds, torch.arange(s))  # s
            cos = repeat("s -> (p s) 1", cos, p=p)
            sin = repeat("s -> (p s) 1", sin, p=p)
            replica_position_embeddings = (cos, sin)
        else:
            replica_position_embeddings = None

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            # will be passed to decoder layers as kwargs
            replica_position_embeddings=replica_position_embeddings,  # type: ignore
            **flash_attn_kwargs,
        )


class ParScaleCrossKVForCausalLM(ParScaleBaseForCausalLM):
    """
    ParScale model with cross-replica attention for causal language modeling.

    Extends the base ParScale causal LM to use the cross-attention enabled model.
    """

    def __init__(self, config):
        # Same as super, except different model
        super().__init__(config)
        self.model = ParScaleCrossKVModel(config)
        self.post_init()
