"""
Cross-attention model extensions for ParScale models.
Builds on top of the base ParScale implementation to add cross-replica attention capabilities.
"""

import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack

from ..configs import ParScaleConfig
from .cross_replica_attn import CrossReplicaAttention
from .base_model import (
    ParScaleBaseModel,
    ParScaleBaseDecoderLayer,
    ParScaleBaseForCausalLM,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)


class ParScaleCrossAttnDecoderLayer(ParScaleBaseDecoderLayer):
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
            self.cross_replica_layernorm = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.cross_replica_attn = CrossReplicaAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        replica_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # Apply cross-replica attention first if enabled
        if self.enable_cross_attn:
            residual = hidden_states
            hidden_states = self.cross_replica_layernorm(hidden_states)
            hidden_states = self.cross_replica_attn(
                hidden_states, replica_position_embeddings
            )
            hidden_states = residual + hidden_states

        # Call parent's forward method for standard self-attention and MLP
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )


class ParScaleModel(ParScaleBaseModel):
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
                ParScaleCrossAttnDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Add replica rotary embedding if enabled
        if self.config.enable_replica_rope:
            # Create a copy of config with parscale_n as max_position_embeddings for replica positioning
            replica_rope_config = copy.deepcopy(config)
            replica_rope_config.max_position_embeddings = config.parscale_n
            self.replica_rotary_emb = Qwen2RotaryEmbedding(config=replica_rope_config)

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
        # create replica-specific position embeddings for cross-attention if enabled
        if self.config.enable_cross_attn and self.config.enable_replica_rope:
            batch_size = inputs_embeds.size(0) // self.config.parscale_n
            device = inputs_embeds.device
            head_dim = self.config.hidden_size // self.config.num_attention_heads

            # Create replica position IDs: each replica gets its replica_idx as position
            replica_position_ids = (
                torch.arange(
                    self.config.parscale_n,
                    device=device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            # Generate RoPE embeddings for replica positions
            # Create a minimal tensor just for head_dim calculation
            dummy_tensor = torch.empty(
                batch_size,
                self.config.parscale_n,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            replica_position_embeddings = self.replica_rotary_emb(
                dummy_tensor, replica_position_ids
            )
        else:
            replica_position_embeddings = None

        return super().forward(
            input_ids=input_ids,
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


class ParScaleForCausalLM(ParScaleBaseForCausalLM):
    """
    ParScale model with cross-replica attention for causal language modeling.

    Extends the base ParScale causal LM to use the cross-attention enabled model.
    """

    def __init__(self, config):
        # Same as super, except different model
        super().__init__(config)
        self.model = ParScaleModel(config)
        self.post_init()

    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        Cross-attention layers are initialized here, while other layers
        are initialized by the parent class.
        """
        std = self.config.initializer_range
        if isinstance(module, CrossReplicaAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=std)
            if module.q_proj.bias is not None:
                module.q_proj.bias.data.zero_()
            module.k_proj.weight.data.normal_(mean=0.0, std=std)
            if module.k_proj.bias is not None:
                module.k_proj.bias.data.zero_()
            module.v_proj.weight.data.normal_(mean=0.0, std=std)
            if module.v_proj.bias is not None:
                module.v_proj.bias.data.zero_()
            module.o_proj.weight.data.normal_(mean=0.0, std=std)
            if module.o_proj.bias is not None:
                module.o_proj.bias.data.zero_()
        else:
            super()._init_weights(module)
