"""
Cross-attention model extensions for ParScale models.
Builds on top of the base ParScale implementation to add cross-replica attention capabilities.
"""

import copy
from typing import Optional, Tuple, Union, List
import warnings

import torch
from torch import nn
from einops import repeat, rearrange
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack

from ..configs import ParScaleConfig
from .layers.cross_replica_attn import CrossReplicaAttention
from .base_model import (
    ParscaleCache,
    ParScaleBaseModel,
    ParScaleBaseDecoderLayer,
    ParScaleBaseForCausalLM,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    KwargsForCausalLM,
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

    @property
    def use_prefix_cache(self) -> bool:
        """Whether to use prefix cache based on parscale configuration."""
        return self.config.parscale_n > 1 and self.config.parscale_n_tokens > 0

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
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warn(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.parscale_n > 1:
            # Input transformation: we directly copy the input for n_parscale times.
            inputs_embeds = repeat(
                inputs_embeds, "b s h -> (n_parscale b) s h", n_parscale=self.parscale_n
            )
            if attention_mask is not None:
                attention_mask = repeat(
                    attention_mask,
                    "b s -> (n_parscale b) s",
                    n_parscale=self.parscale_n,
                )
            if position_ids is not None:
                position_ids = repeat(
                    position_ids, "b s -> (n_parscale b) s", n_parscale=self.parscale_n
                )

        # ParScale cache initialization
        if self.use_prefix_cache:
            if use_cache and (
                past_key_values is None or past_key_values.get_seq_length() == 0
            ):
                past_key_values = ParscaleCache(
                    [layer.self_attn.prefix_k for layer in self.layers],
                    [layer.self_attn.prefix_v for layer in self.layers],
                )

        # Standard Hugging Face cache initialization for non-parscale cases
        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # create replica-specific position embeddings for cross-attention if enabled
        replica_position_embeddings = None
        if self.config.enable_cross_attn and self.config.enable_replica_rope:
            batch_size = hidden_states.size(0) // self.config.parscale_n
            # Create replica position IDs: each replica gets its replica_idx as position
            replica_position_ids = (
                torch.arange(
                    self.config.parscale_n,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            # Generate RoPE embeddings for replica positions
            # Create a minimal tensor just for head_dim calculation
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            dummy_tensor = torch.empty(
                batch_size,
                self.config.parscale_n,
                head_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            replica_position_embeddings = self.replica_rotary_emb(
                dummy_tensor, replica_position_ids
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    replica_position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    replica_position_embeddings=replica_position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if self.parscale_n > 1:
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
            if self.parscale_aggregate_attn_smoothing != 0.0:
                attn = attn * (1 - self.parscale_aggregate_attn_smoothing) + (
                    self.parscale_aggregate_attn_smoothing / self.parscale_n
                )
            hidden_states = torch.sum(
                rearrange(
                    hidden_states,
                    "(n_parscale b) s h -> b s n_parscale h",
                    n_parscale=self.parscale_n,
                )
                * attn,
                dim=2,
                keepdim=False,
            ).to(hidden_states.dtype)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class ParScaleForCausalLM(ParScaleBaseForCausalLM):
    """
    ParScale model with cross-replica attention for causal language modeling.

    Extends the base ParScale causal LM to use the cross-attention enabled model.
    """

    def __init__(self, config):
        # Initialize with base class, but replace the model
        super().__init__(config)
        self.model = ParScaleModel(config)

        # Re-initialize weights since we replaced the model
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
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
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
