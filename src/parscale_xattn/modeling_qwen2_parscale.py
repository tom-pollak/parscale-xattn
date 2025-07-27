"""
ParScale model implementation with cross-attention extensions.
This module provides the main classes for ParScale models, building on the modular
base implementation and cross-attention extensions.

All ParScale modifications are wrapped within 'parscale_n > 1' conditions.
Cross-attention features are enabled via configuration parameters.
"""

# Import from the new modular structure
from .configuration_qwen2_parscale import Qwen2ParScaleConfig
from .modeling_cross_attn import (
    ParScaleCrossAttnForCausalLM,
    ParScaleCrossAttnModel,
    ParScaleCrossAttnDecoderLayer,
)
from .modeling_base import (
    ParScaleBaseForCausalLM,
    ParScaleBaseModel,
    ParScaleBaseDecoderLayer,
    ParScaleBasePreTrainedModel,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    ParscaleCache,
)

# Re-export key classes for backward compatibility
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

# Alias main classes to maintain backward compatibility
# Users can continue to use these classes as before
Qwen2Model = ParScaleCrossAttnModel
Qwen2ForCausalLM = ParScaleCrossAttnForCausalLM
Qwen2ParScaleForCausalLM = ParScaleCrossAttnForCausalLM
Qwen2DecoderLayer = ParScaleCrossAttnDecoderLayer
Qwen2PreTrainedModel = ParScaleBasePreTrainedModel

# Also re-export everything for easy access
__all__ = [
    # Main model classes (with cross-attention support)
    "Qwen2ParScaleForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2Model",
    "Qwen2DecoderLayer",
    "Qwen2PreTrainedModel",
    # Base classes (without cross-attention)
    "ParScaleBaseForCausalLM",
    "ParScaleBaseModel",
    "ParScaleBaseDecoderLayer",
    "ParScaleBasePreTrainedModel",
    # Cross-attention extension classes
    "ParScaleCrossAttnForCausalLM",
    "ParScaleCrossAttnModel",
    "ParScaleCrossAttnDecoderLayer",
    # Utility classes
    "Qwen2MLP",
    "Qwen2RMSNorm",
    "Qwen2RotaryEmbedding",
    "ParscaleCache",
    "Qwen2ParScaleConfig",
]
