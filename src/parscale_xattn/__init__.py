from .configuration_qwen2_parscale import Qwen2ParScaleConfig
from .modeling_qwen2_parscale import (
    Qwen2Model,
    Qwen2ParScaleForCausalLM,
    Qwen2PreTrainedModel,
)
from .modeling_base import KwargsForCausalLM

from .convert import convert_qwen2_to_parscale
