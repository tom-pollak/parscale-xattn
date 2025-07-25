"""Qwen2 model configuration, with support for ParScale"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen2ParScaleConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        parscale_n (`int`, *optional*, defaults to 1):
            Number of parallel replicas for ParScale. When set to 1, behaves as standard Qwen2.
        parscale_n_tokens (`int`, *optional*, defaults to 48):
            Number of prefix tokens for cross-replica communication via prefix tokens.
            Set to 0 to disable learnt prefix tokens and use only cross-attention.
        parscale_attn_smooth (`float`, *optional*, defaults to 0.01):
            Attention smoothing parameter for output aggregation across replicas.
        enable_cross_attn (`bool`, *optional*, defaults to False):
            Whether to enable cross-attention between same-position tokens across replicas.
            This provides data-dependent communication beyond prefix tokens.
        parscale_cross_attn_layers (`list[int]`, *optional*, defaults to None):
            List of layer indices where cross-attention is enabled. If None, applies to all layers
            when enable_cross_attn is True.
        enable_replica_rope (`bool`, *optional*, defaults to False):
            Whether to apply replica-specific RoPE to cross-attention. Each replica gets a different
            rotational position embedding based on its index (0, 1, ..., parscale_n-1).
            Uses existing rope_scaling configuration with parscale_n as max sequence length.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_parscale"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        parscale_n=1,
        parscale_n_tokens=48,
        parscale_attn_smooth=0.01,
        enable_cross_attn=False,
        parscale_cross_attn_layers=None,
        enable_replica_rope=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.parscale_n = parscale_n
        self.parscale_n_tokens = parscale_n_tokens
        self.parscale_attn_smooth = parscale_attn_smooth
        self.enable_cross_attn = enable_cross_attn
        self.parscale_cross_attn_layers = parscale_cross_attn_layers
        self.enable_replica_rope = enable_replica_rope

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self._validate_parscale_config()

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate_parscale_config(self):
        """Validate ParScale-specific configuration parameters."""
        # Basic bounds checking
        if self.parscale_n < 1:
            raise ValueError(f"parscale_n must be >= 1, got {self.parscale_n}")

        if self.parscale_n_tokens < 0:
            raise ValueError(
                f"parscale_n_tokens must be >= 0, got {self.parscale_n_tokens}"
            )

        # Cross-attention validation
        if self.enable_cross_attn and self.parscale_n == 1:
            raise ValueError(
                "Cross-attention (enable_cross_attn=True) requires parscale_n > 1, "
                f"but got parscale_n={self.parscale_n}. Either disable cross-attention "
                "or increase parscale_n."
            )

        # When parscale_n=1, enforce standard Qwen2 behavior
        if self.parscale_n == 1:
            if self.enable_cross_attn:
                raise ValueError(
                    f"Cross-attention should be disabled when parscale_n=1, "
                    f"but enable_cross_attn={self.enable_cross_attn}"
                )
            if self.parscale_n_tokens > 0:
                raise ValueError(
                    f"Prefix tokens should be 0 when parscale_n=1 (standard Qwen2 mode), "
                    f"but parscale_n_tokens={self.parscale_n_tokens}"
                )

        # Cross-attention layers validation
        if self.parscale_cross_attn_layers is not None:
            if not self.enable_cross_attn:
                raise ValueError(
                    "parscale_cross_attn_layers is specified but enable_cross_attn=False. "
                    "Either enable cross-attention or set parscale_cross_attn_layers=None."
                )

            if not isinstance(self.parscale_cross_attn_layers, (list, tuple)):
                raise ValueError(
                    f"parscale_cross_attn_layers must be a list or tuple, "
                    f"got {type(self.parscale_cross_attn_layers)}"
                )

            # Check layer indices are valid
            for layer_idx in self.parscale_cross_attn_layers:
                if not isinstance(layer_idx, int) or layer_idx < 0:
                    raise ValueError(
                        f"All layer indices in parscale_cross_attn_layers must be non-negative integers, "
                        f"got {layer_idx}"
                    )
                if layer_idx >= self.num_hidden_layers:
                    raise ValueError(
                        f"Layer index {layer_idx} in parscale_cross_attn_layers is >= num_hidden_layers "
                        f"({self.num_hidden_layers})"
                    )

        # Replica RoPE validation
        if self.enable_replica_rope and not self.enable_cross_attn:
            raise ValueError(
                "Replica RoPE (enable_replica_rope=True) requires cross-attention to be enabled "
                "(enable_cross_attn=True), but enable_cross_attn=False."
            )

        if self.enable_replica_rope and self.parscale_n == 1:
            raise ValueError(
                "Replica RoPE (enable_replica_rope=True) requires parscale_n > 1, "
                f"but got parscale_n={self.parscale_n}."
            )
