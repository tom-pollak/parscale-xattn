"""Base ParScale configuration classes."""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ParScaleBaseConfig(PretrainedConfig):
    """
    Base configuration class for ParScale models, containing core ParScale parameters
    without cross-attention extensions.

    This configuration implements the original ParScale functionality:
    - Multiple parallel replicas (parscale_n)
    - Prefix token communication (parscale_n_tokens)
    - Output aggregation with attention smoothing

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key_value heads for Grouped Query Attention.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        parscale_n (`int`, *optional*, defaults to 1):
            Number of parallel replicas for ParScale. When set to 1, behaves as standard Qwen2.
        parscale_n_tokens (`int`, *optional*, defaults to 0):
            Number of prefix tokens for cross-replica communication via prefix tokens.
        parscale_attn_smooth (`float`, *optional*, defaults to 0.01):
            Attention smoothing parameter for output aggregation across replicas.
    """

    model_type = "parscale_base"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model
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
        num_key_value_heads=None,
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
        parscale_n_tokens=0,
        parscale_attn_smooth=0.01,
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

        # ParScale base parameters
        self.parscale_n = parscale_n
        self.parscale_n_tokens = parscale_n_tokens
        self.parscale_attn_smooth = parscale_attn_smooth

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

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

        self._validate_parscale_base_config()

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate_parscale_base_config(self):
        """Validate base ParScale configuration parameters."""
        # Basic bounds checking
        if self.parscale_n < 1:
            raise ValueError(f"parscale_n must be >= 1, got {self.parscale_n}")

        if self.parscale_n_tokens < 0:
            raise ValueError(
                f"parscale_n_tokens must be >= 0, got {self.parscale_n_tokens}"
            )
