"""Cross-attention configuration extension for ParScale models."""

from .base_config import ParScaleBaseConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ParScaleConfig(ParScaleBaseConfig):
    """
    Configuration class for ParScale models with cross-attention extensions.

    This extends the base ParScale configuration with cross-replica attention capabilities:
    - Cross-attention between same-position tokens across replicas
    - Layer-specific cross-attention configuration
    - Replica-specific RoPE embeddings for cross-attention

    Args:
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
    """

    model_type = "parscale_cross_attn"

    def __init__(
        self,
        enable_cross_attn=False,
        enable_xkv_attn=False,
        parscale_cross_attn_layers=None,
        enable_replica_rope=False,
        **kwargs,
    ):
        # Cross-attention parameters
        self.enable_cross_attn = enable_cross_attn
        self.enable_xkv_attn = enable_xkv_attn
        self.parscale_cross_attn_layers = parscale_cross_attn_layers
        self.enable_replica_rope = enable_replica_rope

        # Initialize base config first
        super().__init__(**kwargs)

        # Validate cross-attention specific parameters
        self._validate_cross_attn_config()

    def _validate_cross_attn_config(self):
        """Validate cross-attention specific configuration parameters."""
        # Cross Attn and XKV attn cannot be enabled at the same time
        if self.enable_cross_attn and self.enable_xkv_attn:
            raise ValueError(
                "CrossAttn and XKV attn cannot be enabled at the same time."
            )

        # When parscale_n=1, no cross-attention features should be enabled
        if self.parscale_n == 1:
            if self.enable_replica_rope:
                raise ValueError(
                    "Replica RoPE (enable_replica_rope=True) requires parscale_n > 1, but got parscale_n=1."
                )
            if self.enable_cross_attn:
                raise ValueError(
                    "Cross-attention (enable_cross_attn=True) requires parscale_n > 1, but got parscale_n=1."
                )

        # Replica RoPE validation
        if self.enable_replica_rope:
            if not self.enable_cross_attn:
                raise ValueError(
                    "Replica RoPE (enable_replica_rope=True) requires cross-attention to be enabled "
                    "(enable_cross_attn=True), but enable_cross_attn=False."
                )
            if self.parscale_n <= 1:
                raise ValueError(
                    f"Replica RoPE (enable_replica_rope=True) requires parscale_n > 1, "
                    f"but got parscale_n={self.parscale_n}."
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
