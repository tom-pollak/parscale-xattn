import torch
from transformers import AutoModelForCausalLM


from parscale_xattn.configuration_qwen2_parscale import Qwen2ParScaleConfig
from parscale_xattn.modeling_qwen2_parscale import Qwen2ParScaleForCausalLM


def convert_qwen2_to_parscale(
    base_model_name: str,
    config: Qwen2ParScaleConfig,
) -> Qwen2ParScaleForCausalLM:
    """Convert Qwen2 model to ParScale."""
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )
    parscale_model = Qwen2ParScaleForCausalLM(config).to(torch.bfloat16)
    parscale_model.load_state_dict(base_model.state_dict(), strict=False)

    # Initialize the new prefix parameters if present
    if config.parscale_n_tokens > 0:
        for layer in parscale_model.model.layers:
            if hasattr(layer.self_attn, "prefix_k"):
                torch.nn.init.normal_(
                    layer.self_attn.prefix_k, std=config.initializer_range
                )
            if hasattr(layer.self_attn, "prefix_v"):
                torch.nn.init.normal_(
                    layer.self_attn.prefix_v, std=config.initializer_range
                )

    if config.enable_cross_attn:
        for layer in parscale_model.model.layers:
            if layer.enable_cross_attn:
                torch.nn.init.normal_(
                    layer.cross_replica_attn.q_proj.weight, std=config.initializer_range
                )
                torch.nn.init.normal_(
                    layer.cross_replica_attn.k_proj.weight, std=config.initializer_range
                )
                torch.nn.init.normal_(
                    layer.cross_replica_attn.v_proj.weight, std=config.initializer_range
                )
                torch.nn.init.normal_(
                    layer.cross_replica_attn.o_proj.weight, std=config.initializer_range
                )

    return parscale_model
