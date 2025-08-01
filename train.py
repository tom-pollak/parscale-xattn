import os
from dataclasses import asdict, field
from typing import Optional

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from datasets import load_dataset
from omegaconf import OmegaConf, SCMode
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from parscale_xattn import (
    Qwen2ParScaleForCausalLM,
    Qwen2ParScaleConfig,
)


DEBUG_CONFIG = Qwen2ParScaleConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=12,
    num_attention_heads=32,
    num_key_value_heads=16,
)


@dataclass
class ParScaleConfig:
    parscale_n: int = 1
    parscale_n_tokens: int = 48
    enable_cross_attn: bool = False
    parscale_cross_attn_layers: Optional[list[int]] = None
    enable_replica_rope: bool = False
    use_cache: bool = False


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2-0.5B"
    dataset: str = "pajama"
    output_dir: str = "./models/"
    max_length: int = 2048
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 76294  # 20B tokens ÷ (4×4×2048×8) ≈ 76k steps from paper
    learning_rate: float = 3e-4  # Stage 2 learning rate from paper
    warmup_steps: int = 2000  # 2K warm-up from paper
    save_steps: int = 76294  # Save only at the end
    save_total_limit: int = 1  # Keep only final checkpoint
    logging_steps: int = 25
    seed: int = 42
    lr_scheduler_type: str = "constant_with_warmup"
    bf16: bool = field(default_factory=torch.cuda.is_available)
    freeze_pretrained: bool = True
    debug: bool = False

    def training_arguments(self):
        ignore_keys = {
            "model_name",
            "dataset",
            "max_length",
            "debug",
            "freeze_pretrained",
        }
        return {k: v for k, v in asdict(self).items() if k not in ignore_keys}


@dataclass
class Config:
    parscale: ParScaleConfig = field(default_factory=ParScaleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def mk_model_config(
    model_name: str,
    parscale_config: ParScaleConfig,
) -> Qwen2ParScaleConfig:
    if model_name == "debug":
        model_config = DEBUG_CONFIG
    else:
        model_config = AutoConfig.from_pretrained(model_name)

    return Qwen2ParScaleConfig(
        **{
            **model_config.to_dict(),
            **asdict(parscale_config),
        }
    )


def freeze_pretrained_weights(
    model: Qwen2ParScaleForCausalLM, config: Qwen2ParScaleConfig
) -> None:
    """
    Freeze all pretrained weights except ParScale-specific components:
    - prefix_k, prefix_v (prefix cache parameters)
    - aggregate_layer (aggregation MLP)
    - CrossReplicaAttention modules (cross-attention components)

    When parscale_n=1, this effectively disables freezing since there are no
    ParScale-specific parameters to unfreeze.
    """
    # If parscale_n=1, don't freeze anything since there are no ParScale parameters
    if config.parscale_n == 1:
        return

    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze prefix parameters (these exist when parscale_n_tokens > 0)
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "prefix_k"):
            layer.self_attn.prefix_k.requires_grad = True
        if hasattr(layer.self_attn, "prefix_v"):
            layer.self_attn.prefix_v.requires_grad = True

    # Unfreeze ParScale-specific components
    if config.parscale_n > 1:
        # Unfreeze aggregation layer if it exists
        if hasattr(model.model, "aggregate_layer"):
            for param in model.model.aggregate_layer.parameters():
                param.requires_grad = True

        # Unfreeze cross-attention components
        if config.enable_cross_attn:
            for layer in model.model.layers:
                if hasattr(layer, "cross_replica_attn"):
                    for param in layer.cross_replica_attn.parameters():
                        param.requires_grad = True
                if hasattr(layer, "cross_attn_norm"):
                    for param in layer.cross_attn_norm.parameters():
                        param.requires_grad = True


def mk_model(
    model_name: str,
    config: Qwen2ParScaleConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen2ParScaleForCausalLM:
    """Convert Qwen2 model to ParScale."""
    parscale_model = Qwen2ParScaleForCausalLM(config).to(dtype)  # type: ignore
    if model_name == "debug":
        return parscale_model

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    parscale_model.load_state_dict(base_model.state_dict(), strict=False)
    return parscale_model


def proc_dataset(dataset_name):
    """Load and tokenize dataset."""
    match dataset_name:
        case "stack":
            return load_dataset(
                "bigcode/the-stack-dedup",
                "Python",
                split="train",
                streaming=True,
            ).rename_column("content", "text")
        case "pajama":
            return load_dataset(
                "cerebras/SlimPajama-627B",
                split="train",
                streaming=True,
            )
        case "debug":
            return load_dataset(
                "roneneldan/TinyStories",
                split="train",
                streaming=False,
            )
        case _:
            raise ValueError("invalid name", dataset_name)


def init_wandb(accelerator: Accelerator) -> dict:
    """
    Initialize wandb on main process and return wandb config dict for all processes.

    Args:
        accelerator: The Accelerator instance

    Returns:
        Dictionary of wandb config values
    """
    # Initialize on main process
    shared_object = [None]
    if accelerator.is_main_process:
        wandb.init(project=os.environ.get("WANDB_PROJECT", "parscale-xattn"))
        wandb_config = OmegaConf.from_dotlist(
            [f"{k}={v}" for k, v in dict(wandb.config).items()]
        )
        shared_object[0] = wandb_config
        wandb.summary["wandb_config"] = dict(wandb_config)

    wandb_config = broadcast_object_list(shared_object, from_process=0).pop()
    return wandb_config


def mk_config(accelerator: Accelerator, wandb_config) -> Config:
    base_config: Config = OmegaConf.structured(Config)
    yaml_config = (
        OmegaConf.load(config_file)
        if (config_file := os.environ.get("CONFIG_FILE"))
        else {}
    )
    cli_config = OmegaConf.from_cli()

    config = OmegaConf.merge(
        base_config,
        yaml_config,
        cli_config,
        wandb_config,  # sweep config
    )

    ## Validate Pydantic
    config = OmegaConf.to_container(config, structured_config_mode=SCMode.DICT)
    config = TypeAdapter(Config).validate_python(config)
    if accelerator.is_main_process:
        wandb.summary["config"] = dict(asdict(config))
    return config


def main():
    accelerator = Accelerator()
    wandb_config = init_wandb(accelerator)
    config = mk_config(accelerator, wandb_config)

    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)

    if config.training.debug:
        model_name = dataset_name = "debug"
    else:
        model_name = config.training.model_name
        dataset_name = config.training.dataset

    dataset = proc_dataset(dataset_name)
    model_config = mk_model_config(model_name, config.parscale)
    model = mk_model(model_name, model_config)

    if config.training.freeze_pretrained:
        freeze_pretrained_weights(model, model_config)

    def collate_fn(features):
        texts = [f["text"] for f in features]
        batch = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.training.max_length,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    training_args = TrainingArguments(
        **config.training.training_arguments(),
        report_to="wandb" if accelerator.is_main_process else "none",
        remove_unused_columns=False,
        # Multi-GPU setup
        ddp_find_unused_parameters=True,
        # fsdp="no_shard",
        # fsdp_config={
        #     "fsdp_activation_checkpointing": False,
        #     "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        #     "fsdp_cpu_ram_efficient_loading": False,
        #     "fsdp_offload_params": False,
        #     "fsdp_reshard_after_forward": False,
        #     "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        #     "fsdp_transformer_layer_cls_to_wrap": [
        #         "Qwen2DecoderLayer",
        #         "ParScaleCrossAttnDecoderLayer",
        #     ],
        # },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.training.output_dir)


if __name__ == "__main__":
    main()
