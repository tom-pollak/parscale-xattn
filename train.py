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
    Qwen2ParScaleConfig,
    Qwen2ParScaleForCausalLM,
    Qwen2ParScaleConfig,
    Qwen2ParScaleForCausalLM,
)


@dataclass
class ModelConfig:
    hidden_size: int = 256
    intermediate_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 32
    num_key_value_heads: int = 16
    max_position_embeddings: int = 10000


@dataclass
class ParScaleConfig:
    parscale_n: int = 1
    parscale_n_tokens: int = 48
    enable_cross_attn: bool = False
    parscale_cross_attn_layers: Optional[list[int]] = None
    enable_replica_rope: bool = False


@dataclass
class TrainingConfig:
    base_model: str = "Qwen/Qwen2-0.5B"
    dataset: str = "pajama"
    output_dir: str = "./parscale-model"
    max_length: int = 2048
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 76294  # 20B tokens ÷ (4×4×2048×8) ≈ 76k steps from paper
    learning_rate: float = 3e-4  # Stage 2 learning rate from paper
    warmup_steps: int = 2000  # 2K warm-up from paper
    save_steps: int = 76294  # Save only at the end
    save_total_limit: int = 1  # Keep only final checkpoint
    logging_steps: int = 1000  # Log every 1000 steps for long training
    seed: int = 42
    lr_scheduler_type: str = "constant_with_warmup"
    bf16: bool = field(default_factory=torch.cuda.is_available)
    debug: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    parscale: ParScaleConfig = field(default_factory=ParScaleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def mk_model_config(
    base_model_name: str, model_config: ModelConfig, parscale_config: ParScaleConfig
) -> Qwen2ParScaleConfig:
    base_config = AutoConfig.from_pretrained(base_model_name)
    return Qwen2ParScaleConfig(
        **base_config.to_dict(),
        **asdict(parscale_config),
        **asdict(model_config),
    )


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
    if accelerator.is_main_process:
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "parscale-xattn"))
        os.environ["WANDB_RUN_ID"] = run.id
        wandb_config = OmegaConf.from_dotlist(
            [f"{k}={v}" for k, v in dict(wandb.config).items()]
        )
    else:
        wandb_config = {}

    wandb_config = broadcast_object_list([wandb_config], from_process=0).pop()
    assert isinstance(wandb_config, dict)
    return wandb_config


def mk_config(wandb_config) -> Config:
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
    return config


def main():
    accelerator = Accelerator()
    wandb_config = init_wandb(accelerator)
    config = mk_config(wandb_config)

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    if config.training.debug:
        # Define a tiny configuration
        tiny_config = Qwen2ParScaleConfig(
            vocab_size=tokenizer.vocab_size,
            **asdict(config.model),
            **asdict(config.parscale),
        )
        model = Qwen2ParScaleForCausalLM(tiny_config).to(torch.bfloat16)
        dataset = proc_dataset("debug")
    else:
        config = mk_model_config(
            config.training.base_model,
            model_config=config.model,
            parscale_config=config.parscale,
        )
        model = convert_qwen2_to_parscale(config.training.base_model, config)
        dataset = proc_dataset(config.training.dataset)

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
        **asdict(config.training),
        report_to="wandb" if accelerator.is_main_process else "none",
        # Multi-GPU setup
        ddp_find_unused_parameters=False,
        fsdp="full_shard",
        fsdp_config={
            "fsdp_activation_checkpointing": True,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_offload_params": False,
            "fsdp_reshard_after_forward": True,
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
        },
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
