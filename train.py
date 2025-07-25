import os
from dataclasses import field
from typing import Optional

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast
from datasets import load_dataset
from omegaconf import OmegaConf, SCMode
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.parscale_xattn import Qwen2ParScaleConfig, Qwen2ParScaleForCausalLM


@dataclass
class ParScaleConfig:
    parscale_n: int = 1
    parscale_n_tokens: int = 48
    enable_cross_attn: bool = False
    parscale_cross_attn_layers: Optional[list[int]] = None
    enable_replica_rope: bool = False


@dataclass
class TrainingConfig:
    base_model: str = "Qwen/Qwen2-1.5B"
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
    debug: bool = False


@dataclass
class Config:
    parscale: ParScaleConfig = field(default_factory=ParScaleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def convert_qwen2_to_parscale(
    base_model_name: str, parscale_config: ParScaleConfig
) -> Qwen2ParScaleForCausalLM:
    """Convert Qwen2 model to ParScale."""
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16
    )
    base_config = base_model.config

    config_dict = base_config.to_dict()
    config_dict.update(
        parscale_n=parscale_config.parscale_n,
        parscale_n_tokens=parscale_config.parscale_n_tokens,
        enable_cross_attn=parscale_config.enable_cross_attn,
        parscale_cross_attn_layers=parscale_config.parscale_cross_attn_layers,
        enable_replica_rope=parscale_config.enable_replica_rope,
    )
    config = Qwen2ParScaleConfig(**config_dict)

    parscale_model = Qwen2ParScaleForCausalLM(config).to(torch.bfloat16)

    # Copy weights
    parscale_model.model.embed_tokens.load_state_dict(
        base_model.model.embed_tokens.state_dict()
    )
    parscale_model.lm_head.load_state_dict(base_model.lm_head.state_dict())
    parscale_model.model.norm.load_state_dict(base_model.model.norm.state_dict())

    for i, (base_layer, parscale_layer) in enumerate(
        zip(base_model.model.layers, parscale_model.model.layers)
    ):
        parscale_layer.self_attn.q_proj.load_state_dict(
            base_layer.self_attn.q_proj.state_dict()
        )
        parscale_layer.self_attn.k_proj.load_state_dict(
            base_layer.self_attn.k_proj.state_dict()
        )
        parscale_layer.self_attn.v_proj.load_state_dict(
            base_layer.self_attn.v_proj.state_dict()
        )
        parscale_layer.self_attn.o_proj.load_state_dict(
            base_layer.self_attn.o_proj.state_dict()
        )
        parscale_layer.mlp.load_state_dict(base_layer.mlp.state_dict())
        parscale_layer.input_layernorm.load_state_dict(
            base_layer.input_layernorm.state_dict()
        )
        parscale_layer.post_attention_layernorm.load_state_dict(
            base_layer.post_attention_layernorm.state_dict()
        )

        if config.parscale_n > 1:
            torch.nn.init.normal_(
                parscale_layer.self_attn.prefix_k, std=config.initializer_range
            )
            torch.nn.init.normal_(
                parscale_layer.self_attn.prefix_v, std=config.initializer_range
            )

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
            return load_dataset("roneneldan/TinyStories", split="train", streaming=False)
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

    wandb_config = broadcast(wandb_config, from_process=0)
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
    if accelerator.is_main_process:
        wandb.init(project=os.environ.get("WANDB_PROJECT", "parscale-xattn"))

    wandb_config = init_wandb(accelerator)
    config = mk_config(wandb_config)

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    if config.training.debug:
        # Define a tiny configuration
        tiny_config = Qwen2ParScaleConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            parscale_n=config.parscale.parscale_n,
            parscale_n_tokens=config.parscale.parscale_n_tokens,
            enable_cross_attn=config.parscale.enable_cross_attn,
            parscale_cross_attn_layers=config.parscale.parscale_cross_attn_layers,
            enable_replica_rope=config.parscale.enable_replica_rope,
        )
        model = Qwen2ParScaleForCausalLM(tiny_config).to(torch.bfloat16)
        dataset = proc_dataset("debug")
    else:
        model = convert_qwen2_to_parscale(config.training.base_model, config.parscale)
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

    fsdp_config = {
        "fsdp_activation_checkpointing": True,
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_offload_params": False,
        "fsdp_reshard_after_forward": True,
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
    }

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="wandb" if accelerator.is_main_process else "none",
        # Learning rate schedule - constant with warmup like paper's stage 2
        lr_scheduler_type="constant_with_warmup",
        # Multi-GPU setup
        ddp_find_unused_parameters=False,
        fsdp="full_shard",
        fsdp_config=fsdp_config,
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
