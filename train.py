import os
from dataclasses import field
from typing import Literal, Optional

import torch
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf, SCMode
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.parscale_xattn import Qwen2ParScaleConfig, Qwen2ParScaleForCausalLM


@dataclass
class ParScaleConfig:
    parscale_n: int = 4
    parscale_n_tokens: int = 48
    enable_cross_attn: bool = False
    parscale_cross_attn_layers: Optional[list[int]] = None


@dataclass
class TrainingConfig:
    base_model: str = "Qwen/Qwen2-1.5B"
    dataset: str = "stack"
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

    config = Qwen2ParScaleConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=getattr(
            base_config, "num_key_value_heads", base_config.num_attention_heads
        ),
        hidden_act=base_config.hidden_act,
        max_position_embeddings=base_config.max_position_embeddings,
        initializer_range=base_config.initializer_range,
        rms_norm_eps=base_config.rms_norm_eps,
        use_cache=base_config.use_cache,
        tie_word_embeddings=getattr(base_config, "tie_word_embeddings", False),
        rope_theta=getattr(base_config, "rope_theta", 10000.0),
        parscale_n=parscale_config.parscale_n,
        parscale_n_tokens=parscale_config.parscale_n_tokens,
        enable_cross_attn=parscale_config.enable_cross_attn,
        parscale_cross_attn_layers=parscale_config.parscale_cross_attn_layers,
    )

    parscale_model = Qwen2ParScaleForCausalLM(config)

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
                data_dir="data/python",
                split="train",
                streaming=True,
            ).rename_column("content", "text")
        case "pile":
            return load_dataset("EleutherAI/pile", split="train", streaming=True)
        case _:
            raise ValueError("invalid name", dataset_name)


def mk_config() -> Config:
    base_config: Config = OmegaConf.structured(Config)
    yaml_config = (
        OmegaConf.load(config_file)
        if (config_file := os.environ.get("CONFIG_FILE"))
        else {}
    )
    cli_config = OmegaConf.from_cli()
    wandb_config = OmegaConf.from_dotlist(
        [f"{k}={v}" for k, v in dict(wandb.config).items()]
    )

    config = OmegaConf.merge(
        base_config,
        yaml_config,
        cli_config,
        wandb_config,  # sweep config
    )
    config = OmegaConf.to_container(config, structured_config_mode=SCMode.DICT)

    config = TypeAdapter(Config).validate_python(config)
    print(f"####\n{config.to_yaml()}\n####")
    return config


def main():
    wandb.init(project=os.environ["WANDB_PROJECT"])

    config = mk_config()

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    model = convert_qwen2_to_parscale(config.training.base_model, config.parscale)
    dataset = proc_dataset(config.training.dataset)

    data_collator = DataCollatorWithPadding(
        tokenizer,
        padding="longest",
        max_length=config.training.max_length,
    )

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
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb",
        # Learning rate schedule - constant with warmup like paper's stage 2
        lr_scheduler_type="constant_with_warmup",
        # Multi-GPU setup
        ddp_find_unused_parameters=False,
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="Qwen2DecoderLayer",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.training.output_dir)
    print(f"Training completed! Model saved to {config.training.output_dir}")


if __name__ == "__main__":
    main()
