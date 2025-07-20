#!/usr/bin/env python3
"""Simple training script for ParScale Cross-Attention using OmegaConf + Pydantic."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import Optional
import sys

from src.parscale_xattn import Qwen2ParScaleForCausalLM, Qwen2ParScaleConfig

class ParScaleConfig(BaseModel):
    parscale_n: int = 4
    parscale_n_tokens: int = 48
    parscale_enable_cross_attn: bool = False
    parscale_cross_attn_layers: Optional[list[int]] = None

class TrainingConfig(BaseModel):
    base_model: str = "Qwen/Qwen2-1.5B"
    dataset: str = "bigcode/the-stack"
    output_dir: str = "./parscale-model"
    max_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    save_steps: int = 1000
    logging_steps: int = 100
    seed: int = 42

class Config(BaseModel):
    parscale: ParScaleConfig = ParScaleConfig()
    training: TrainingConfig = TrainingConfig()

def convert_qwen2_to_parscale(base_model_name: str, parscale_config: ParScaleConfig) -> Qwen2ParScaleForCausalLM:
    """Convert Qwen2 model to ParScale."""
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    base_config = base_model.config
    
    config = Qwen2ParScaleConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=getattr(base_config, 'num_key_value_heads', base_config.num_attention_heads),
        hidden_act=base_config.hidden_act,
        max_position_embeddings=base_config.max_position_embeddings,
        initializer_range=base_config.initializer_range,
        rms_norm_eps=base_config.rms_norm_eps,
        use_cache=base_config.use_cache,
        tie_word_embeddings=getattr(base_config, 'tie_word_embeddings', False),
        rope_theta=getattr(base_config, 'rope_theta', 10000.0),
        parscale_n=parscale_config.parscale_n,
        parscale_n_tokens=parscale_config.parscale_n_tokens,
        parscale_enable_cross_attn=parscale_config.parscale_enable_cross_attn,
        parscale_cross_attn_layers=parscale_config.parscale_cross_attn_layers,
    )
    
    parscale_model = Qwen2ParScaleForCausalLM(config)
    
    # Copy weights
    parscale_model.model.embed_tokens.load_state_dict(base_model.model.embed_tokens.state_dict())
    parscale_model.lm_head.load_state_dict(base_model.lm_head.state_dict())
    parscale_model.model.norm.load_state_dict(base_model.model.norm.state_dict())
    
    for i, (base_layer, parscale_layer) in enumerate(zip(base_model.model.layers, parscale_model.model.layers)):
        parscale_layer.self_attn.q_proj.load_state_dict(base_layer.self_attn.q_proj.state_dict())
        parscale_layer.self_attn.k_proj.load_state_dict(base_layer.self_attn.k_proj.state_dict())
        parscale_layer.self_attn.v_proj.load_state_dict(base_layer.self_attn.v_proj.state_dict())
        parscale_layer.self_attn.o_proj.load_state_dict(base_layer.self_attn.o_proj.state_dict())
        parscale_layer.mlp.load_state_dict(base_layer.mlp.state_dict())
        parscale_layer.input_layernorm.load_state_dict(base_layer.input_layernorm.state_dict())
        parscale_layer.post_attention_layernorm.load_state_dict(base_layer.post_attention_layernorm.state_dict())
        
        if config.parscale_n > 1:
            torch.nn.init.normal_(parscale_layer.self_attn.prefix_k, std=config.initializer_range)
            torch.nn.init.normal_(parscale_layer.self_attn.prefix_v, std=config.initializer_range)
    
    return parscale_model

def load_dataset_simple(dataset_name: str, tokenizer, max_length: int):
    """Load and tokenize dataset."""
    if "stack" in dataset_name.lower():
        dataset = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train", streaming=True)
        text_column = "content"
    elif "pile" in dataset_name.lower():
        dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
        text_column = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)
    
    return dataset.map(tokenize_function, batched=True).take(10000)  # Small subset for testing

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_simple.py config.yaml")
        sys.exit(1)
    
    # Load config
    omega_conf = OmegaConf.load(sys.argv[1])
    config = Config(**omega_conf)
    
    print(f"Training with config: {config}")
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = convert_qwen2_to_parscale(config.training.base_model, config.parscale)
    dataset = load_dataset_simple(config.training.dataset, tokenizer, config.training.max_length)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=None,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.training.output_dir)
    print(f"Training completed! Model saved to {config.training.output_dir}")

if __name__ == "__main__":
    main()