from accelerate import Accelerator
import torch.distributed as dist
from transformers import AutoTokenizer, Trainer, TrainingArguments

from parscale_xattn.trainer import (
    init_wandb,
    mk_config,
    proc_dataset,
    mk_model_config,
    mk_model,
    freeze_pretrained_weights,
)


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
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
