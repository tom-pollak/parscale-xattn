"""
Wandb sweep script to replicate original ParScale paper experiments.
Focused sweeps with only the parameters we're changing.
"""

import sys

import wandb

PROJECT_NAME = "parscale-xattn"

BASE_CONFIG = {
    "command": ["uv", "run", "accelerate", "launch", "train.py"],
    "metric": {
        "name": "train/loss",
        "goal": "minimize",
    },
}

BASE_PARAMS = {
    "training.save_total_limit": {"value": 0},
    "training.warmup_steps": {"value": 500},
    "training.max_steps": {"value": 2000},
}

SWEEP_CONFIGS = {
    "full_sweep_no_xattn": {
        "name": "Baseline-No-CrossAttn-P1-2-4",
        "description": "Baseline without cross attention",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 2, 4]},
            "parscale.parscale_n_tokens": {"value": 48},
            "parscale.enable_cross_attn": {"value": False},
        },
    },
    "full_sweep_with_xattn": {
        "name": "Baseline-With-CrossAttn-P1-2-4",
        "description": "Baseline with cross attention variations",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [2, 4]},
            "parscale.parscale_n_tokens": {"values": [0, 48]},
            "parscale.enable_cross_attn": {"value": True},
            "parscale.enable_replica_rope": {"values": [False, True]},
            "parscale.parscale_cross_attn_layers": {
                "values": [None, [0], [0, 6, 12, 18]]
            },
        },
    },
    "full_sweep_no_xattn_unfrozen": {
        "name": "Unfrozen-No-CrossAttn-P1-2-4",
        "description": "Baseline without cross attention",
        "method": "grid",
        "parameters": {
            "training.freeze_pretrained": {"value": False},
            "parscale.parscale_n": {"values": [1, 2, 4]},
            "parscale.parscale_n_tokens": {"value": 48},
            "parscale.enable_cross_attn": {"value": False},
        },
    },
    "full_sweep_with_xattn_unfrozen": {
        "name": "Unfrozen-With-CrossAttn-P1-2-4",
        "description": "Baseline with cross attention variations",
        "method": "grid",
        "parameters": {
            # unfrozen + cross-attn runs out of memory with 4x4
            "training.per_device_train_batch_size": {"value": 2},
            "training.gradient_accumulation_steps": {"value": 8},
            "training.freeze_pretrained": {"value": False},
            "parscale.parscale_n": {"values": [2, 4]},
            "parscale.parscale_n_tokens": {"values": [0, 48]},
            "parscale.enable_cross_attn": {"value": True},
            "parscale.enable_replica_rope": {"values": [False, True]},
            "parscale.parscale_cross_attn_layers": {
                "values": [None, [0], [0, 6, 12, 18]]
            },
        },
    },
    # 1. First verify learning rate with P=1, P=4, P=8
    "lr_verification": {
        "name": "LR-Sweep-P-4-8",
        "description": "Verify 3e-4 learning rate with P=1, P=4, P=8",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n_tokens": {"value": 48},
            "parscale.parscale_n": {"values": [1, 4, 8]},
            "parscale.enable_cross_attn": {"value": False},
            "training.learning_rate": {"values": [1e-4, 3e-4, 5e-4, 1e-3]},
        },
    },
}


def main():
    sweep_name = sys.argv[1] if len(sys.argv) > 1 else None
    if sweep_name not in SWEEP_CONFIGS:
        print(f"Unknown sweep: {sweep_name}")
        print(f"Available sweeps: {list(SWEEP_CONFIGS.keys())}")
        return None

    sweep_config = SWEEP_CONFIGS[sweep_name]
    sweep_config["parameters"] = {**BASE_PARAMS, **sweep_config["parameters"]}
    config = {**BASE_CONFIG, **sweep_config}
    sweep_id = wandb.sweep(config, project=PROJECT_NAME)
    print("Run with:\n", f"uv run wandb agent graphcore/{PROJECT_NAME}/{sweep_id}")


if __name__ == "__main__":
    main()
