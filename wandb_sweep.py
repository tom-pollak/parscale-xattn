"""
Wandb sweep script to replicate original ParScale paper experiments.
Focused sweeps with only the parameters we're changing.
"""

import sys

import wandb

PROJECT_NAME = "parscale-xattn"

BASE_CONFIG = {
    "command": ["accelerate", "launch", "train.py"],
    "metric": {
        "name": "train/loss",
        "goal": "minimize",
    },
}

SWEEP_CONFIGS = {
    # 1. First verify learning rate with P=1 and P=4
    "lr_verification_1": {
        "name": "LR-Sweep-P1",
        "description": "Verify 3e-4 learning rate with P=1 and P=4",
        "method": "grid",
        "parameters": {
            "training.save_total_limit": {"value": 0},
            "parscale.parscale_n": {"value": 1},
            "parscale.parscale_n_tokens": {"value": 0},
            "parscale.enable_cross_attn": {"value": False},
            "training.learning_rate": {"values": [1e-4, 3e-4, 5e-4, 1e-3]},
            "training.max_steps": {"value": 7500},
        },
    },
    # 1b. Verify with 4/8 P
    "lr_verification_48": {
        "name": "LR-Sweep-P-4-8",
        "description": "Verify 3e-4 learning rate with P=1 and P=4",
        "method": "grid",
        "parameters": {
            "training.save_total_limit": {"value": 0},
            "parscale.parscale_n": {"values": [4, 8]},
            "parscale.enable_cross_attn": {"value": False},
            "training.learning_rate": {"values": [1e-4, 3e-4, 5e-4, 1e-3]},
            "training.max_steps": {"value": 7500},
        },
    },
    # 2. Fixed LR, sweep over P values (original paper replication)
    "parscale_scaling": {
        "name": "ParScale-Scaling-P1-2-4-8",
        "description": "Original paper replication: P=1,2,4,8 with fixed LR",
        "method": "grid",
        "parameters": {
            "training.save_total_limit": {"value": 0},
            "parscale.parscale_n": {"values": [2, 4, 8]},
            "parscale.enable_cross_attn": {"value": False},
        },
    },
    # 3. Same as above but with cross-attention on all layers
    "xattn_all_layers": {
        "name": "Cross-Attention-All-Layers-P1-2-4-8",
        "description": "Cross-attention on all layers with P=1,2,4,8",
        "method": "grid",
        "parameters": {
            "training.save_total_limit": {"value": 0},
            "parscale.parscale_n": {"values": [2, 4, 8]},
            "parscale.enable_cross_attn": {"value": True},
        },
    },
    # 4. Same but cross-attention on preset layers
    "xattn_preset_layers": {
        "name": "Cross-Attention-Preset-Layers-P1-2-4-8",
        "description": "Cross-attention on preset layers with P=1,2,4,8",
        "method": "grid",
        "parameters": {
            "training.save_total_limit": {"value": 0},
            "parscale.parscale_n": {"values": [2, 4, 8]},
            "parscale.enable_cross_attn": {"value": True},
            "parscale.parscale_cross_attn_layers": {
                "value": [0, 6, 12, 18]
            },  # Early, mid, late layers
        },
    },
}


def main():
    sweep_name = sys.argv[1] if len(sys.argv) > 1 else None
    if sweep_name not in SWEEP_CONFIGS:
        print(f"Unknown sweep: {sweep_name}")
        print(f"Available sweeps: {list(SWEEP_CONFIGS.keys())}")
        return None

    config = {**BASE_CONFIG, **SWEEP_CONFIGS[sweep_name]}
    sweep_id = wandb.sweep(config, project=PROJECT_NAME)
    print("Run with:\n", f"uv run wandb agent graphcore/{PROJECT_NAME}/{sweep_id}")


if __name__ == "__main__":
    main()
