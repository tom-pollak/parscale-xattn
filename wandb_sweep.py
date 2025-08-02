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
    "training.max_steps": {"value": 5000},
}

SWEEP_CONFIGS = {
    # 0. Baseline: Prefix token ablation with single replica
    "baseline_prefix_ablation": {
        "name": "Baseline-Prefix-Token-Ablation-P1",
        "description": "Isolate prefix token contribution with P=1 (no replicas)",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"value": 1},
            "parscale.parscale_n_tokens": {"values": [0, 24, 48, 96]},
            "parscale.enable_cross_attn": {"value": False},
            "training.learning_rate": {"value": 3e-4},
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
    # 2. Fixed LR, sweep over P values (original paper replication)
    "parscale_scaling": {
        "name": "ParScale-Scaling-P1-2-4-8",
        "description": "Original paper replication: P=1,2,4,8 with fixed LR",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n_tokens": {"value": 48},
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
            "parscale.parscale_n_tokens": {"values": [0, 48]},
            "parscale.parscale_n": {"values": [2, 4, 8]},
            "parscale.enable_cross_attn": {"value": True},
            "parscale.enable_replica_rope": {"value": True},
        },
    },
    # 4. Same but cross-attention on preset layers
    "xattn_preset_layers": {
        "name": "Cross-Attention-Preset-Layers-P1-2-4-8",
        "description": "Cross-attention on preset layers with P=1,2,4,8",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n_tokens": {"values": [0, 48]},
            "parscale.parscale_n": {"values": [2, 4, 8]},
            "parscale.enable_cross_attn": {"value": True},
            "parscale.enable_replica_rope": {"value": True},
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

    sweep_config = SWEEP_CONFIGS[sweep_name]
    sweep_config["parameters"] = {**BASE_PARAMS, **sweep_config["parameters"]}
    config = {**BASE_CONFIG, **sweep_config}
    sweep_id = wandb.sweep(config, project=PROJECT_NAME)
    print("Run with:\n", f"uv run wandb agent graphcore/{PROJECT_NAME}/{sweep_id}")


if __name__ == "__main__":
    main()
