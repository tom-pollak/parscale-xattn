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
    # 0. Baseline: Prefix token ablation with single replica
    "full_sweep": {
        "name": "Baseline-Prefix-Token-Ablation-P1-2-3",
        "description": "Isolate prefix token contribution with P=1 (no replicas)",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 2, 4]},
            "parscale.parscale_n_tokens": {"values": [0, 48]},
            "parscale.enable_cross_attn": {"values": [False, True]},
            "parscale.enable_replica_rope": {"values": [False, True]},
            "parscale.parscale_cross_attn_layers": {
                "values": [None, [0], [0, 8, 16, 24]]
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
