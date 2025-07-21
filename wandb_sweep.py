#!/usr/bin/env python3
"""
Wandb sweep script to replicate original ParScale paper experiments.
Focused sweeps with only the parameters we're changing.
"""

import subprocess
import sys
import tempfile

import wandb
from omegaconf import OmegaConf

# Simplified sweep configurations
SWEEP_CONFIGS = {
    # 1. First verify learning rate with P=1 and P=4
    "lr_verification": {
        "name": "LR-Verification-P1-P4",
        "description": "Verify 3e-4 learning rate with P=1 and P=4",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 4]},
            "parscale.enable_cross_attn": {"values": False},
            "training.learning_rate": {"values": [1e-4, 3e-4, 5e-4, 1e-3]},
        },
    },
    # 2. Fixed LR, sweep over P values (original paper replication)
    "parscale_scaling": {
        "name": "ParScale-Scaling-P1-2-4-8",
        "description": "Original paper replication: P=1,2,4,8 with fixed LR",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 2, 4, 8]},
            "parscale.enable_cross_attn": {"values": False},
        },
    },
    # 3. Same as above but with cross-attention on all layers
    "xattn_all_layers": {
        "name": "Cross-Attention-All-Layers-P1-2-4-8",
        "description": "Cross-attention on all layers with P=1,2,4,8",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 2, 4, 8]},
            "parscale.enable_cross_attn": {"value": True},
        },
    },
    # 4. Same but cross-attention on preset layers
    "xattn_preset_layers": {
        "name": "Cross-Attention-Preset-Layers-P1-2-4-8",
        "description": "Cross-attention on preset layers with P=1,2,4,8",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 2, 4, 8]},
            "parscale.enable_cross_attn": {"value": True},
            "parscale.parscale_cross_attn_layers": {
                "value": [0, 6, 12, 18]
            },  # Early, mid, late layers
        },
    },
}


def single_sweep():
    """Training function called by wandb agent."""

    # Initialize wandb run
    run = wandb.init()

    wandb_config = OmegaConf.from_dotlist(
        [f"{k}={v}" for k, v in dict(wandb.config).items()]
    )
    wandb_config.training.output_dir = f"./models/{run.name}"

    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=wandb_config, f=fp.name)

    cmd = [
        f"CONFIG_FILE={wandb_config}",
        "torchrun",
        "--nproc_per_node",
        "8",
        "train.py",
        f"--training.output_dir./sweeps/{run.name}",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        wandb.log({"training_failed": True})
        return

    wandb.finish()


def create_sweep(sweep_name: str, project: str = "parscale-cross-attention"):
    """Create a wandb sweep."""
    if sweep_name not in SWEEP_CONFIGS:
        print(f"Unknown sweep: {sweep_name}")
        print(f"Available sweeps: {list(SWEEP_CONFIGS.keys())}")
        return None

    sweep_config = SWEEP_CONFIGS[sweep_name].copy()
    sweep_config["program"] = "wandb_sweep.py run"

    print(f"Creating sweep: {sweep_name}")

    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"Created sweep: {sweep_id}")
    print(f"Run with: wandb agent {sweep_id}")

    return sweep_id


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python wandb_sweep.py create <sweep_name>  # Create a sweep")
        print(
            "  python wandb_sweep.py run                  # Run training (called by wandb agent)"
        )
        print("\nAvailable sweeps:")
        for name, config in SWEEP_CONFIGS.items():
            print(f"  {name}: {config['description']}")
        return

    command = sys.argv[1]

    if command == "create":
        if len(sys.argv) < 3:
            print("Available sweeps:")
            for name, config in SWEEP_CONFIGS.items():
                print(f"  {name}: {config['description']}")
            return
        sweep_name = sys.argv[2]
        project = sys.argv[3] if len(sys.argv) > 3 else "parscale-cross-attention"
        create_sweep(sweep_name, project)

    elif command == "run":
        single_sweep()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
