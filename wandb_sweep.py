#!/usr/bin/env python3
"""
Wandb sweep script to replicate original ParScale paper experiments.
Focused sweeps with only the parameters we're changing.
"""

import wandb
import yaml
import subprocess
import sys
import os

# Simplified sweep configurations
SWEEP_CONFIGS = {
    # 1. First verify learning rate with P=1 and P=4
    "lr_verification": {
        "name": "LR-Verification-P1-P4", 
        "description": "Verify 3e-4 learning rate with P=1 and P=4",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 4]},
            "training.learning_rate": {"values": [1e-4, 3e-4, 5e-4, 1e-3]},
        }
    },
    
    # 2. Fixed LR, sweep over P values (original paper replication)
    "parscale_scaling": {
        "name": "ParScale-Scaling-P1-2-4-8",
        "description": "Original paper replication: P=1,2,4,8 with fixed LR",
        "method": "grid", 
        "parameters": {
            "parscale.parscale_n": {"values": [1, 2, 4, 8]},
        }
    },

    # 3. Same as above but with cross-attention on all layers
    "xattn_all_layers": {
        "name": "Cross-Attention-All-Layers-P1-2-4-8",
        "description": "Cross-attention on all layers with P=1,2,4,8",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 2, 4, 8]},
            "parscale.parscale_enable_cross_attn": {"value": True},
        }
    },

    # 4. Same but cross-attention on preset layers
    "xattn_preset_layers": {
        "name": "Cross-Attention-Preset-Layers-P1-2-4-8", 
        "description": "Cross-attention on preset layers with P=1,2,4,8",
        "method": "grid",
        "parameters": {
            "parscale.parscale_n": {"values": [1, 2, 4, 8]},
            "parscale.parscale_enable_cross_attn": {"value": True},
            "parscale.parscale_cross_attn_layers": {"value": [0, 6, 12, 18]},  # Early, mid, late layers
        }
    }
}

def run_training_with_wandb():
    """Training function called by wandb agent."""
    import wandb
    from omegaconf import OmegaConf
    
    # Initialize wandb run
    run = wandb.init()
    
    # Get config from wandb
    config_dict = dict(wandb.config)
    
    # Convert to hierarchical structure for OmegaConf
    structured_config = {"parscale": {}, "training": {}}
    for key, value in config_dict.items():
        if key.startswith("parscale."):
            param_name = key.replace("parscale.", "")
            structured_config["parscale"][param_name] = value
        elif key.startswith("training."):
            param_name = key.replace("training.", "")
            structured_config["training"][param_name] = value
    
    # Add output directory with run name
    structured_config["training"]["output_dir"] = f"./models/{run.name}"
    
    # Create config file for this run
    config_path = f"/tmp/sweep_config_{run.id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(structured_config, f)
    
    # Run training script
    cmd = [
        "python", "train.py",
        "--config-path", "/tmp",
        "--config-name", f"sweep_config_{run.id}",
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(f"stdout: {e.stdout}")  
        print(f"stderr: {e.stderr}")
        wandb.log({"training_failed": True})
        return
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)
    
    wandb.finish()

def create_sweep(sweep_name: str, project: str = "parscale-cross-attention"):
    """Create a wandb sweep."""
    if sweep_name not in SWEEP_CONFIGS:
        print(f"Unknown sweep: {sweep_name}")
        print(f"Available sweeps: {list(SWEEP_CONFIGS.keys())}")
        return None
    
    sweep_config = SWEEP_CONFIGS[sweep_name].copy()
    sweep_config["program"] = "wandb_sweep.py"
    
    print(f"Creating sweep: {sweep_name}")
    print(f"Config: {yaml.dump(sweep_config, default_flow_style=False)}")
    
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"Created sweep: {sweep_id}")
    print(f"Run with: wandb agent {sweep_id}")
    
    return sweep_id

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python wandb_sweep.py create <sweep_name>  # Create a sweep")
        print("  python wandb_sweep.py run                  # Run training (called by wandb agent)")
        print(f"\nAvailable sweeps:")
        for name, config in SWEEP_CONFIGS.items():
            print(f"  {name}: {config['description']}")
        return
    
    command = sys.argv[1]
    
    if command == "create":
        if len(sys.argv) < 3:
            print(f"Available sweeps:")
            for name, config in SWEEP_CONFIGS.items():
                print(f"  {name}: {config['description']}")
            return
        sweep_name = sys.argv[2]
        project = sys.argv[3] if len(sys.argv) > 3 else "parscale-cross-attention"
        create_sweep(sweep_name, project)
        
    elif command == "run":
        # This is called by wandb agent
        run_training_with_wandb()
        
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()