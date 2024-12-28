import os
import yaml
from pathlib import Path

def create_directory_structure():
    """Create the configs directory structure"""
    # Create main directories
    dirs = [
        "configs/model",
        "configs/training",
        "configs/wandb"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create main config.yaml
    main_config = {
        "defaults": [
            {"_self_": None},
            {"model": "default"},
            {"training": "default"},
            {"wandb": "default"}
        ]
    }
    
    with open("configs/config.yaml", "w") as f:
        yaml.safe_dump(main_config, f, sort_keys=False)
    
    # Create model config
    model_config = {
        "name": "multitask-molecular",
        "pretrained_model": "seyonec/ChemBERTa-zinc-base-v1",
        "num_features": 134,
        "z_score_threshold": 2.0,
        "reg_weight": 1.0,
        "cls_weight": 1.0
    }
    
    with open("configs/model/default.yaml", "w") as f:
        yaml.safe_dump(model_config, f)
    
    # Create training config
    training_config = {
        "batch_size": 32,
        "num_workers": 4,
        "learning_rate": 1e-4,
        "max_epochs": 100,
        "patience": 10,
        "gradient_clip_val": 1.0,
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs"
    }
    
    with open("configs/training/default.yaml", "w") as f:
        yaml.safe_dump(training_config, f)
    
    # Create wandb config
    wandb_config = {
        "project": "molecular-multitask",
        "run_name": None,
        "tags": ["multitask", "molecular"]
    }
    
    with open("configs/wandb/default.yaml", "w") as f:
        yaml.safe_dump(wandb_config, f)

def validate_hydra_config():
    """Simple script to validate Hydra config loading"""
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(version_base=None, config_path="configs", config_name="config")
    def validate_config(cfg: DictConfig):
        print("\nLoaded configuration:")
        print("=" * 40)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 40)
        
        # Validate essential parameters exist
        essential_params = {
            "model.name": "Model name",
            "model.num_features": "Number of features",
            "training.batch_size": "Batch size",
            "wandb.project": "WandB project name"
        }
        
        print("\nValidating essential parameters:")
        all_valid = True
        for param_path, param_name in essential_params.items():
            try:
                value = OmegaConf.select(cfg, param_path)
                print(f"✓ {param_name}: {value}")
            except:
                print(f"✗ Missing {param_name} ({param_path})")
                all_valid = False
        
        if all_valid:
            print("\nConfig validation successful! You can now use this configuration in your training script.")
        else:
            print("\nConfig validation failed. Please check the missing parameters.")

    validate_config()

if __name__ == "__main__":
    print("Setting up Hydra configuration structure...")
    create_directory_structure()
    print("\nDirectory structure created!")
    print("\nValidating configuration...")
    validate_hydra_config()
