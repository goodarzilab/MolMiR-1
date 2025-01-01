"""
run_sweep_agent.py - Script for running sweep agents in parallel
"""

import os
import logging
from pathlib import Path
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import argparse
from datetime import datetime

def setup_logging(agent_id: int):
    """Set up logging for this agent."""
    log_dir = Path("agent_logs")
    log_dir.mkdir(exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - Agent[{agent_id}] - %(levelname)s - %(message)s'.format(agent_id=agent_id),
        datefmt='%Y-%m-%d %H:%M:%S'  # Specify a string date format
    )
    
    file_handler = logging.FileHandler(
        f"agent_logs/agent_{agent_id}_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

def update_config(cfg: DictConfig, params: dict, sweep_id: str, run_id: str, agent_id: str) -> DictConfig:
    """Update Hydra config with sweep parameters."""
    # Create a copy of the config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # First apply base config
    base_config = cfg_dict
    
    # Then overlay sweep parameters (ensuring they take precedence)
    for param_name, param_value in params.items():
        if param_name.startswith('+'):
            param_name = param_name[1:]
        update_nested(base_config, param_name, param_value)
    
    # Convert back to OmegaConf
    cfg = OmegaConf.create(base_config)
    
    cfg.training.checkpoint_dir = f"checkpoints/sweep_{sweep_id}/{run_id}"
    cfg.training.log_dir = f"logs/sweep_{sweep_id}/{run_id}"
    cfg.wandb.run_name = f"{run_id}"
    
    # Log updated config
    logging.info(f"Updated config:\n{OmegaConf.to_yaml(cfg)}")
    
    return cfg

def train_and_evaluate(cfg: DictConfig):
    """Train and evaluate an ensemble model with given configuration."""
    try:
        # Import here to avoid circular imports
        from train import main as train_eval
        import os
        
        # Convert wandb nested parameters to Hydra override format
        def convert_to_overrides(params, prefix=''):
            overrides = []
            for key, value in params.items():
                if isinstance(value, dict):
                    overrides.extend(convert_to_overrides(value, f"{prefix}{key}."))
                else:
                    # Handle boolean values specially
                    if isinstance(value, bool):
                        value = str(value).lower()
                    overrides.append(f"{prefix}{key}={value}")
            return overrides
        
        # Get all wandb parameters from the config
        wandb_params = OmegaConf.to_container(cfg, resolve=True)
        overrides = convert_to_overrides(wandb_params)
        
        # Set as Hydra overrides
        os.environ['HYDRA_FULL_ERROR'] = '1'
        os.environ['HYDRA_OVERRIDES'] = ' '.join(overrides)
        
        # Call the training function
        train_eval(cfg)
        
    except Exception as e:
        logging.error(f"Error during training/testing: {str(e)}")
        raise

def run_sweep(sweep_id: str, cfg: DictConfig, agent_id: int):
    """Run sweep agent."""
    def train_sweep():
        """Run one sweep trial."""
        # Initialize wandb run before accessing config
        run = wandb.init(
            project=os.environ.get('WANDB_PROJECT'),
            entity=os.environ.get('WANDB_ENTITY'),
            config=OmegaConf.to_container(cfg, resolve=True)  # Initialize with base config
        )
        
        # Each sweep trial gets a unique ID from wandb
        run_id = run.id
        
        # Get the merged config (base config + sweep values)
        sweep_config = {**OmegaConf.to_container(cfg, resolve=True), **run.config}
        sweep_cfg = OmegaConf.create(sweep_config)
        
        # Update paths for this specific sweep run
        sweep_cfg.training.checkpoint_dir = f"checkpoints/sweep_{sweep_id}/{run_id}"
        sweep_cfg.training.log_dir = f"logs/sweep_{sweep_id}/{run_id}"
        sweep_cfg.wandb.run_name = run_id
        
        try:
            # Train and evaluate with merged config
            train_and_evaluate(sweep_cfg)
        finally:
            # Finish the wandb run
            wandb.finish()
    
    # Run the agent
    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=1,
        project=os.environ.get('WANDB_PROJECT'),
        entity=os.environ.get('WANDB_ENTITY')
    )

def main():
    """Main function to run a sweep agent."""
    parser = argparse.ArgumentParser(description='Run a sweep agent')
    parser.add_argument('--config', type=str, required=True, help='Path to sweep config file')
    parser.add_argument('--sweep_id', type=str, required=True, help='W&B Sweep ID to join')
    parser.add_argument('--agent_id', type=int, required=True, help='Agent ID (usually SLURM array task ID)')
    args = parser.parse_args()
    
    # Set up logging for this agent
    setup_logging(args.agent_id)
    
    try:
        # Load base Hydra config
        with hydra.initialize(version_base=None, config_path="configs"):
            cfg = hydra.compose(config_name="config")
        
        # Run sweep
        run_sweep(args.sweep_id, cfg, args.agent_id)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
