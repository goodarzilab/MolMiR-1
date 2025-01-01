#!/usr/bin/env python
"""
preview_wandb_runs.py - Preview how wandb runs and checkpoints will be organized
"""

import yaml
import itertools
from typing import Dict, List
import pprint
from pathlib import Path
from src.utils.name_generator import generate_name  # Import the name generator

def get_parameter_combinations(params: Dict) -> List[Dict]:
    """Get all possible combinations of parameter values."""
    param_values = {}
    for param_path, param_info in params.items():
        if isinstance(param_info, dict):
            if 'values' in param_info:
                param_values[param_path] = param_info['values']
            elif 'value' in param_info:
                param_values[param_path] = [param_info['value']]

    # Get names of parameters and their possible values
    param_names = list(param_values.keys())
    value_combinations = list(itertools.product(
        *[param_values[param] for param in param_names]
    ))

    # Convert to list of dictionaries
    return [
        {param: value for param, value in zip(param_names, combo)}
        for combo in value_combinations
    ]

def get_model_identifier(params: Dict) -> str:
    """Get model directory name based on architecture type."""
    if params.get('model.architecture.type') == 'transformer':
        foundation_model = params.get('model.architecture.foundation_model', '')
        return foundation_model.split('/')[-1]
    else:
        return f"{params.get('model.architecture.type')}_model"

def extract_wandb_params(sweep_config: Dict) -> Dict:
    """Extract wandb-related parameters from sweep config."""
    wandb_params = {}
    
    # Known wandb-related keys to look for
    wandb_keys = [
        'project', 'entity', 'name', 'group', 
        'tags', 'notes', 'config'
    ]
    
    # Check for wandb parameters in sweep config
    for key in wandb_keys:
        if f'wandb.{key}' in sweep_config['parameters']:
            param_info = sweep_config['parameters'][f'wandb.{key}']
            if 'values' in param_info:
                wandb_params[f'wandb.{key}'] = param_info['values']
            elif 'value' in param_info:
                wandb_params[f'wandb.{key}'] = [param_info['value']]
    
    return wandb_params

def main():
    # Load sweep config
    with open('configs/sweep.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)['sweep']

    # Try to read the current sweep ID 
    try:
        with open('current_sweep_id.txt', 'r') as f:
            current_sweep_id = f.read().strip()
    except FileNotFoundError:
        # If no sweep ID file exists, generate a placeholder
        current_sweep_id = "SWEEP_ID_NOT_FOUND"

    # Get all parameter combinations
    combinations = get_parameter_combinations(sweep_config['parameters'])
    
    # Extract wandb parameters
    wandb_params = extract_wandb_params(sweep_config)
    
    num_folds = next(c['training.ensemble.num_folds'] 
                    for c in combinations 
                    if 'training.ensemble.num_folds' in c)

    print(f"\n🔍 Sweep Analysis")
    print(f"=" * 80)
    print(f"Sweep ID: {current_sweep_id}")
    print(f"Sweep URL: https://wandb.ai/hani-goodarzi/molmir_sweep/sweeps/{current_sweep_id}")
    print(f"\nFound {len(combinations)} possible parameter combinations")
    print(f"Each combination will run {num_folds} folds")
    print(f"Total expected wandb runs: {len(combinations) * num_folds}")
    print("\nExample wandb and checkpoint structure:")
    
    # Show structure for first few combinations
    for i, params in enumerate(combinations):  # Limit to first 3 for brevity
        # Use name generator for base name
        base_name = generate_name()
        run_id = current_sweep_id  # Use actual sweep ID
        model_id = get_model_identifier(params)
        
        print(f"\n📊 Parameter combination {i+1}:")
        print("-" * 80)
        
        # Show wandb structure
        print("🌐 WandB Structure:")
        print(f"Project: molmir_sweep")
        print(f"Sweep ID: {current_sweep_id}")
        print(f"Group: ensemble_{base_name}")
        for fold in range(num_folds):
            print(f"  Run: {base_name}_fold_{fold}")
        
        # Show checkpoint structure
        print("\n💾 Checkpoint Structure:")
        checkpoint_base = Path(f"checkpoints/sweep_{run_id}/ensemble/{base_name}/{model_id}")
        print(f"{checkpoint_base}/")
        for fold in range(num_folds):
            print(f"    ├── fold_{fold}/")
            print(f"    │   └── best model checkpoints")
        print(f"    └── ensemble_metadata.json")
        
        # Show parameters
        print("\n⚙️ Model Parameters:")
        model_params = {k: v for k, v in params.items() if k.startswith('model.')}
        pprint.pprint(model_params, indent=2)
        
        print("\n⚙️ Training Parameters:")
        training_params = {k: v for k, v in params.items() if k.startswith('training.')}
        pprint.pprint(training_params, indent=2)
        
        print("\n⚙️ Wandb Parameters:")
        sweep_wandb_params = {k: v for k, v in wandb_params.items()}
        pprint.pprint(sweep_wandb_params, indent=2)

if __name__ == '__main__':
    main()
