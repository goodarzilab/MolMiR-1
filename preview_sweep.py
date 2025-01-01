#!/usr/bin/env python
"""
preview_sweep.py - Preview all commands that would be run in the sweep
"""

import yaml
import itertools
from typing import Dict, List

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten a nested dictionary with dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            if 'value' in v:
                items.append((new_key, [v['value']]))
            elif 'values' in v:
                items.append((new_key, v['values']))
            else:
                items.extend(flatten_dict(v, new_key, sep=sep).items())
    return dict(items)

def get_value_combinations(params: Dict) -> List[Dict]:
    """Get all possible combinations of parameter values."""
    # Flatten the nested parameters
    flat_params = flatten_dict(params)
    
    # Get parameter names and their possible values
    param_names = list(flat_params.keys())
    param_values = [flat_params[param] for param in param_names]
    
    # Generate all combinations
    value_combinations = list(itertools.product(*param_values))
    
    # Convert to list of dictionaries
    return [
        {param: value for param, value in zip(param_names, combo)}
        for combo in value_combinations
    ]

def format_command(base_cmd: str, params: Dict) -> str:
    """Format a command with given parameters."""
    param_strs = []
    for param, value in params.items():
        # Handle boolean values
        if isinstance(value, bool):
            value = str(value).lower()
        param_strs.append(f"{param}={value}")
    return f"{base_cmd} \\\n    " + " \\\n    ".join(param_strs)

def main():
    # Load sweep config
    with open('configs/sweep.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)['sweep']

    # Get all parameter combinations
    combinations = get_value_combinations(sweep_config['parameters'])
    
    # Print summary
    print(f"\nFound {len(combinations)} possible combinations")
    print(f"Each combination will run with {sweep_config['parameters'].get('training.ensemble.num_folds', {}).get('value', 'unknown')} folds")
    print()
    
    # Print each command
    for i, params in enumerate(combinations, 1):
        print(f"Command {i}/{len(combinations)}:")
        cmd = format_command("python train_ensemble.py", params)
        print(cmd)
        print()

if __name__ == '__main__':
    main()
