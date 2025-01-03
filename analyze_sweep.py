#!/usr/bin/env python
"""
analyze_sweep.py - Script to analyze sweep results and find best configurations

Usage:
python analyze_sweep.py --sweep_id SWEEP_ID --entity ENTITY --project PROJECT
"""

import wandb
import pandas as pd
import argparse
from tabulate import tabulate
import sys
import logging

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_sweep_exists(api, entity, project, sweep_id):
    """Validate that the sweep exists before attempting to analyze it."""
    try:
        # Clean up sweep_id - remove any URL components if full URL was passed
        if '/' in sweep_id:
            # Handle full URLs like 'https://wandb.ai/entity/project/sweeps/sweep_id'
            parts = sweep_id.split('/')
            sweep_id = parts[-1]
            if parts[-2] == 'sweeps':
                # If URL format detected, also update entity and project if available
                if len(parts) >= 4:
                    entity = parts[-4]
                    project = parts[-3]
        
        sweep_path = f"{entity}/{project}/{sweep_id}"
        logging.info(f"Attempting to access sweep at: {sweep_path}")
        
        # Try to get the sweep
        sweep = api.sweep(sweep_path)
        logging.info(f"Successfully found sweep: {sweep_id}")
        return sweep
    except wandb.CommError as e:
        if "Could not find sweep" in str(e):
            logging.error(f"Sweep {sweep_id} not found in project {entity}/{project}")
            logging.error("Please check:")
            logging.error("1. The sweep ID is correct")
            logging.error("2. You have access to the project")
            logging.error("3. The entity and project names are correct")
        else:
            logging.error(f"Error connecting to W&B: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

def get_sweep_results(sweep_id, entity, project):
    """Get results from a wandb sweep."""
    api = wandb.Api()
    
    # Validate sweep exists
    sweep = validate_sweep_exists(api, entity, project, sweep_id)
    
    # Collect runs data
    runs_data = []
    total_runs = len(sweep.runs)
    finished_runs = 0
    
    logging.info(f"Processing {total_runs} runs...")
    
    for run in sweep.runs:
        if run.state == "finished":
            finished_runs += 1
            try:
                config = run.config
                summary = run.summary._json_dict
                
                # Flatten config dictionary
                flat_config = {}
                for k, v in config.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            flat_config[f"{k}.{sub_k}"] = sub_v
                    else:
                        flat_config[k] = v
                
                # Combine config and metrics
                run_data = {
                    'run_id': run.id,
                    'run_name': run.name,
                    'val_auc': summary.get('val_auc', None),
                    'val_pr_auc': summary.get('val_pr_auc', None),
                    **flat_config
                }
                runs_data.append(run_data)
            except Exception as e:
                logging.warning(f"Error processing run {run.id}: {e}")
    
    logging.info(f"Successfully processed {finished_runs} finished runs out of {total_runs} total runs")
    
    if not runs_data:
        logging.error("No valid runs found in the sweep")
        sys.exit(1)
        
    return pd.DataFrame(runs_data)

def analyze_sweep(df, metrics=['val_auc', 'val_pr_auc'], n_best=5):
    """Analyze sweep results and print best configurations."""
    print("\n=== Sweep Analysis ===\n")
    
    print(f"Total completed runs: {len(df)}")
    print("\nMetrics Summary:")
    for metric in metrics:
        if metric in df.columns:
            valid_values = df[metric].dropna()
            if not valid_values.empty:
                print(f"\n{metric}:")
                print(f"Mean: {valid_values.mean():.4f}")
                print(f"Std:  {valid_values.std():.4f}")
                print(f"Max:  {valid_values.max():.4f}")
                print(f"Min:  {valid_values.min():.4f}")
            else:
                print(f"\nNo valid values found for {metric}")
    
    print("\n=== Top Configurations ===")
    
    for metric in metrics:
        if metric in df.columns:
            print(f"\nTop {n_best} configurations by {metric}:")
            
            # Sort by metric, handling NaN values
            top_runs = df.dropna(subset=[metric]).nlargest(n_best, metric)
            
            if top_runs.empty:
                print(f"No valid runs found for {metric}")
                continue
            
            # Select relevant columns for display
            display_cols = ['run_name', 'run_id', metric, 'model_type']
            
            # Add architecture-specific columns
            if 'transformer_model.foundation_model' in df.columns:
                display_cols.extend(['transformer_model.foundation_model', 
                                   'transformer_model.freeze_backbone',
                                   'transformer_model.unfreeze_layers'])
            if 'graph_model.num_layers' in df.columns:
                display_cols.extend(['graph_model.num_layers',
                                   'graph_model.hidden_channels'])
            
            # Add shared parameters
            display_cols.extend(['learning_rate', 'batch_size', 'dropout'])
            
            # Only include columns that exist in the DataFrame
            display_cols = [col for col in display_cols if col in df.columns]
            
            print(tabulate(top_runs[display_cols], headers='keys', tablefmt='pipe', floatfmt='.4f'))
            
            # Print detailed config for the best run
            best_run = top_runs.iloc[0]
            print(f"\nBest configuration details for {metric} = {best_run[metric]:.4f}:")
            for col in df.columns:
                if col not in ['run_id', 'run_name', 'val_auc', 'val_pr_auc']:
                    print(f"{col}: {best_run[col]}")

def main():
    parser = argparse.ArgumentParser(description='Analyze wandb sweep results')
    parser.add_argument('--sweep_id', type=str, required=True, 
                       help='Sweep ID or full sweep URL (e.g., https://wandb.ai/entity/project/sweeps/sweep_id)')
    parser.add_argument('--entity', type=str, default='hani-goodarzi', help='WandB entity')
    parser.add_argument('--project', type=str, default='molmir_sweep', help='WandB project')
    parser.add_argument('--n_best', type=int, default=5, help='Number of best configs to show')
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # Get sweep results
        df = get_sweep_results(args.sweep_id, args.entity, args.project)
        
        # Analyze results
        analyze_sweep(df, n_best=args.n_best)
        
    except KeyboardInterrupt:
        logging.info("\nAnalysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
