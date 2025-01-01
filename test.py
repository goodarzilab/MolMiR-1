"""
Test script for evaluating MolMir model on held-out test set.

Examples:
    ChemBERTa model testing:
        $ python test.py model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM \
            "+test.checkpoint_path='checkpoints/molmir_ChemBERTa-77M-MLM_epoch=01_lossval_loss=1.33_aucval_auc=0.556_praucval_pr_auc=0.041.ckpt'"

    MPNN model testing:
        $ python test.py model.architecture.type=mpnn \
            "+test.checkpoint_path='checkpoints/molmir_mpnn_model_epoch=03_lossval_loss=1.32_aucval_auc=0.633_praucval_pr_auc=0.064.ckpt'"

    GCN model testing:
        $ python test.py model.architecture.type=gcn \
            "+test.checkpoint_path='checkpoints/molmir_gcn_model_epoch=01_lossval_loss=1.99_aucval_auc=0.648_praucval_pr_auc=0.065.ckpt'"
"""

import os
import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
import pandas as pd
import json
from datetime import datetime
import numpy as np
from scipy import stats
from typing import Dict, Any
import traceback

from src.data import MolecularDataModule
from src.models import MolMir
from src.models.graph_models import (
    MolMirGCN, 
    MolMirMPNN, 
    MolMirGIN, 
    MolMirAttentiveFP
)

def setup_logging(output_dir: str = "test_results") -> None:
    """Set up logging and create output directory."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def load_split_info() -> Dict[str, Any]:
    """Load drug split information from cache."""
    split_info_path = Path("cache/split_info.json")
    
    if not split_info_path.exists():
        logging.error(f"Split information not found at {split_info_path}")
        logging.error("Please run training first to generate the split")
        sys.exit(1)
    
    try:
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        
        logging.info("\nLoaded Split Information:")
        logging.info(f"Total drugs: {split_info.get('total_drugs', 'N/A')}")
        logging.info(f"Train drugs: {len(split_info.get('train_drugs', []))}")
        logging.info(f"Validation drugs: {len(split_info.get('val_drugs', []))}")
        logging.info(f"Test drugs: {len(split_info.get('test_drugs', []))}")
        
        return split_info
    except Exception as e:
        logging.error(f"Error loading split info: {e}")
        raise

def verify_cache(model_type: str) -> Dict:
    """Verify cache file for the specific model type and return cached data."""
    cache_path = Path(f"cache/processed_dataset_{model_type}.pt")
    logging.info(f"Checking for cache file at: {cache_path.absolute()}")
    
    if not cache_path.exists():
        logging.error(f"Cache file not found: {cache_path}")
        logging.error(f"Please run training first with model type: {model_type}")
        sys.exit(1)
    
    try:
        # Load and verify cache contents
        cached_data = torch.load(cache_path)
        logging.info("\nCache File Contents:")
        for key, value in cached_data.items():
            logging.info(f"{key}: {type(value)}")
            if hasattr(value, '__len__'):
                logging.info(f"Length: {len(value)}")
                
        # Verify model type matches
        if 'model_type' in cached_data and cached_data['model_type'] != model_type:
            logging.warning(f"Cache file model type '{cached_data['model_type']}' "
                          f"does not match requested model type '{model_type}'")
                          
        # Verify dataset exists
        if 'dataset' not in cached_data:
            logging.error("Cache file does not contain dataset")
            sys.exit(1)
            
        return cached_data
        
    except Exception as e:
        logging.error(f"Error reading cache file: {e}")
        traceback.print_exc()
        sys.exit(1)

def analyze_feature_metrics(model_metrics: Dict) -> Dict:
    """Compute aggregate metrics across features."""
    # Initialize metric collectors
    rmses = []
    aucs = []
    pr_aucs = []
    pearson_rs = []
    pearson_ps = []
    feature_indices = set()
    
    # Parse metrics keys
    for key, value in model_metrics.items():
        try:
            if not isinstance(key, str) or not key.startswith('test_feature_'):
                continue
            
            parts = key.split('_')
            if len(parts) < 4:
                continue
            
            # Extract feature index (always last part)
            feature_idx = int(parts[-1])
            feature_indices.add(feature_idx)
            
            # Handle each metric type
            if 'rmse' in parts:
                rmses.append(float(value))
            elif 'auc' in parts and 'pr' not in parts:
                aucs.append(float(value))
            elif 'pr_auc' in key:
                pr_aucs.append(float(value))
            elif 'pearson' in parts:
                if parts[-2] == 'r':  # test_feature_pearson_r_idx
                    pearson_rs.append(float(value))
                elif parts[-2] == 'p':  # test_feature_pearson_p_idx
                    pearson_ps.append(float(value))
                
        except Exception as e:
            logging.error(f"Error processing key {key}: {str(e)}")
            continue
    
    # Summary logging
    logging.info(f"\nMetric Collection Summary:")
    logging.info(f"Total features found: {len(feature_indices)}")
    logging.info(f"RMSE metrics: {len(rmses)}")
    logging.info(f"AUC metrics: {len(aucs)}")
    logging.info(f"PR-AUC metrics: {len(pr_aucs)}")
    logging.info(f"Pearson r metrics: {len(pearson_rs)}")
    logging.info(f"Pearson p metrics: {len(pearson_ps)}")
    
    # Compute aggregated metrics
    aggregate_metrics = {
        'mean_feature_pearson_r': np.mean(pearson_rs) if pearson_rs else None,
        'std_feature_pearson_r': np.std(pearson_rs) if pearson_rs else None,
        'mean_feature_pearson_p': np.mean(pearson_ps) if pearson_ps else None,
        'mean_feature_rmse': np.mean(rmses) if rmses else None,
        'mean_feature_auc': np.mean(aucs) if aucs else None,
        'mean_feature_pr_auc': np.mean(pr_aucs) if pr_aucs else None,
        'num_features_analyzed': len(feature_indices)
    }
    
    # Add individual feature counts
    aggregate_metrics.update({
        'num_features_with_rmse': len(rmses),
        'num_features_with_auc': len(aucs),
        'num_features_with_pr_auc': len(pr_aucs),
        'num_features_with_pearson': len(pearson_rs)
    })
    
    return aggregate_metrics

def save_results(metrics: dict, model_basename: str, split_info: dict) -> None:
    """Save test results to files."""
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = results_dir / f"test_results_{model_basename}_{timestamp}"
    
    # Combine metrics with split information
    full_results = {
        'test_metrics': metrics,
        'split_info': split_info
    }
    
    # Save overall metrics
    with open(f"{results_base}_full_results.json", 'w') as f:
        json.dump(full_results, f, indent=2)
    
    # Save feature metrics as CSV if they exist
    if 'test_metrics' in metrics and isinstance(metrics['test_metrics'], dict):
        feature_df = pd.DataFrame.from_dict(metrics['test_metrics'], orient='index')
        feature_df.to_csv(f"{results_base}_feature_metrics.csv")
    
    logging.info(f"\nSaved test results to:")
    logging.info(f"- {results_base}_full_results.json")
    logging.info(f"- {results_base}_feature_metrics.csv")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main test function."""
    setup_logging()
    
    try:
        # Load split information
        split_info = load_split_info()
        
        # Get model type and verify cache
        model_type = cfg.model.architecture.type
        logging.info(f"Testing model type: {model_type}")
        cached_data = verify_cache(model_type)
        
        num_features = cached_data['num_features']
        logging.info(f"Number of features: {num_features}")
        
        # Model mapping
        model_classes = {
            'transformer': MolMir,
            'gcn': MolMirGCN,
            'mpnn': MolMirMPNN,
            'gin': MolMirGIN,
            'attentivefp': MolMirAttentiveFP
        }
        
        # Select appropriate model class
        ModelClass = model_classes.get(model_type)
        if ModelClass is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize model based on type
        try:
            if model_type == 'transformer':
                logging.info(f"Initializing transformer model with {cfg.model.architecture.foundation_model}")
                model = MolMir(
                    model_name=cfg.model.architecture.foundation_model,
                    num_features=num_features,
                    learning_rate=cfg.training.learning_rate,
                    hidden_dropout_prob=cfg.model.architecture.hidden_dropout_prob,
                    hidden_size=cfg.model.architecture.hidden_size,
                    freeze_backbone=cfg.model.architecture.freeze_backbone,
                    unfreeze_layers=cfg.model.architecture.unfreeze_layers
                )
            else:
                # Get the appropriate graph model class
                graph_model_classes = {
                    'gcn': MolMirGCN,
                    'mpnn': MolMirMPNN,
                    'gin': MolMirGIN,
                    'attentivefp': MolMirAttentiveFP
                }
                GraphModelClass = graph_model_classes.get(model_type)
                if GraphModelClass is None:
                    raise ValueError(f"Unsupported graph model type: {model_type}")
                
                logging.info(f"Initializing {model_type} model")
                model = GraphModelClass(
                    num_features=num_features,
                    num_node_features=cfg.model.architecture.graph.num_node_features,
                    hidden_channels=cfg.model.architecture.graph.hidden_channels,
                    num_layers=cfg.model.architecture.graph.num_layers,
                    dropout=cfg.model.architecture.graph.dropout,
                    learning_rate=cfg.training.learning_rate
                )
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise


        # Create data module after model initialization
        data_module = MolecularDataModule(
            smiles_path=cfg.data.smiles_path,
            activity_path=cfg.data.activity_path,
            model_type=model_type,
            model_name=cfg.model.architecture.get('foundation_model', None),
            batch_size=cfg.training.batch_size,
            num_workers=4,
            prefetch_factor=cfg.data.dataloader.prefetch_factor,
            persistent_workers=cfg.data.dataloader.persistent_workers,
            pin_memory=cfg.data.dataloader.pin_memory,
            max_length=cfg.model.architecture.get('max_length', 512),
            z_score_threshold=cfg.model.z_score_threshold,
            cache_only=False
        )
        # Set cache properties after initialization
        data_module.cache_only = True
        data_module.model_type = model_type
            
        # Verify checkpoint path
        if not os.path.exists(cfg.test.checkpoint_path):
            raise ValueError(f"Checkpoint not found: {cfg.test.checkpoint_path}")
        
        logging.info(f"Loading model from {cfg.test.checkpoint_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(cfg.test.checkpoint_path)
            logging.info("Checkpoint keys:")
            for key in checkpoint.keys():
                logging.info(f"- {key}")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            raise
        
        # Load state dict with proper error handling
        try:
            new_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key in model.state_dict():
                    new_state_dict[key] = value
                else:
                    logging.warning(f"Skipping key not in model: {key}")
            
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
        except Exception as e:
            logging.error(f"Error loading state dict: {e}")
            raise

        # Create trainer
        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            logger=False,
            enable_checkpointing=False,
        )

        # Run test with proper error handling
        try:
            logging.info("\nStarting test evaluation...")
            results = trainer.test(model, data_module)
            
            if not results:
                raise ValueError("No results returned from test")
                
            metrics = results[0]

            logging.info("\nRaw Test Results:")
            for key, value in metrics.items():
                try:
                    if isinstance(value, (int, float)):
                        logging.info(f"{key}: {value}")
                except Exception as e:
                    logging.error(f"Error printing metric {key}: {str(e)}")
        
            # Analyze feature-level metrics
            logging.info("\nAnalyzing feature-level metrics...")
            feature_aggregate_metrics = analyze_feature_metrics(metrics)
            metrics.update(feature_aggregate_metrics)
        
            # Save results
            model_basename = (
                cfg.model.architecture.foundation_model.split('/')[-1] 
                if model_type == 'transformer' 
                else model_type
            )
            save_results(metrics, model_basename, split_info)
        
        except Exception as e:
            logging.error(f"Error during testing: {e}")
            raise

        # Print comprehensive summary
        logging.info("\nTest Results Summary:")
        logging.info(f"Model Type: {model_type}")
        
        # Overall metrics with proper formatting
        def format_metric(value):
            return f"{value:.3f}" if value is not None else "N/A"
            
        # Overall performance
        logging.info("\nOverall Performance:")
        logging.info(f"Overall RMSE: {format_metric(metrics.get('test_rmse'))}")
        logging.info(f"Overall AUC: {format_metric(metrics.get('test_auc'))}")
        logging.info(f"Overall PR-AUC: {format_metric(metrics.get('test_pr_auc'))}")

        # Overall Pearson correlation
        if 'test_pearson_r' in metrics:
            logging.info(f"Overall Pearson r: {format_metric(metrics['test_pearson_r'])} "
                        f"(p={format_metric(metrics.get('test_pearson_p', 0))})")
        
        # Feature-level metrics
        logging.info("\nFeature Metrics Distribution:")
        num_features = metrics.get('num_features_analyzed', 0)
        logging.info(f"Total Features Analyzed: {num_features}")
        
        # RMSE metrics
        if metrics.get('mean_feature_rmse') is not None:
            logging.info("\nRMSE Metrics:")
            logging.info(f"Mean Feature RMSE: {format_metric(metrics['mean_feature_rmse'])}")
            logging.info(f"Features with RMSE: {metrics.get('num_features_with_rmse', 0)}")
        
        # AUC metrics
        if metrics.get('mean_feature_auc') is not None:
            logging.info("\nAUC Metrics:")
            logging.info(f"Mean Feature AUC: {format_metric(metrics['mean_feature_auc'])}")
            logging.info(f"Features with AUC: {metrics.get('num_features_with_auc', 0)}")
        
        # PR-AUC metrics
        if metrics.get('mean_feature_pr_auc') is not None:
            logging.info("\nPR-AUC Metrics:")
            logging.info(f"Mean Feature PR-AUC: {format_metric(metrics['mean_feature_pr_auc'])}")
            logging.info(f"Features with PR-AUC: {metrics.get('num_features_with_pr_auc', 0)}")
        
        # Pearson correlation metrics
        if metrics.get('mean_feature_pearson_r') is not None:
            logging.info("\nPearson Correlation Metrics:")
            logging.info(f"Mean Feature Pearson r: {format_metric(metrics['mean_feature_pearson_r'])} "
                      f"± {format_metric(metrics.get('std_feature_pearson_r', 0))}")
            logging.info(f"Features with Pearson: {metrics.get('num_features_with_pearson', 0)}")
            
            if metrics.get('mean_feature_pearson_p') is not None:
                logging.info(f"Mean Feature Pearson p-value: "
                           f"{format_metric(metrics['mean_feature_pearson_p'])}")
        
        # Performance summary
        logging.info("\nPerformance Summary:")
        logging.info(f"RMSE: {format_metric(metrics.get('test_rmse'))} (Overall) vs "
                    f"{format_metric(metrics.get('mean_feature_rmse'))} (Feature Mean)")
        logging.info(f"AUC: {format_metric(metrics.get('test_auc'))} (Overall) vs "
                    f"{format_metric(metrics.get('mean_feature_auc'))} (Feature Mean)")
        logging.info(f"PR-AUC: {format_metric(metrics.get('test_pr_auc'))} (Overall) vs "
                    f"{format_metric(metrics.get('mean_feature_pr_auc'))} (Feature Mean)")

    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
