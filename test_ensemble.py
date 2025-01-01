"""
Test script for evaluating ensemble MolMir models on held-out test set.

Examples:
    Test transformer ensemble:
        $ python test_ensemble.py model.architecture.type=transformer \
            test.ensemble.model_dir=checkpoints/ensemble/ChemBERTa-77M-MLM \
            test.ensemble.enabled=true

    Test GCN ensemble:
        $ python test_ensemble.py model.architecture.type=gcn \
            test.ensemble.model_dir=checkpoints/ensemble/gcn_model \
            test.ensemble.enabled=true
"""

import os
import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
import json
from datetime import datetime
import numpy as np
from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

from src.data import MolecularDataModule
from src.models.ensemble import MolMirEnsemble

def setup_logging(output_dir: str = "test_results") -> None:
    """Set up logging and create output directory."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def load_ensemble_metadata(ensemble_dir: Path) -> Dict:
    """Load ensemble configuration and metadata."""
    metadata_path = ensemble_dir / 'ensemble_metadata.json'
    if not metadata_path.exists():
        raise ValueError(f"Ensemble metadata not found at {metadata_path}")
        
    with open(metadata_path, 'r') as f:
        return json.load(f)

def get_model_params(metadata: Dict, cfg: DictConfig, num_features: int) -> dict:
    """Extract relevant model parameters from metadata and config."""
    # Get the model config from metadata
    model_config = metadata['config']['model']
    
    base_params = {
        'num_features': num_features,
        'learning_rate': model_config['training']['learning_rate'],
        'reg_weight': model_config['training']['reg_weight'],
        'cls_weight': model_config['training']['cls_weight'],
    }
    
    if metadata['model_type'] == 'transformer':
        base_params.update({
            'model_name': model_config['architecture']['foundation_model'],
            'hidden_dropout_prob': model_config['architecture']['hidden_dropout_prob'],
            'hidden_size': model_config['architecture']['hidden_size'],
            'freeze_backbone': model_config['architecture']['freeze_backbone'],
            'unfreeze_layers': model_config['architecture']['unfreeze_layers'],
            'trust_remote_code': model_config.get('trust_remote_code', True)
        })
    else:  # graph models
        base_params.update({
            'num_node_features': model_config['architecture']['graph']['num_node_features'],
            'hidden_channels': model_config['architecture']['graph']['hidden_channels'],
            'num_layers': model_config['architecture']['graph']['num_layers'],
            'dropout': model_config['architecture']['graph']['dropout']
        })
        
        # Add edge features for AttentiveFP
        if metadata['model_type'] == 'attentivefp':
            base_params['num_edge_features'] = model_config['architecture']['graph']['num_edge_features']
            
    return base_params

class EnsembleTestModule(pl.LightningModule):
    """Lightning module for ensemble testing."""
    
    def __init__(self, ensemble: MolMirEnsemble, features: List[str]):
        super().__init__()
        self.ensemble = ensemble
        self.features = features
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=['ensemble'])
        
    def forward(self, batch):
        return self.ensemble.predict_with_uncertainty(batch)
    
    def test_step(self, batch, batch_idx):
        results = self(batch)
        
        # Store predictions and targets
        outputs = {
            'reg_preds': results['regression_pred'],
            'reg_std': results['regression_std'],
            'cls_preds': results['classification_prob'],
            'cls_std': results['classification_std'],
            'reg_targets': batch['z_score'],
            'cls_targets': batch['is_hit'],
            'feature_idx': batch['feature_idx']
        }
        
        # Calculate batch metrics
        outputs['rmse'] = torch.sqrt(torch.mean((outputs['reg_preds'] - outputs['reg_targets']) ** 2))
        
        # Get actual batch size
        batch_size = outputs['reg_preds'].size(0)
        
        # Log batch metrics with explicit batch_size
        self.log('test_batch_rmse', outputs['rmse'], prog_bar=True, batch_size=batch_size)
        self.log('test_batch_reg_std', torch.mean(outputs['reg_std']), prog_bar=True, batch_size=batch_size)
        self.log('test_batch_cls_std', torch.mean(outputs['cls_std']), prog_bar=True, batch_size=batch_size)
        
        self.test_step_outputs.append(outputs)
        return outputs
    
    def on_test_epoch_end(self):
        # Gather all outputs
        all_reg_preds = torch.cat([x['reg_preds'] for x in self.test_step_outputs])
        all_cls_preds = torch.cat([x['cls_preds'] for x in self.test_step_outputs])
        all_reg_targets = torch.cat([x['reg_targets'] for x in self.test_step_outputs])
        all_cls_targets = torch.cat([x['cls_targets'] for x in self.test_step_outputs])
        all_feature_idx = torch.cat([x['feature_idx'] for x in self.test_step_outputs])
        all_reg_std = torch.cat([x['reg_std'] for x in self.test_step_outputs])
        all_cls_std = torch.cat([x['cls_std'] for x in self.test_step_outputs])
        
        # Calculate overall metrics
        metrics = {}
        
        # Regression metrics
        rmse = torch.sqrt(torch.mean((all_reg_preds - all_reg_targets) ** 2)).item()
        self.log('test_rmse', rmse)
        metrics['test_rmse'] = rmse
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(
            all_reg_preds.cpu().numpy(),
            all_reg_targets.cpu().numpy()
        )
        self.log('test_pearson_r', pearson_r)
        self.log('test_pearson_p', pearson_p)
        metrics['test_pearson_r'] = pearson_r
        metrics['test_pearson_p'] = pearson_p
        
        # Classification metrics
        cls_preds_np = all_cls_preds.cpu().numpy()
        cls_targets_np = all_cls_targets.cpu().numpy()
        auc = roc_auc_score(cls_targets_np, cls_preds_np)
        pr_auc = average_precision_score(cls_targets_np, cls_preds_np)
        self.log('test_auc', auc)
        self.log('test_pr_auc', pr_auc)
        metrics['test_auc'] = auc
        metrics['test_pr_auc'] = pr_auc
        
        # Uncertainty metrics
        mean_reg_uncertainty = torch.mean(all_reg_std).item()
        std_reg_uncertainty = torch.std(all_reg_std).item()
        mean_cls_uncertainty = torch.mean(all_cls_std).item()
        std_cls_uncertainty = torch.std(all_cls_std).item()
        
        self.log('test_mean_reg_uncertainty', mean_reg_uncertainty)
        self.log('test_std_reg_uncertainty', std_reg_uncertainty)
        self.log('test_mean_cls_uncertainty', mean_cls_uncertainty)
        self.log('test_std_cls_uncertainty', std_cls_uncertainty)
        
        metrics.update({
            'test_mean_reg_uncertainty': mean_reg_uncertainty,
            'test_std_reg_uncertainty': std_reg_uncertainty,
            'test_mean_cls_uncertainty': mean_cls_uncertainty,
            'test_std_cls_uncertainty': std_cls_uncertainty
        })
        
        # Calculate per-feature metrics
        num_features_with_metrics = 0
        for feat_idx, feature_name in enumerate(self.features):
            mask = (all_feature_idx == feat_idx)
            if not torch.any(mask):
                continue
                
            num_features_with_metrics += 1
            prefix = f'test_feature_{feat_idx}'
            
            # Store feature name
            metrics[f'{prefix}_name'] = feature_name
            
            # Feature regression metrics
            feat_reg_preds = all_reg_preds[mask]
            feat_reg_targets = all_reg_targets[mask]
            feat_reg_std = all_reg_std[mask]
            
            feat_rmse = torch.sqrt(torch.mean((feat_reg_preds - feat_reg_targets) ** 2)).item()
            feat_reg_uncertainty = torch.mean(feat_reg_std).item()
            
            metrics[f'{prefix}_rmse'] = feat_rmse
            metrics[f'{prefix}_reg_uncertainty'] = feat_reg_uncertainty
            
            # Log feature metrics
            self.log(f'{prefix}_rmse', feat_rmse)
            self.log(f'{prefix}_reg_uncertainty', feat_reg_uncertainty)
            
            # Feature Pearson correlation
            if len(feat_reg_preds) > 1:  # Need at least 2 points for correlation
                pearson_r, pearson_p = stats.pearsonr(
                    feat_reg_preds.cpu().numpy(),
                    feat_reg_targets.cpu().numpy()
                )
                metrics[f'{prefix}_pearson_r'] = pearson_r
                metrics[f'{prefix}_pearson_p'] = pearson_p
                self.log(f'{prefix}_pearson_r', pearson_r)
            
            # Feature classification metrics
            feat_cls_preds = all_cls_preds[mask]
            feat_cls_targets = all_cls_targets[mask]
            feat_cls_std = all_cls_std[mask]
            
            # Only calculate classification metrics if we have both positive and negative examples
            if torch.any(feat_cls_targets > 0) and torch.any(feat_cls_targets == 0):
                try:
                    feat_auc = roc_auc_score(
                        feat_cls_targets.cpu().numpy(),
                        feat_cls_preds.cpu().numpy()
                    )
                    feat_pr_auc = average_precision_score(
                        feat_cls_targets.cpu().numpy(),
                        feat_cls_preds.cpu().numpy()
                    )
                    
                    metrics[f'{prefix}_auc'] = feat_auc
                    metrics[f'{prefix}_pr_auc'] = feat_pr_auc
                    metrics[f'{prefix}_cls_uncertainty'] = torch.mean(feat_cls_std).item()
                    
                    self.log(f'{prefix}_auc', feat_auc)
                    self.log(f'{prefix}_pr_auc', feat_pr_auc)
                except Exception as e:
                    logging.warning(f"Could not calculate classification metrics for feature {feature_name}: {e}")
        
        # Store number of features with metrics
        metrics['num_features_with_metrics'] = num_features_with_metrics
        self.log('num_features_with_metrics', num_features_with_metrics)
        
        self.test_step_outputs.clear()
        
        # Return metrics dict
        return metrics

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

def save_results(
    metrics: Dict[str, float],
    ensemble_dir: Path,
    model_identifier: str,
    output_dir: str = "test_results"
) -> None:
    """Save test results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path(output_dir) / f"ensemble_test_results_{timestamp}"
    
    # Convert numpy types to Python native types
    metrics = convert_numpy_types(metrics)
    
    # Save all metrics as JSON
    with open(f"{results_base}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create feature metrics DataFrame
    feature_metrics = {}
    for key, value in metrics.items():
        if key.startswith('test_feature_'):
            parts = key.split('_')
            feature_idx = parts[-2] if parts[-1] in ['name', 'rmse', 'auc', 'pr_auc', 'pearson_r', 'pearson_p'] else parts[-1]
            metric_name = parts[-1]
            
            if feature_idx not in feature_metrics:
                feature_metrics[feature_idx] = {}
            feature_metrics[feature_idx][metric_name] = value
    
    if feature_metrics:
        feature_df = pd.DataFrame.from_dict(feature_metrics, orient='index')
        feature_df.to_csv(f"{results_base}_feature_metrics.csv")
    
    logging.info(f"\nSaved test results to:")
    logging.info(f"- {results_base}_metrics.json")
    logging.info(f"- {results_base}_feature_metrics.csv")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main test function for ensemble models."""
    setup_logging()
    
    try:
        if not cfg.test.ensemble.enabled:
            raise ValueError("Ensemble testing is not enabled. Set test.ensemble.enabled=true")
            
        # Load ensemble metadata
        ensemble_dir = Path(cfg.test.ensemble.model_dir)
        if not ensemble_dir.exists():
            raise ValueError(f"Ensemble directory not found: {ensemble_dir}")
            
        metadata = load_ensemble_metadata(ensemble_dir)
        model_type = metadata['model_type']
        model_identifier = metadata['model_identifier']
        
        logging.info(f"Testing ensemble model: {model_identifier}")
        logging.info(f"Model type: {model_type}")
        logging.info(f"Ensemble directory: {ensemble_dir}")
        
        # Initialize data module
        data_module = MolecularDataModule(
            smiles_path=cfg.data.smiles_path,
            activity_path=cfg.data.activity_path,
            model_type=model_type,
            model_name=metadata['config']['model']['architecture'].get('foundation_model', None),
            batch_size=cfg.training.batch_size,
            num_workers=4,
            prefetch_factor=cfg.data.dataloader.prefetch_factor,
            persistent_workers=cfg.data.dataloader.persistent_workers,
            pin_memory=cfg.data.dataloader.pin_memory,
            max_length=metadata['config']['model']['architecture'].get('max_length', 512),
            z_score_threshold=cfg.model.z_score_threshold
        )
        
        # Setup data
        data_module.setup()
        features = data_module.train_dataset.dataset.features
        
        logging.info(f"\nDataset loaded:")
        logging.info(f"Number of features: {len(features)}")
        logging.info(f"Test samples: {len(data_module.test_dataset)}")
        
        # Get model parameters
        model_params = get_model_params(metadata, cfg, data_module.num_features)
        
        # Initialize ensemble
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble = MolMirEnsemble(
            model_type=model_type,
            model_params=model_params,
            num_folds=metadata['num_folds'],
            ensemble_dir=ensemble_dir,
            device=device
        )
        
        # Load ensemble models
        ensemble.load_models()
        logging.info(f"Loaded {len(ensemble.models)} models for ensemble")
        
        # Initialize test module
        test_module = EnsembleTestModule(ensemble=ensemble, features=features)
        
        # Create trainer
        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            logger=False,
            enable_checkpointing=False,
            default_root_dir=str(ensemble_dir / "test_logs")
        )
        
        # Run test
        logging.info("\nStarting test evaluation...")
        test_metrics = trainer.test(test_module, datamodule=data_module)[0]
        
        # Save results
        save_results(test_metrics, ensemble_dir, model_identifier, output_dir=ensemble_dir)
        
        # Print summary
        logging.info("\nTest Results Summary:")
        logging.info(f"Overall RMSE: {test_metrics['test_rmse']:.3f}")
        logging.info(f"Overall AUC: {test_metrics['test_auc']:.3f}")
        logging.info(f"Overall PR-AUC: {test_metrics['test_pr_auc']:.3f}")
        logging.info(f"Overall Pearson r: {test_metrics['test_pearson_r']:.3f} "
                    f"(p={test_metrics['test_pearson_p']:.3e})")

        # Print uncertainty summary
        logging.info(f"\nUncertainty Summary:")
        if 'test_mean_reg_uncertainty' in test_metrics:
            logging.info(f"Mean Regression Uncertainty: {test_metrics['test_mean_reg_uncertainty']:.3f} "
                        f"± {test_metrics['test_std_reg_uncertainty']:.3f}")
        if 'test_mean_cls_uncertainty' in test_metrics:
            logging.info(f"Mean Classification Uncertainty: {test_metrics['test_mean_cls_uncertainty']:.3f} "
                        f"± {test_metrics['test_std_cls_uncertainty']:.3f}")
        
        # Feature summary
        num_features = test_metrics.get('num_features_with_metrics', 0)
        logging.info(f"\nFeature Summary:")
        logging.info(f"Features with metrics: {num_features}")
        
        # Collect feature metrics
        feature_metrics = {}
        for feat_idx in range(len(features)):
            prefix = f'test_feature_{feat_idx}'
            
            # Skip if no metrics for this feature
            if f'{prefix}_rmse' not in test_metrics:
                continue
                
            feature_metrics[feat_idx] = {
                'name': test_metrics.get(f'{prefix}_name', ''),
                'rmse': test_metrics.get(f'{prefix}_rmse', None),
                'auc': test_metrics.get(f'{prefix}_auc', None),
                'pr_auc': test_metrics.get(f'{prefix}_pr_auc', None),
                'reg_uncertainty': test_metrics.get(f'{prefix}_reg_uncertainty', None),
                'cls_uncertainty': test_metrics.get(f'{prefix}_cls_uncertainty', None)
            }
        
        # Print summary statistics for features
        if feature_metrics:
            logging.info("\nFeature Metrics Summary:")
            
            # RMSE statistics
            rmse_values = [m['rmse'] for m in feature_metrics.values() if m['rmse'] is not None]
            if rmse_values:
                logging.info(f"RMSE: {np.mean(rmse_values):.3f} ± {np.std(rmse_values):.3f}")
            
            # AUC statistics
            auc_values = [m['auc'] for m in feature_metrics.values() if m['auc'] is not None]
            if auc_values:
                logging.info(f"AUC: {np.mean(auc_values):.3f} ± {np.std(auc_values):.3f}")
            
            # PR-AUC statistics
            pr_auc_values = [m['pr_auc'] for m in feature_metrics.values() if m['pr_auc'] is not None]
            if pr_auc_values:
                logging.info(f"PR-AUC: {np.mean(pr_auc_values):.3f} ± {np.std(pr_auc_values):.3f}")
            
            # Uncertainty statistics
            reg_uncert = [m['reg_uncertainty'] for m in feature_metrics.values() if m['reg_uncertainty'] is not None]
            cls_uncert = [m['cls_uncertainty'] for m in feature_metrics.values() if m['cls_uncertainty'] is not None]
            
            if reg_uncert:
                logging.info(f"Regression Uncertainty: {np.mean(reg_uncert):.3f} ± {np.std(reg_uncert):.3f}")
            if cls_uncert:
                logging.info(f"Classification Uncertainty: {np.mean(cls_uncert):.3f} ± {np.std(cls_uncert):.3f}")
        
    except Exception as e:
        logging.error(f"Error during ensemble testing: {str(e)}")
        raise

if __name__ == '__main__':
    main()
