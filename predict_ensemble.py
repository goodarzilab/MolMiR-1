"""
Prediction script for ensemble MolMir models.
Takes SMILES input and returns predictions using a trained ensemble model.

Examples:
    Predict with transformer ensemble:
        $ python predict_ensemble.py model.architecture.type=transformer \
            predict.ensemble.model_dir=checkpoints/ensemble/ChemBERTa-77M-MLM \
            predict.ensemble.enabled=true \
            predict.smiles_file=test/test_SMILES.tsv

    Predict with GCN ensemble:
        $ python predict_ensemble.py model.architecture.type=gcn \
            predict.ensemble.model_dir=checkpoints/ensemble/gcn_model \
            predict.ensemble.enabled=true \
            predict.smiles_file=test/test_SMILES.tsv
"""

import os
# Set tokenizers parallelism before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.predict_data import PredictionDataModule
from src.models.ensemble import MolMirEnsemble

def setup_logging(output_dir: str = "predictions") -> None:
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
        metadata = json.load(f)
    
    # Verify that the model has been trained
    for fold_idx in range(metadata['num_folds']):
        fold_dir = ensemble_dir / f"fold_{fold_idx}"
        if not any(fold_dir.glob("*.ckpt")):
            raise ValueError(
                f"No checkpoint found in {fold_dir}. "
                "Please ensure the model has been trained before using it for predictions."
            )
    
    return metadata

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

def save_predictions(
    predictions: Dict[str, List],
    ensemble_dir: Path,
    model_identifier: str,
    output_dir: str = "predictions"
) -> None:
    """Save predictions to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(output_dir) / f"ensemble_predictions_{model_identifier}_{timestamp}"
    
    # Create DataFrame with all predictions
    pred_df = pd.DataFrame({
        'compound_id': predictions['compound_id'],
        'feature_idx': predictions['feature_idx'],
        'regression_pred': predictions['regression_pred'],
        'regression_std': predictions['regression_std'],
        'classification_prob': predictions['classification_prob'],
        'classification_std': predictions['classification_std']
    })
    
    # Save predictions
    pred_df.to_csv(f"{output_base}.csv", index=False)
    
    logging.info(f"\nSaved predictions to:")
    logging.info(f"- {output_base}.csv")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main prediction function for ensemble models."""
    setup_logging(cfg.predict.output_dir)
    
    try:
        if not cfg.predict.ensemble.enabled:
            raise ValueError("Ensemble prediction is not enabled. Set predict.ensemble.enabled=true")
            
        # Load ensemble metadata
        ensemble_dir = Path(cfg.predict.ensemble.model_dir)
        if not ensemble_dir.exists():
            raise ValueError(f"Ensemble directory not found: {ensemble_dir}")
            
        metadata = load_ensemble_metadata(ensemble_dir)
        model_type = metadata['model_type']
        model_identifier = metadata['model_identifier']
        
        logging.info(f"Predicting with ensemble model: {model_identifier}")
        logging.info(f"Model type: {model_type}")
        logging.info(f"Ensemble directory: {ensemble_dir}")
        
        # Read SMILES data
        smiles_df = pd.read_csv(cfg.predict.smiles_file, sep='\t')
        required_cols = ['IDNUMBER', 'Canonical_SMILES']
        if not all(col in smiles_df.columns for col in required_cols):
            raise ValueError(f"SMILES file must contain columns: {required_cols}")
            
        # Get number of features from metadata
        num_features = metadata['config']['model'].get('num_features', None)
        if num_features is None:
            # Try to infer from first model checkpoint
            first_fold = next(ensemble_dir.glob('fold_*/fold_*.ckpt'))
            checkpoint = torch.load(first_fold)
            num_features = sum(1 for key in checkpoint['state_dict'].keys() 
                             if 'regression_heads' in key and '.weight' in key)
        
        logging.info(f"Number of features: {num_features}")
        
        # Create data module
        data_module = PredictionDataModule(
            smiles_file=cfg.predict.smiles_file,
            num_features=num_features,
            model_type=model_type,
            model_name=metadata['config']['model']['architecture'].get('foundation_model', None),
            batch_size=cfg.predict.batch_size,
            num_workers=cfg.predict.num_workers,
            max_length=metadata['config']['model']['architecture'].get('max_length', 512)
        )
        
        # Get model parameters
        model_params = get_model_params(metadata, cfg, num_features)
        
        # Verify the model directory contains trained models
        if not any(ensemble_dir.glob("fold_*/fold_*.ckpt")):
            raise ValueError(
                f"No trained model checkpoints found in {ensemble_dir}. "
                "You need to train the model first using train_ensemble.py before running predictions."
            )

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
        
        # Setup data
        data_module.setup()
        predict_dataloader = data_module.predict_dataloader()
        
        # Create a mapping of batch indices to compound IDs
        smiles_df_list = smiles_df['IDNUMBER'].tolist()
        
        # Initialize predictions storage
        all_predictions = {
            'compound_id': [],
            'feature_idx': [],
            'regression_pred': [],
            'regression_std': [],
            'classification_prob': [],
            'classification_std': []
        }
        
        # Put all models in eval mode
        for model in ensemble.models:
            model.eval()
            
        # Make predictions
        with torch.no_grad():
            for batch_idx, batch in enumerate(predict_dataloader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get ensemble predictions
                results = ensemble.predict_with_uncertainty(batch)
                
                # Store predictions
                mol_indices = batch['mol_idx'].cpu().numpy()
                
                for i, (feat_idx, mol_idx) in enumerate(zip(batch['feature_idx'].cpu().numpy(), mol_indices)):
                    all_predictions['compound_id'].append(smiles_df_list[mol_idx])
                    all_predictions['feature_idx'].append(feat_idx)
                    all_predictions['regression_pred'].append(results['regression_pred'][i].cpu().item())
                    all_predictions['regression_std'].append(results['regression_std'][i].cpu().item())
                    all_predictions['classification_prob'].append(results['classification_prob'][i].cpu().item())
                    all_predictions['classification_std'].append(results['classification_std'][i].cpu().item())
                    
                    # Log first few predictions for verification
                    if batch_idx == 0 and i < 5:
                        logging.info(f"\nPrediction {i}:")
                        logging.info(f"Compound: {smiles_df_list[mol_idx]}")
                        logging.info(f"Feature: {feat_idx}")
                        logging.info(f"Regression: {results['regression_pred'][i].item():.3f} "
                                   f"± {results['regression_std'][i].item():.3f}")
                        logging.info(f"Classification: {results['classification_prob'][i].item():.3f} "
                                   f"± {results['classification_std'][i].item():.3f}")
        
        # Save predictions
        save_predictions(all_predictions, ensemble_dir, model_identifier, cfg.predict.output_dir)
        
        # Print summary statistics
        logging.info("\nPrediction Summary:")
        logging.info(f"Total predictions: {len(all_predictions['compound_id'])}")
        logging.info(f"Unique compounds: {len(set(all_predictions['compound_id']))}")
        logging.info(f"Unique features: {len(set(all_predictions['feature_idx']))}")
        
        # Calculate uncertainty statistics
        reg_std_mean = np.mean(all_predictions['regression_std'])
        reg_std_std = np.std(all_predictions['regression_std'])
        cls_std_mean = np.mean(all_predictions['classification_std'])
        cls_std_std = np.std(all_predictions['classification_std'])
        
        logging.info("\nUncertainty Statistics:")
        logging.info(f"Regression uncertainty: {reg_std_mean:.3f} ± {reg_std_std:.3f}")
        logging.info(f"Classification uncertainty: {cls_std_mean:.3f} ± {cls_std_std:.3f}")
        
    except Exception as e:
        logging.error(f"Error during ensemble prediction: {str(e)}")
        raise

if __name__ == '__main__':
    main()
