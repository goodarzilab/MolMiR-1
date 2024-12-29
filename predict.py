"""
Prediction script for MolMir model.
Takes SMILES input and returns predictions using a trained model.

Examples:
    ChemBERTa model prediction:
        $ python predict.py model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM \
            predict.smiles_file=test/test_SMILES.tsv \
            "+predict.checkpoint_path='checkpoints/molmir_ChemBERTa-77M-MLM_epoch=01_lossval_loss=1.33_aucval_auc=0.556_praucval_pr_auc=0.041.ckpt'"

    GCN model prediction:
        $ python predict.py model.architecture.type=gcn \
            predict.smiles_file=test/test_SMILES.tsv \
            "+predict.checkpoint_path='checkpoints/molmir_gcn_model.ckpt'"

    MPNN model prediction:
        $ python predict.py model.architecture.type=mpnn \
            predict.smiles_file=test/test_SMILES.tsv \
            "+predict.checkpoint_path='checkpoints/molmir_mpnn_model_epoch=03_lossval_loss=1.32_aucval_auc=0.633_praucval_pr_auc=0.064.ckpt'"
"""

import os
import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.data.predict_data import PredictionDataModule
from src.models import MolMir
from src.models.graph_models import (
    MolMirGCN, 
    MolMirMPNN, 
    MolMirGIN, 
    MolMirAttentiveFP
)

def setup_logging(output_dir: str = "predictions") -> None:
    """Set up logging and create output directory."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def save_predictions(predictions: dict, model_name: str, output_dir: str) -> None:
    """Save predictions to files."""
    # Create predictions directory
    predictions_dir = Path(output_dir)
    predictions_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = predictions_dir / f"predictions_{model_name}_{timestamp}"
    
    # Save predictions as CSV
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(f"{output_base}.csv", index=False)
    
    logging.info(f"\nSaved predictions to:")
    logging.info(f"- {output_base}.csv")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main prediction function."""
    setup_logging(cfg.predict.output_dir)
    
    try:
        # Read SMILES data
        smiles_df = pd.read_csv(cfg.predict.smiles_file, sep='\t')
        required_cols = ['IDNUMBER', 'Canonical_SMILES']
        if not all(col in smiles_df.columns for col in required_cols):
            raise ValueError(f"SMILES file must contain columns: {required_cols}")
        
        # Get model type
        model_type = cfg.model.architecture.type
        logging.info(f"Using model type: {model_type}")
        
        # Load checkpoint to get number of features
        checkpoint = torch.load(cfg.predict.checkpoint_path)
        num_features = sum(1 for key in checkpoint['state_dict'].keys() 
                         if 'regression_heads' in key and '.weight' in key)
        logging.info(f"Detected {num_features} features from checkpoint")
        
        # Create data module
        data_module = PredictionDataModule(
            smiles_file=cfg.predict.smiles_file,
            num_features=num_features,
            model_type=model_type,
            model_name=cfg.model.architecture.foundation_model if model_type == 'transformer' else None,
            batch_size=cfg.predict.batch_size,
            num_workers=cfg.predict.num_workers,
            max_length=cfg.model.architecture.max_length if model_type == 'transformer' else None
        )
        
        # Initialize model based on type
        model_classes = {
            'transformer': MolMir,
            'gcn': MolMirGCN,
            'mpnn': MolMirMPNN,
            'gin': MolMirGIN,
            'attentivefp': MolMirAttentiveFP
        }
        
        ModelClass = model_classes.get(model_type)
        if ModelClass is None:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Get model initialization parameters based on type
        if model_type == 'transformer':
            model_params = {
                'model_name': cfg.model.architecture.foundation_model,
                'num_features': num_features,
                'learning_rate': cfg.model.training.learning_rate,
                'hidden_dropout_prob': cfg.model.architecture.hidden_dropout_prob,
                'hidden_size': cfg.model.architecture.hidden_size,
                'freeze_backbone': cfg.model.architecture.freeze_backbone,
                'unfreeze_layers': cfg.model.architecture.unfreeze_layers
            }
        elif model_type == 'attentivefp':
            model_params = {
                'num_features': num_features,
                'num_node_features': cfg.model.architecture.graph.num_node_features,
                'num_edge_features': cfg.model.architecture.graph.num_edge_features,
                'hidden_channels': cfg.model.architecture.graph.hidden_channels,
                'num_layers': cfg.model.architecture.graph.num_layers,
                'dropout': cfg.model.architecture.graph.dropout,
                'learning_rate': cfg.model.training.learning_rate
            }
        else:  # gcn, gin, mpnn
            model_params = {
                'num_features': num_features,
                'num_node_features': cfg.model.architecture.graph.num_node_features,
                'hidden_channels': cfg.model.architecture.graph.hidden_channels,
                'num_layers': cfg.model.architecture.graph.num_layers,
                'dropout': cfg.model.architecture.graph.dropout,
                'learning_rate': cfg.model.training.learning_rate
            }

        # Initialize model
        model = ModelClass(**model_params)
        
        # Load checkpoint
        if not os.path.exists(cfg.predict.checkpoint_path):
            raise ValueError(f"Checkpoint not found: {cfg.predict.checkpoint_path}")
            
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()
        
        # Setup data
        data_module.setup()
        predict_dataloader = data_module.predict_dataloader()
        
        # Create a mapping of batch indices to compound IDs
        smiles_df_list = smiles_df['IDNUMBER'].tolist()
        
        # Make predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        all_predictions = {
            'compound_id': [],
            'feature_idx': [],
            'regression_pred': [],
            'classification_prob': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(predict_dataloader):
                # Debug first batch
                if batch_idx == 0:
                    logging.info("\nFirst batch keys:")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logging.info(f"{key}: Tensor shape {value.shape}")
                        else:
                            logging.info(f"{key}: {type(value)}")
                
                # Move tensors to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions based on model type
                if model_type == 'transformer':
                    reg_preds, cls_preds = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        feature_idx=batch['feature_idx']
                    )
                else:
                    reg_preds, cls_preds = model(
                        input_ids=batch['x'],
                        attention_mask=batch['edge_index'],
                        feature_idx=batch['feature_idx'],
                        edge_attr=batch.get('edge_attr', None),
                        batch=batch.get('batch', None)
                    )
                
                # Store predictions
                mol_indices = batch['mol_idx'].cpu().numpy()
                for i, (feat_idx, mol_idx) in enumerate(zip(batch['feature_idx'].cpu().numpy(), mol_indices)):
                    all_predictions['compound_id'].append(smiles_df_list[mol_idx])
                    all_predictions['feature_idx'].append(feat_idx)
                    all_predictions['regression_pred'].append(reg_preds[i].cpu().item())
                    all_predictions['classification_prob'].append(torch.sigmoid(cls_preds[i]).cpu().item())
                    
                    # Log first few predictions for verification
                    if batch_idx == 0 and i < 5:
                        logging.info(f"\nPrediction {i}:")
                        logging.info(f"Compound: {smiles_df_list[mol_idx]}")
                        logging.info(f"Feature: {feat_idx}")
                        logging.info(f"Regression: {reg_preds[i].item():.3f}")
                        logging.info(f"Classification: {torch.sigmoid(cls_preds[i]).item():.3f}")
                
                # Log some predictions for verification
                if batch_idx == 0:  # Only first batch
                    logging.info(f"\nSample predictions from first batch:")
                    for i in range(min(5, len(batch['feature_idx']))):
                        mol_idx = mol_indices[i % len(mol_indices)]
                        logging.info(f"Compound: {smiles_df_list[mol_idx]}, "
                                   f"Feature: {batch['feature_idx'][i].item()}, "
                                   f"Regression: {reg_preds[i].item():.3f}, "
                                   f"Classification: {torch.sigmoid(cls_preds[i]).item():.3f}")
        
        # Save predictions
        model_name = (
            cfg.model.architecture.foundation_model.split('/')[-1] 
            if model_type == 'transformer' 
            else model_type
        )
        save_predictions(all_predictions, model_name, cfg.predict.output_dir)
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == '__main__':
    main()
