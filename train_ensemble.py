"""
Training script for ensemble MolMir models using k-fold cross-validation.

Examples:
    Train transformer ensemble:
        $ python train_ensemble.py model.architecture.type=transformer \
            model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM \
            training.ensemble.enabled=true \
            training.ensemble.num_folds=5

    Train GCN ensemble:
        $ python train_ensemble.py model.architecture.type=gcn \
            training.ensemble.enabled=true \
            training.ensemble.num_folds=5

    Train MPNN ensemble:
        $ python train_ensemble.py model.architecture.type=mpnn \
            training.ensemble.enabled=true \
            training.ensemble.num_folds=5
"""

import os
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import json

from src.data import MolecularDataModule
from src.models.ensemble import MolMirEnsemble

def setup_logging(cfg: DictConfig) -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    ensemble_dir = Path(cfg.training.checkpoint_dir) / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.training.log_dir).mkdir(parents=True, exist_ok=True)

def save_ensemble_metadata(ensemble_dir: Path, cfg: DictConfig, model_identifier: str) -> None:
    """Save ensemble configuration and metadata."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    metadata = {
        'model_type': cfg.model.architecture.type,
        'model_identifier': model_identifier,
        'num_folds': cfg.training.ensemble.num_folds,
        'config': cfg_dict
    }
    
    with open(ensemble_dir / 'ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def get_model_identifier(cfg: DictConfig) -> str:
    """Get model identifier based on architecture type."""
    if cfg.model.architecture.type == 'transformer':
        return cfg.model.architecture.foundation_model.split('/')[-1]
    return f"{cfg.model.architecture.type}_model"

def setup_fold_data_module(
    cfg: DictConfig,
    model_type: str,
    train_indices: list,
    val_indices: list
) -> MolecularDataModule:
    """Create and setup a data module for a specific fold."""
    data_module = MolecularDataModule(
        smiles_path=cfg.data.smiles_path,
        activity_path=cfg.data.activity_path,
        model_type=model_type,
        model_name=cfg.model.architecture.foundation_model if model_type == 'transformer' else None,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        max_length=cfg.model.architecture.max_length,
        z_score_threshold=cfg.model.z_score_threshold
    )
    
    # Setup the data module - this will create the internal dataset
    data_module.setup("fit")
    
    # Create new Subset objects with our fold-specific indices
    data_module.train_dataset = torch.utils.data.Subset(data_module.train_dataset.dataset, train_indices)
    data_module.val_dataset = torch.utils.data.Subset(data_module.train_dataset.dataset, val_indices)
    
    return data_module

def create_fold_trainer(
    cfg: DictConfig,
    fold_idx: int,
    fold_dir: Path,
    wandb_logger: WandbLogger
) -> pl.Trainer:
    """Create a trainer for a specific fold."""
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=fold_dir,
            filename=f'fold_{fold_idx}_' + '{epoch:02d}_{val_loss:.2f}_{val_auc:.3f}',
            save_top_k=1,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=cfg.training.patience,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    return pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        val_check_interval=cfg.training.val_check_interval,
        deterministic=True,
        precision=16
    )

def get_model_params(cfg: DictConfig, model_type: str, num_features: int) -> dict:
    """Extract relevant model parameters from config based on model type."""
    base_params = {
        'num_features': num_features,
        'learning_rate': cfg.model.training.learning_rate,
        'reg_weight': cfg.model.training.reg_weight,
        'cls_weight': cfg.model.training.cls_weight,
    }
    
    if model_type == 'transformer':
        base_params.update({
            'model_name': cfg.model.architecture.foundation_model,
            'hidden_dropout_prob': cfg.model.architecture.hidden_dropout_prob,
            'hidden_size': cfg.model.architecture.hidden_size,
            'freeze_backbone': cfg.model.architecture.freeze_backbone,
            'unfreeze_layers': cfg.model.architecture.unfreeze_layers,
            'trust_remote_code': cfg.model.get('trust_remote_code', True)
        })
    else:  # graph models
        base_params.update({
            'num_node_features': cfg.model.architecture.graph.num_node_features,
            'hidden_channels': cfg.model.architecture.graph.hidden_channels,
            'num_layers': cfg.model.architecture.graph.num_layers,
            'dropout': cfg.model.architecture.graph.dropout
        })
        
        # Add edge features for AttentiveFP
        if model_type == 'attentivefp':
            base_params['num_edge_features'] = cfg.model.architecture.graph.num_edge_features
            
    return base_params

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function for ensemble models."""
    try:
        # Basic setup
        setup_logging(cfg)
        torch.set_float32_matmul_precision('high')
        
        # Get model type and identifier
        model_type = cfg.model.architecture.type
        model_identifier = get_model_identifier(cfg)
        logging.info(f"\nInitializing ensemble training for {model_identifier}")
        
        # Set up ensemble directory
        ensemble_dir = Path(cfg.training.checkpoint_dir) / "ensemble" / model_identifier
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble metadata
        save_ensemble_metadata(ensemble_dir, cfg, model_identifier)
        
        # Initialize base data module to get num_features
        base_data_module = MolecularDataModule(
            smiles_path=cfg.data.smiles_path,
            activity_path=cfg.data.activity_path,
            model_type=model_type,
            model_name=cfg.model.architecture.foundation_model if model_type == 'transformer' else None,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            max_length=cfg.model.architecture.max_length,
            z_score_threshold=cfg.model.z_score_threshold
        )
        base_data_module.setup("fit")
        
        # Get model parameters from metadata
        model_params = get_model_params(cfg, model_type, base_data_module.num_features)
        
        # Initialize ensemble
        ensemble = MolMirEnsemble(
            model_type=model_type,
            model_params=model_params,
            num_folds=cfg.training.ensemble.num_folds,
            ensemble_dir=ensemble_dir
        )
        
        # Get all training data indices
        all_train_indices = list(range(len(base_data_module.train_dataset)))
        
        # Get k-fold splits
        splits = ensemble.get_fold_splits(all_train_indices, cfg.training.ensemble.num_folds)
        logging.info(f"\nCreated {len(splits)} fold splits")
        
        # Train each fold
        for fold_idx, (fold_train_indices, fold_val_indices) in enumerate(splits):
            logging.info(f"\nTraining fold {fold_idx + 1}/{cfg.training.ensemble.num_folds}")
            logging.info(f"Train size: {len(fold_train_indices)}, Val size: {len(fold_val_indices)}")
            
            try:
                # Create fold-specific directory
                fold_dir = ensemble_dir / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                
                # Setup data module for this fold
                fold_data_module = setup_fold_data_module(
                    cfg, model_type, fold_train_indices, fold_val_indices
                )
                
                # Initialize wandb logger for this fold
                wandb_logger = WandbLogger(
                    project=cfg.wandb.project,
                    name=f"{cfg.wandb.run_name}_fold_{fold_idx}" if cfg.wandb.run_name else f"fold_{fold_idx}",
                    config=OmegaConf.to_container(cfg, resolve=True),
                    save_dir=fold_dir
                )
                
                # Create model for this fold with correct parameters
                model = ensemble.model_class(**model_params)
                
                # Create trainer
                trainer = create_fold_trainer(cfg, fold_idx, fold_dir, wandb_logger)
                
                # Train fold
                trainer.fit(model, fold_data_module)
                
                # Save fold information
                ensemble.save_fold(
                    fold_idx=fold_idx,
                    model_path=trainer.checkpoint_callback.best_model_path,
                    metrics={
                        'val_loss': float(trainer.callback_metrics.get('val_loss', float('inf'))),
                        'val_auc': float(trainer.callback_metrics.get('val_auc', 0.0))
                    }
                )
                
                logging.info(f"Fold {fold_idx} best validation loss: "
                            f"{trainer.callback_metrics.get('val_loss', float('inf')):.4f}")
                
            except Exception as e:
                logging.error(f"Error training fold {fold_idx}: {str(e)}")
                continue
            finally:
                wandb.finish()
        
        logging.info(f"\nEnsemble training completed!")
        logging.info(f"Ensemble saved in: {ensemble_dir}")
        
    except Exception as e:
        logging.error(f"Error during ensemble training: {str(e)}")
        raise
    finally:
        # Ensure wandb is properly closed
        try:
            wandb.finish()
        except:
            pass

if __name__ == '__main__':
    main()
