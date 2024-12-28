"""
Training script for molecular property prediction models.
Supports both transformer and graph-based architectures.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig
import torch
import wandb

# Set environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.data import MolecularDataModule
from src.models import MolMir
from src.models.graph_models import MolMirGCN, MolMirMPNN, MolMirGIN, MolMirAttentiveFP
from src.utils.model_config import get_model_config


def setup_logging(cfg: DictConfig) -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(level=logging.INFO)
    
    # Create necessary directories
    for directory in [cfg.training.log_dir, cfg.training.checkpoint_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_wandb(cfg: DictConfig, model_type: str, model_name: str) -> WandbLogger:
    """Initialize and set up Weights & Biases logging."""
    try:
        wandb.finish()
    except:
        pass
    
    # Update config with model type info
    cfg_dict = dict(cfg)
    cfg_dict['model_type'] = model_type
    cfg_dict['model_name'] = model_name
    
    return WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=cfg_dict,
        log_model=True,
        save_dir=cfg.training.log_dir,
    )


def create_model(cfg: DictConfig, num_features: int, class_weights: Optional[torch.Tensor] = None):
    """Create model based on architecture type."""
    model_type = cfg.model.architecture.type
    
    if model_type == 'transformer':
        model = MolMir(
            model_name=cfg.model.architecture.foundation_model,
            num_features=num_features,
            learning_rate=cfg.model.training.learning_rate,
            reg_weight=cfg.model.training.reg_weight,
            cls_weight=cfg.model.training.cls_weight,
            hidden_dropout_prob=cfg.model.architecture.hidden_dropout_prob,
            hidden_size=cfg.model.architecture.hidden_size,
            class_weights=class_weights,
            trust_remote_code=cfg.model.get('trust_remote_code', True),
            freeze_backbone=cfg.model.architecture.freeze_backbone,
            unfreeze_layers=cfg.model.architecture.unfreeze_layers
        )
    elif model_type == 'gcn':
        model = MolMirGCN(
            num_features=num_features,
            num_node_features=cfg.model.architecture.graph.num_node_features, 
            hidden_channels=cfg.model.architecture.graph.hidden_channels,
            num_layers=cfg.model.architecture.graph.num_layers,
            dropout=cfg.model.architecture.graph.dropout,
            learning_rate=cfg.model.training.learning_rate,
            reg_weight=cfg.model.training.reg_weight,
            cls_weight=cfg.model.training.cls_weight,
            class_weights=class_weights
        )
    elif model_type == 'mpnn':
        model = MolMirMPNN(
            num_features=num_features,
            num_node_features=cfg.model.architecture.graph.num_node_features,
            hidden_channels=cfg.model.architecture.graph.hidden_channels,
            num_layers=cfg.model.architecture.graph.num_layers,
            dropout=cfg.model.architecture.graph.dropout,
            learning_rate=cfg.model.training.learning_rate,
            reg_weight=cfg.model.training.reg_weight,
            cls_weight=cfg.model.training.cls_weight,
            class_weights=class_weights
        )
    elif model_type == 'gin':
        model = MolMirGIN(
            num_features=num_features,
            num_node_features=cfg.model.architecture.graph.num_node_features,
            hidden_channels=cfg.model.architecture.graph.hidden_channels,
            num_layers=cfg.model.architecture.graph.num_layers,
            dropout=cfg.model.architecture.graph.dropout,
            learning_rate=cfg.model.training.learning_rate,
            reg_weight=cfg.model.training.reg_weight,
            cls_weight=cfg.model.training.cls_weight,
            class_weights=class_weights
        )
    elif model_type == 'attentivefp':
        model = MolMirAttentiveFP(
            num_features=num_features,
            num_node_features=cfg.model.architecture.graph.num_node_features,
            num_edge_features=cfg.model.architecture.graph.num_edge_features,
            hidden_channels=cfg.model.architecture.graph.hidden_channels,
            num_layers=cfg.model.architecture.graph.num_layers,
            dropout=cfg.model.architecture.graph.dropout,
            learning_rate=cfg.model.training.learning_rate,
            reg_weight=cfg.model.training.reg_weight,
            cls_weight=cfg.model.training.cls_weight,
            class_weights=class_weights
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_callbacks(cfg: DictConfig, model_identifier: str) -> list:
    """Create training callbacks."""
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=cfg.training.checkpoint_dir,
            filename=f'molmir_{model_identifier}' + '_{epoch:02d}_loss{val_loss:.2f}_auc{val_auc:.3f}_prauc{val_pr_auc:.3f}',
            save_top_k=3,
            mode='min',
            save_on_train_epoch_end=False
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=cfg.training.patience,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    return callbacks


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Basic setup
    setup_logging(cfg)
    torch.set_float32_matmul_precision('high')
    
    # Determine model type and identifier
    model_type = cfg.model.architecture.type
    if model_type == 'transformer':
        model_identifier = cfg.model.architecture.foundation_model.split('/')[-1]
    else:
        model_identifier = f"{model_type}_model"
    
    # Initialize wandb logger
    logger = setup_wandb(cfg, model_type, model_identifier)
    
    # Create data module
    data_module = MolecularDataModule(
        smiles_path=cfg.data.smiles_path,
        activity_path=cfg.data.activity_path,
        model_type=model_type,
        model_name=cfg.model.architecture.foundation_model if model_type == 'transformer' else None,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        max_length=cfg.model.architecture.max_length,
        z_score_threshold=cfg.model.z_score_threshold,
        train_size=cfg.training.get('train_size', 0.8),
        val_size=cfg.training.get('val_size', 0.05),
        auto_weight=cfg.model.get('auto_weight', True)
    )
    
    # Set up data module
    data_module.setup()
    
    # Create model
    model = create_model(
        cfg,
        num_features=data_module.num_features,
        class_weights=data_module.class_weights if hasattr(data_module, 'class_weights') else None
    )
    
    # Update wandb config with model details
    logger.experiment.config.update({
        "model_config": {
            "type": model_type,
            "identifier": model_identifier,
            "num_features": data_module.num_features
        }
    })
    
    # Create callbacks
    callbacks = create_callbacks(cfg, model_identifier)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        val_check_interval=cfg.training.val_check_interval,
        deterministic=True,
        precision=16
    )
    
    # Train model
    try:
        trainer.fit(model, data_module)
        
        # Save final model
        final_model_path = Path(cfg.training.checkpoint_dir) / f"molmir_{model_identifier}_final.ckpt"
        trainer.save_checkpoint(final_model_path)
        
        # Log best metric values
        best_val_loss = trainer.checkpoint_callback.best_model_score
        logging.info(f"Best validation loss: {best_val_loss:.4f}")
        logging.info(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()

###
# Train transformer model
#python train.py model.architecture.type=transformer model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM
#
# Train GCN model
#python train.py model.architecture.type=gcn
#
# Train MPNN model
#python train.py model.architecture.type=mpnn
###
