import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc
from typing import Dict, List, Tuple, Optional
import hydra
from omegaconf import DictConfig
import wandb
import logging
from pathlib import Path

class MolecularDataset(Dataset):
    def __init__(self, smiles_data: pd.DataFrame, activity_data: pd.DataFrame, 
                 tokenizer: RobertaTokenizer, z_score_threshold: float = 2.0):
        """
        Args:
            smiles_data: DataFrame with columns ['IDNUMBER', 'Canonical_SMILES']
            activity_data: DataFrame with columns ['drug', 'feature', 'z.score']
                         where 'drug' matches 'IDNUMBER' from smiles_data
            tokenizer: Tokenizer for SMILES strings
            z_score_threshold: Threshold for binary classification (applied to z.score)
        """
        super().__init__()
        
        # Store tokenizer name instead of tokenizer object
        self.tokenizer_name = tokenizer.name_or_path
        self.z_score_threshold = z_score_threshold
        
        # Create mapping from IDNUMBER to Canonical_SMILES
        self.smiles_dict = dict(zip(smiles_data['IDNUMBER'], smiles_data['Canonical_SMILES']))
        
        # Create feature mapping
        self.features = sorted(activity_data['feature'].unique())
        self.feature_to_idx = {f: i for i, f in enumerate(self.features)}
        
        # Prepare data
        self.data = []
        for drug in smiles_data['IDNUMBER'].unique():
            drug_data = activity_data[activity_data['drug'] == drug]
            for feature in self.features:
                feature_data = drug_data[drug_data['feature'] == feature]
                if not feature_data.empty:
                    z_score = feature_data['z.score'].iloc[0]
                    is_hit = float(z_score >= z_score_threshold)
                    self.data.append({
                        'drug': drug,
                        'feature': feature,
                        'z_score': z_score,
                        'is_hit': is_hit
                    })
        
        logging.info(f"Created dataset with {len(self.data)} samples")
        logging.info(f"Number of unique drugs: {len(smiles_data['IDNUMBER'].unique())}")
        logging.info(f"Number of features: {len(self.features)}")
        
        # Initialize tokenizer to None, will be loaded when needed
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        smiles = self.smiles_dict[item['drug']]
        
        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get feature index
        feature_idx = self.feature_to_idx[item['feature']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'feature_idx': torch.tensor(feature_idx),
            'z_score': torch.tensor(item['z_score'], dtype=torch.float),
            'is_hit': torch.tensor(item['is_hit'], dtype=torch.float)
        }

class MultitaskMolecularPredictor(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'seyonec/ChemBERTa-zinc-base-v1',
        num_features: int = 134,
        learning_rate: float = 1e-4,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model and config
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Prediction heads
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        # Shared layers
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.hidden_dropout_prob)
        )
        
        # Task-specific heads
        self.regression_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_features)
        ])
        self.classification_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_features)
        ])
        
        # Metrics
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        
        # Store predictions for epoch end metrics
        self.validation_step_outputs = []
    
    def forward(self, input_ids, attention_mask, feature_idx):
        # Get molecular embeddings
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token representation
        pooled_output = outputs[0][:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Shared features
        shared_features = self.shared_layer(pooled_output)
        
        # Get predictions for specific features
        batch_size = feature_idx.size(0)
        regression_preds = torch.zeros(batch_size, device=self.device)
        classification_preds = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            idx = feature_idx[i]
            regression_preds[i] = self.regression_heads[idx](shared_features[i])
            classification_preds[i] = self.classification_heads[idx](shared_features[i])
        
        return regression_preds, classification_preds

    def training_step(self, batch, batch_idx):
        reg_preds, cls_preds = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['feature_idx']
        )
        
        reg_loss = F.mse_loss(reg_preds, batch['z_score'])
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_preds, batch['is_hit']
        )
        
        # Combined loss
        loss = self.reg_weight * reg_loss + self.cls_weight * cls_loss
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_reg_loss', reg_loss, on_step=True, on_epoch=True)
        self.log('train_cls_loss', cls_loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        reg_preds, cls_preds = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['feature_idx']
        )
        
        reg_loss = F.mse_loss(reg_preds, batch['z_score'])
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_preds, batch['is_hit']
        )
        
        loss = self.reg_weight * reg_loss + self.cls_weight * cls_loss
        
        # Store predictions and targets for epoch end metrics
        self.validation_step_outputs.append({
            'reg_preds': reg_preds.detach(),
            'cls_preds': torch.sigmoid(cls_preds).detach(),
            'z_score': batch['z_score'],
            'is_hit': batch['is_hit'],
            'feature_idx': batch['feature_idx']
        })
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets
        all_reg_preds = torch.cat([x['reg_preds'] for x in self.validation_step_outputs])
        all_cls_preds = torch.cat([x['cls_preds'] for x in self.validation_step_outputs])
        all_z_scores = torch.cat([x['z_score'] for x in self.validation_step_outputs])
        all_is_hit = torch.cat([x['is_hit'] for x in self.validation_step_outputs])
        all_feature_idx = torch.cat([x['feature_idx'] for x in self.validation_step_outputs])
        
        # Calculate overall metrics
        val_rmse = torch.sqrt(F.mse_loss(all_reg_preds, all_z_scores))
        # Only calculate AUC if we have both positive and negative samples
        unique_classes = torch.unique(all_is_hit)
        if len(unique_classes) > 1:
            val_auc = roc_auc_score(all_is_hit.cpu(), all_cls_preds.cpu())
            precision, recall, _ = precision_recall_curve(all_is_hit.cpu(), all_cls_preds.cpu())
            val_pr_auc = auc(recall, precision)
            
            self.log('val_auc', val_auc)
            self.log('val_pr_auc', val_pr_auc)
        
        # Calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(all_is_hit.cpu(), all_cls_preds.cpu())
        val_pr_auc = auc(recall, precision)
        
        # Log metrics
        self.log('val_rmse', val_rmse)
        
        # Calculate per-feature metrics
        for feature_idx in torch.unique(all_feature_idx):
            mask = all_feature_idx == feature_idx
            feature_rmse = torch.sqrt(F.mse_loss(
                all_reg_preds[mask], 
                all_z_scores[mask]
            ))
            self.log(f'val_feature_rmse_{feature_idx}', feature_rmse)
            
            # Only calculate AUC if we have both positive and negative samples for this feature
            feature_labels = all_is_hit[mask]
            unique_feature_classes = torch.unique(feature_labels)
            if len(unique_feature_classes) > 1:
                feature_auc = roc_auc_score(
                    feature_labels.cpu(),
                    all_cls_preds[mask].cpu()
                )
                self.log(f'val_feature_auc_{feature_idx}', feature_auc)
        
        # Clear stored predictions
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }

class MolecularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        smiles_path: str,
        activity_path: str,
        tokenizer_name: str = 'seyonec/ChemBERTa-zinc-base-v1',
        batch_size: int = 32,
        num_workers: int = 4,
        z_score_threshold: float = 2.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.num_features = None  # Will be set during setup
        
    def setup(self, stage: Optional[str] = None):
        # Define cache path
        cache_path = Path("cache")
        cache_path.mkdir(exist_ok=True)
        dataset_cache = cache_path / "processed_dataset.pt"
        
        try:
            if dataset_cache.exists():
                logging.info("Loading preprocessed dataset from cache...")
                cached_data = torch.load(dataset_cache)
                dataset = cached_data['dataset']
                self.num_features = cached_data['num_features']
                logging.info(f"Loaded dataset with {len(dataset)} samples and {self.num_features} features")
            else:
                logging.info(f"Loading SMILES data from {self.hparams.smiles_path}")
                logging.info(f"Loading activity data from {self.hparams.activity_path}")
                
                # Load data
                smiles_data = pd.read_csv(self.hparams.smiles_path, sep='\t')
                logging.info(f"SMILES data shape: {smiles_data.shape}")
                logging.info(f"SMILES data columns: {smiles_data.columns.tolist()}")
                
                activity_data = pd.read_csv(self.hparams.activity_path)
                logging.info(f"Activity data shape: {activity_data.shape}")
                logging.info(f"Activity data columns: {activity_data.columns.tolist()}")
                
                logging.info("Creating dataset... (this might take a few minutes)")
                # Create dataset
                dataset = MolecularDataset(
                    smiles_data,
                    activity_data,
                    self.tokenizer,
                    self.hparams.z_score_threshold
                )
                self.num_features = len(dataset.features)
                logging.info(f"Created dataset with {len(dataset)} samples")
                
                # Cache the processed dataset
                logging.info("Saving dataset to cache...")
                torch.save({
                    'dataset': dataset,
                    'num_features': self.num_features
                }, dataset_cache)
                logging.info("Dataset cached successfully")
            
            # Split data
            logging.info("Splitting dataset into train and validation sets...")
            train_idx, val_idx = train_test_split(
                range(len(dataset)),
                test_size=0.2,
                random_state=42
            )
            
            self.train_dataset = torch.utils.data.Subset(dataset, train_idx)
            self.val_dataset = torch.utils.data.Subset(dataset, val_idx)
            logging.info(f"Split into {len(self.train_dataset)} train and {len(self.val_dataset)} validation samples")
            logging.info("Data setup complete!")
            
        except FileNotFoundError as e:
            logging.error(f"Error: Could not find file: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during data loading: {e}")
            raise
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create necessary directories
    Path(cfg.training.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Set Tensor Cores precision
    torch.set_float32_matmul_precision('high')
    
    # Make sure any existing wandb runs are finished
    try:
        wandb.finish()
    except:
        pass
    
    # Create logger first
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=dict(cfg),
        log_model=True,
        save_dir=cfg.training.log_dir,
    )
    
    # Create data module first
    data_module = MolecularDataModule(
        smiles_path=cfg.data.smiles_path,
        activity_path=cfg.data.activity_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        z_score_threshold=cfg.model.z_score_threshold
    )
    
    # Set up data module to get number of features
    data_module.setup()
    
    # Create model with correct number of features
    model = MultitaskMolecularPredictor(
        model_name=cfg.model.pretrained_model,
        num_features=data_module.num_features,  # Use actual number of features
        learning_rate=cfg.training.learning_rate,
        reg_weight=cfg.model.reg_weight,
        cls_weight=cfg.model.cls_weight
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.training.checkpoint_dir,
        filename='multitask-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=cfg.training.patience,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        gradient_clip_val=cfg.training.gradient_clip_val,
        val_check_interval=cfg.training.val_check_interval,
        deterministic=True,  # For reproducibility
        precision=16  # Use mixed precision training
    )
    
    # Train model
    try:
        trainer.fit(model, data_module)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise
    finally:
        # Make sure to cleanup wandb
        wandb.finish()

    # Save final model
    final_model_path = Path(cfg.training.checkpoint_dir) / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    
    # Log best metric values
    best_val_loss = checkpoint_callback.best_model_score
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()