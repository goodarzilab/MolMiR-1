"""
MolMir model for molecular miRNA target prediction.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging
import numpy as np

class MolMir(pl.LightningModule):
    """
    MolMir model for molecular miRNA target prediction combining regression and classification tasks.
    
    Attributes:
        backbone (AutoModel): Pre-trained transformer model
        shared_layer (nn.Sequential): Shared processing layer
        regression_heads (nn.ModuleList): Feature-specific regression heads
        classification_heads (nn.ModuleList): Feature-specific classification heads
        reg_weight (float): Weight for regression loss
        cls_weight (float): Weight for classification loss
        class_weights (Optional[torch.Tensor]): Class weights for handling imbalanced data
    """
    
    def __init__(
        self,
        model_name: str,
        num_features: int,
        learning_rate: float = 1e-4,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0,
        hidden_dropout_prob: float = 0.1,
        hidden_size: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        trust_remote_code: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layers: Optional[int] = 2
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        
        # Load pre-trained model and config first
        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Use model's native hidden size if none provided
        if hidden_size is None:
            hidden_size = self.config.hidden_size
            logging.info(f"Using model's native hidden size: {hidden_size}")
        else:
            self.config.hidden_size = hidden_size
            
        # Load the model with potentially mismatched sizes
        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=True
        )

        #  Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze last few layers if specified
            if unfreeze_layers > 0:
                try:
                    # Attempt to unfreeze last few layers of transformer 
                    # This could vary depending on the specific model architecture
                    if hasattr(self.backbone, 'encoder'):
                        layers = list(self.backbone.encoder.layer)
                        for layer in layers[-unfreeze_layers:]:
                            for param in layer.parameters():
                                param.requires_grad = True
                    elif hasattr(self.backbone, 'layers'):
                        layers = list(self.backbone.layers)
                        for layer in layers[-unfreeze_layers:]:
                            for param in layer.parameters():
                                param.requires_grad = True
                
                    # Always make sure embedding layers for special tokens can be updated
                    if hasattr(self.backbone, 'embeddings'):
                        for param in self.backbone.embeddings.parameters():
                            param.requires_grad = True
                except Exception as e:
                    logging.warning(f"Could not partially unfreeze backbone: {e}")
        
        # Prediction heads
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Shared layers
        self.shared_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob)
        )
        
        # Task-specific heads
        self.regression_heads = nn.ModuleList([
            nn.Linear(self.config.hidden_size, 1) for _ in range(num_features)
        ])
        self.classification_heads = nn.ModuleList([
            nn.Linear(self.config.hidden_size, 1) for _ in range(num_features)
        ])
        
        # Ensure weights are never None
        self.reg_weight = float(reg_weight)
        self.cls_weight = float(cls_weight)
        
        # Register class weights as buffer to move with model
        self.register_buffer('class_weights', class_weights)
        
        # Initialize validation step outputs list and feature metrics
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.feature_metrics = {}
        
        logging.info(f"Initialized MolMir model with {num_features} features")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        feature_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            feature_idx: Feature indices [batch_size]
            
        Returns:
            Tuple of regression predictions and classification predictions [batch_size]
        """
        # Get molecular embeddings
        outputs = self.backbone(
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

    def _compute_loss(
        self, 
        batch: Dict[str, torch.Tensor], 
        prefix: str = ''
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss for regression and classification tasks.
        
        Args:
            batch: Batch of data
            prefix: Prefix for metric logging
            
        Returns:
            Tuple of (total loss, regression predictions, classification predictions)
        """
        reg_preds, cls_preds = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['feature_idx']
        )
        
        # Regression loss
        reg_loss = F.mse_loss(reg_preds, batch['z_score'])
        
        # Classification loss with class weights if available
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            pos_weight = self.class_weights[1].to(cls_preds.device)
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_preds,
                batch['is_hit'],
                pos_weight=pos_weight
            )
        else:
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_preds,
                batch['is_hit']
            )
        
        # Combined loss
        loss = self.reg_weight * reg_loss + self.cls_weight * cls_loss
        
        # Log metrics
        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_reg_loss', reg_loss, on_step=True, on_epoch=True)
        self.log(f'{prefix}_cls_loss', cls_loss, on_step=True, on_epoch=True)
        
        return loss, reg_preds, cls_preds

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, _, _ = self._compute_loss(batch, prefix='train')
        return loss

    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        loss, reg_preds, cls_preds = self._compute_loss(batch, prefix='val')
        
        # Store predictions and targets for epoch end metrics
        self.validation_step_outputs.append({
            'reg_preds': reg_preds.detach(),
            'cls_preds': torch.sigmoid(cls_preds).detach(),
            'z_score': batch['z_score'],
            'is_hit': batch['is_hit'],
            'feature_idx': batch['feature_idx']
        })
        
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step logic."""
        loss, reg_preds, cls_preds = self._compute_loss(batch, prefix='test')
        
        # Store predictions and targets for epoch end metrics
        self.test_step_outputs.append({
            'reg_preds': reg_preds.detach(),
            'cls_preds': torch.sigmoid(cls_preds).detach(),
            'z_score': batch['z_score'],
            'is_hit': batch['is_hit'],
            'feature_idx': batch['feature_idx']
        })
        
        return loss
        
    def setup_test(self) -> None:
        """Initialize test-specific attributes."""
        self.test_step_outputs = []
        self.test_metrics = {}
        
    def on_test_epoch_start(self) -> None:
        """Setup for test epoch."""
        self.setup_test()
    def on_test_epoch_end(self) -> None:
        """Compute end of epoch test metrics."""
        # Exit early if no test outputs
        if not self.test_step_outputs:
            logging.warning("No test step outputs found")
            return
            
        # Concatenate all outputs
        all_outputs = {
            key: torch.cat([x[key] for x in self.test_step_outputs])
            for key in self.test_step_outputs[0].keys()
        }
        
        # Calculate overall regression metrics
        test_metrics = {}
        test_rmse = torch.sqrt(F.mse_loss(all_outputs['reg_preds'], all_outputs['z_score']))
        test_metrics['test_rmse'] = test_rmse
        self.log('test_rmse', test_rmse)
        
        # Calculate overall Pearson correlation
        if len(all_outputs['reg_preds']) > 1:
            try:
                from scipy import stats
                pearson_r, pearson_p = stats.pearsonr(
                    all_outputs['reg_preds'].cpu().numpy(),
                    all_outputs['z_score'].cpu().numpy()
                )
                test_metrics['test_pearson_r'] = float(pearson_r)
                test_metrics['test_pearson_p'] = float(pearson_p)
                self.log('test_pearson_r', float(pearson_r))
                self.log('test_pearson_p', float(pearson_p))
            except Exception as e:
                logging.warning(f"Could not compute overall Pearson correlation: {e}")
        
        # Calculate overall AUC if we have both positive and negative samples
        test_auc = self._safe_compute_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if test_auc is not None:
            test_metrics['test_auc'] = test_auc
            self.log('test_auc', test_auc)
            
        test_pr_auc = self._safe_compute_pr_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if test_pr_auc is not None:
            test_metrics['test_pr_auc'] = test_pr_auc
            self.log('test_pr_auc', test_pr_auc)
        
        # Calculate per-feature metrics
        feature_metrics = {}
        for feature_idx in torch.unique(all_outputs['feature_idx']):
            mask = all_outputs['feature_idx'] == feature_idx
            feature_metric = self._compute_feature_metrics(all_outputs, mask, feature_idx)
            
            if feature_metric:
                idx = int(feature_idx.item())
                feature_metrics[idx] = feature_metric
                # Log feature metrics
                for metric_name, value in feature_metric.items():
                    if value is not None:
                        metric_key = f'test_feature_{metric_name}_{idx}'
                        test_metrics[metric_key] = value
                        self.log(metric_key, value)
        
        # Add feature metrics to overall metrics
        test_metrics['test_metrics'] = feature_metrics
        
        # Log number of samples processed
        logging.info(f"Processed test results for {len(all_outputs['feature_idx'])} samples "
                    f"across {len(feature_metrics)} features")
        
        # Clear stored predictions
        self.test_step_outputs.clear()
    
    def _safe_compute_auc(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Optional[float]:
        """Safely compute AUC score handling edge cases and NaN values."""
        try:
            # Remove NaN values
            mask = ~torch.isnan(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            # Convert to numpy and check for valid values
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            if len(np.unique(y_true_np)) < 2:
                return None
            
            return roc_auc_score(y_true_np, y_pred_np)
        except Exception as e:
            logging.warning(f"Error computing AUC: {e}")
            return None

    def _safe_compute_pr_auc(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Optional[float]:
        """Safely compute PR-AUC score handling edge cases and NaN values."""
        try:
            # Remove NaN values
            mask = ~torch.isnan(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            # Convert to numpy and check for valid values
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            if len(np.unique(y_true_np)) < 2:
                return None
            
            precision, recall, _ = precision_recall_curve(y_true_np, y_pred_np)
            return auc(recall, precision)
        except Exception as e:
            logging.warning(f"Error computing PR-AUC: {e}")
            return None

    def _compute_feature_metrics(self, all_outputs, mask, feature_idx):
        """Compute metrics for a specific feature."""
        metrics = {}
        
        # Skip if no samples for this feature
        if not torch.any(mask):
            return metrics
        
        # Regression metrics
        reg_preds = all_outputs['reg_preds'][mask]
        z_scores = all_outputs['z_score'][mask]
        if not torch.all(torch.isnan(reg_preds)):
            # RMSE
            metrics['rmse'] = torch.sqrt(F.mse_loss(reg_preds, z_scores)).item()
            
            # Pearson correlation
            if len(reg_preds) > 1:  # Need at least 2 points for correlation
                reg_preds_numpy = reg_preds.cpu().numpy()
                z_scores_numpy = z_scores.cpu().numpy()
                try:
                    from scipy import stats
                    pearson_r, pearson_p = stats.pearsonr(reg_preds_numpy, z_scores_numpy)
                    metrics['pearson_r'] = float(pearson_r)
                    metrics['pearson_p'] = float(pearson_p)
                except Exception as e:
                    logging.warning(f"Could not compute Pearson correlation for feature {feature_idx}: {e}")
        
        # Classification metrics
        cls_preds = all_outputs['cls_preds'][mask]
        is_hit = all_outputs['is_hit'][mask]
        
        auc_score = self._safe_compute_auc(is_hit, cls_preds)
        if auc_score is not None:
            metrics['auc'] = auc_score
            
        pr_auc_score = self._safe_compute_pr_auc(is_hit, cls_preds)
        if pr_auc_score is not None:
            metrics['pr_auc'] = pr_auc_score
            
        metrics['num_samples'] = int(torch.sum(mask).item())
        return metrics

    def on_validation_epoch_end(self) -> None:
        """Compute end of epoch validation metrics."""
        # Concatenate all outputs
        all_outputs = {
            key: torch.cat([x[key] for x in self.validation_step_outputs])
            for key in self.validation_step_outputs[0].keys()
        }
        
        # Calculate overall regression metrics
        val_rmse = torch.sqrt(F.mse_loss(all_outputs['reg_preds'], all_outputs['z_score']))
        self.log('val_rmse', val_rmse)
        
        # Calculate overall Pearson correlation
        if len(all_outputs['reg_preds']) > 1:
            try:
                from scipy import stats
                pearson_r, pearson_p = stats.pearsonr(
                    all_outputs['reg_preds'].cpu().numpy(),
                    all_outputs['z_score'].cpu().numpy()
                )
                self.log('val_pearson_r', pearson_r)
                self.log('val_pearson_p', pearson_p)
            except Exception as e:
                logging.warning(f"Could not compute overall Pearson correlation: {e}")
        
        # Calculate AUC if we have both positive and negative samples
        val_auc = self._safe_compute_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if val_auc is not None:
            self.log('val_auc', val_auc)
            
        val_pr_auc = self._safe_compute_pr_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if val_pr_auc is not None:
            self.log('val_pr_auc', val_pr_auc)
        
        # Calculate per-feature metrics
        self.feature_metrics = {}
        for feature_idx in torch.unique(all_outputs['feature_idx']):
            mask = all_outputs['feature_idx'] == feature_idx
            feature_metrics = self._compute_feature_metrics(all_outputs, mask, feature_idx)
            
            if feature_metrics:
                self.feature_metrics[int(feature_idx.item())] = feature_metrics
                # Log feature metrics
                for metric_name, value in feature_metrics.items():
                    if value is not None:
                        self.log(f'val_feature_{metric_name}_{feature_idx}', value)
                        
    def on_validation_epoch_end(self) -> None:
        """Compute end of epoch validation metrics."""
        # Concatenate all outputs
        all_outputs = {
            key: torch.cat([x[key] for x in self.validation_step_outputs])
            for key in self.validation_step_outputs[0].keys()
        }
        
        # Calculate overall regression metric
        val_rmse = torch.sqrt(F.mse_loss(all_outputs['reg_preds'], all_outputs['z_score']))
        self.log('val_rmse', val_rmse)
        
        # Calculate overall classification metrics
        val_auc = self._safe_compute_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if val_auc is not None:
            self.log('val_auc', val_auc)
            
        val_pr_auc = self._safe_compute_pr_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if val_pr_auc is not None:
            self.log('val_pr_auc', val_pr_auc)
        
        # Calculate per-feature metrics
        self.feature_metrics = {}
        for feature_idx in torch.unique(all_outputs['feature_idx']):
            mask = all_outputs['feature_idx'] == feature_idx
            feature_metrics = self._compute_feature_metrics(all_outputs, mask, feature_idx)
            
            if feature_metrics:
                self.feature_metrics[int(feature_idx.item())] = feature_metrics
                # Log feature metrics
                for metric_name, value in feature_metrics.items():
                    if value is not None:
                        self.log(f'val_feature_{metric_name}_{feature_idx}', value)
        
        # Clear stored predictions
        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, Dict]]:
        """Configure optimizer and learning rate scheduler."""
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

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """Custom load state dict that handles class weights properly."""
        # Create a copy of the state dict so we can modify it
        state_dict = state_dict.copy()
        
        # Handle class weights if present
        if 'class_weights' in state_dict:
            class_weights = state_dict.pop('class_weights')
            self.register_buffer('class_weights', class_weights)
        
        # Call parent class load_state_dict with modified state_dict and non-strict mode
        # This allows loading without class weights
        return super().load_state_dict(state_dict, strict=False)
