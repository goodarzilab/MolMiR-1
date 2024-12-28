"""
Graph-based models for molecular property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.data import Data
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from src.utils.molecule_utils import smiles_to_graph

class MolMirGCN(pl.LightningModule):
    """
    Graph Convolutional Network version of MolMir.
    """
    def __init__(
        self,
        num_features: int,
        num_node_features: int = 10,     # Number of atom features we compute
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: num_node_features -> hidden_channels
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Task-specific heads
        self.dropout = nn.Dropout(dropout)
        
        # Shared layer after graph pooling
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction heads
        self.regression_heads = nn.ModuleList([
            nn.Linear(hidden_channels, 1) for _ in range(num_features)
        ])
        self.classification_heads = nn.ModuleList([
            nn.Linear(hidden_channels, 1) for _ in range(num_features)
        ])
        
        # Weights
        self.reg_weight = float(reg_weight)
        self.cls_weight = float(cls_weight)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
            
        # Initialize step outputs lists
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(
        self,
        input_ids: torch.Tensor,  # Will be x for graphs
        attention_mask: torch.Tensor,  # Will be edge_index for graphs
        feature_idx: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GCN model.
        For compatibility with the transformer interface, we use generic parameter names
        but interpret them as graph components.
        """
        #logging.info(f"Input node features shape: {input_ids.shape}")
        #logging.info(f"Edge index shape: {attention_mask.shape}")
        #logging.info(f"Batch tensor shape: {batch.shape if batch is not None else 'None'}")
        
        # Interpret inputs as graph components
        x = input_ids  # Node features
        edge_index = attention_mask  # Edge indices
        
        # Graph convolutions
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Shared features
        x = self.shared_layer(x)
        
        # Task-specific predictions
        batch_size = feature_idx.size(0)
        regression_preds = torch.zeros(batch_size, device=x.device)
        classification_preds = torch.zeros(batch_size, device=x.device)
        
        for i in range(batch_size):
            idx = feature_idx[i]
            regression_preds[i] = self.regression_heads[idx](x[i])
            classification_preds[i] = self.classification_heads[idx](x[i])
        
        return regression_preds, classification_preds

    def _compute_loss(
        self, 
        batch: Dict[str, torch.Tensor], 
        prefix: str = ''
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss with graph data."""
        reg_preds, cls_preds = self(
            input_ids=batch['x'],
            attention_mask=batch['edge_index'],
            feature_idx=batch['feature_idx'],
            edge_attr=batch['edge_attr'],
            batch=batch['batch']
        )
        
        batch_size = batch['feature_idx'].size(0)  # Get actual batch size
        
        # Regression loss
        reg_loss = F.mse_loss(reg_preds, batch['z_score'])
        
        # Classification loss
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            pos_weight = self.class_weights[1].to(cls_preds.device)
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_preds, batch['is_hit'], pos_weight=pos_weight
            )
        else:
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_preds, batch['is_hit']
            )
        
        # Combined loss
        loss = self.reg_weight * reg_loss + self.cls_weight * cls_loss
        
        # Log metrics with explicit batch_size
        on_step = True if prefix == 'train' else False  # Only log steps during training
        self.log(f'{prefix}_loss', loss, batch_size=batch_size, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_reg_loss', reg_loss, batch_size=batch_size, on_step=on_step, on_epoch=True)
        self.log(f'{prefix}_cls_loss', cls_loss, batch_size=batch_size, on_step=on_step, on_epoch=True)
        
        return loss, reg_preds, cls_preds
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._compute_loss(batch, prefix='train')
        return loss
    
    def validation_step(self, batch, batch_idx):
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
        # Exit early if no validation outputs
        if not self.validation_step_outputs:
            return
        
        # Concatenate all outputs
        all_outputs = {
            key: torch.cat([x[key] for x in self.validation_step_outputs])
            for key in self.validation_step_outputs[0].keys()
        }
        
        # Calculate overall regression metric
        val_rmse = torch.sqrt(F.mse_loss(all_outputs['reg_preds'], all_outputs['z_score']))
        self.log('val_rmse', val_rmse)
        
        # Calculate overall Pearson correlation
        if len(all_outputs['reg_preds']) > 1:
            try:
                pearson_r, pearson_p = stats.pearsonr(
                    all_outputs['reg_preds'].cpu().numpy(),
                    all_outputs['z_score'].cpu().numpy()
                )
                self.log('val_pearson_r', pearson_r)
                self.log('val_pearson_p', pearson_p)
            except Exception as e:
                logging.warning(f"Could not compute overall Pearson correlation: {e}")
        
        # Calculate overall classification metrics
        val_auc = self._safe_compute_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if val_auc is not None:
            self.log('val_auc', val_auc)
            
        val_pr_auc = self._safe_compute_pr_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if val_pr_auc is not None:
            self.log('val_pr_auc', val_pr_auc)
        
        # Calculate per-feature metrics
        feature_metrics = {}
        for feature_idx in torch.unique(all_outputs['feature_idx']):
            mask = all_outputs['feature_idx'] == feature_idx
            feature_metric = self._compute_feature_metrics(all_outputs, mask, feature_idx)
            
            if feature_metric:
                feature_metrics[int(feature_idx.item())] = feature_metric
                # Log feature metrics
                for metric_name, value in feature_metric.items():
                    if value is not None:
                        self.log(f'val_feature_{metric_name}_{feature_idx}', value)
        
        # Clear stored predictions
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
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
    
    def on_test_epoch_start(self) -> None:
        """Setup test step outputs."""
        self.test_step_outputs = []

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
    
        logging.info(f"Processing test results for {len(all_outputs['feature_idx'])} samples")
    
        # Calculate overall regression metrics
        test_metrics = {}
        test_rmse = torch.sqrt(F.mse_loss(all_outputs['reg_preds'], all_outputs['z_score']))
        test_metrics['test_rmse'] = test_rmse
    
        # Calculate overall Pearson correlation
        if len(all_outputs['reg_preds']) > 1:
            try:
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
        
        test_pr_auc = self._safe_compute_pr_auc(all_outputs['is_hit'], all_outputs['cls_preds'])
        if test_pr_auc is not None:
            test_metrics['test_pr_auc'] = test_pr_auc
    
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
        
        # Log all metrics
        for key, value in test_metrics.items():
            if not key.endswith('metrics'):  # Don't log the nested metrics dict
                self.log(key, value)
        
        logging.info(f"Computed metrics for {len(feature_metrics)} features")
        
        # Clear stored predictions
        self.test_step_outputs.clear()

    def configure_optimizers(self):
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

    def load_state_dict(self, state_dict, strict: bool = True):
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


class MolMirMPNN(MolMirGCN):
    """
    Message Passing Neural Network version of MolMir.
    Uses more sophisticated message passing than GCN.
    """
    class MPNNLayer(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr='add')
            # Node encoder to map input features to consistent dimension
            self.node_encoder = nn.Linear(in_channels, out_channels)
        
            # Message network with more flexible dimensionality handling
            self.message_network = nn.Sequential(
                nn.Linear(2 * out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            
            # Update network
            self.node_update = nn.Sequential(
                nn.Linear(out_channels + in_channels, out_channels),
                nn.ReLU()
            )
            
        def forward(self, x, edge_index):
            # Ensure input is properly encoded
            x_encoded = self.node_encoder(x)
            
            # Propagate messages
            return self.propagate(edge_index, x=x_encoded, original_x=x)
        
        def message(self, x_i, x_j):
            # Concatenate neighboring node features
            tmp = torch.cat([x_i, x_j], dim=1)
            return self.message_network(tmp)
        
        def update(self, aggr_out, x, original_x):
            # Combine aggregated message with original features
            tmp = torch.cat([aggr_out, original_x], dim=1)
            return self.node_update(tmp)

    def __init__(
            self,
            num_features: int,
            num_node_features: int = 10,     # Number of atom features we compute
            hidden_channels: int = 128,
            num_layers: int = 3,
            dropout: float = 0.1,
            learning_rate: float = 1e-4,
            reg_weight: float = 1.0,
            cls_weight: float = 1.0,
            class_weights: Optional[torch.Tensor] = None
    ):
        # Call parent constructor with correct arguments
        super().__init__(
            num_features=num_features,
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            reg_weight=reg_weight,
            cls_weight=cls_weight,
            class_weights=class_weights
        )
    
        # Recreate convolution layers with correct dimensions
        self.convs = nn.ModuleList()
    
        # First layer: map input features to hidden dimension
        self.convs.append(self.MPNNLayer(num_node_features, hidden_channels))
    
        # Subsequent layers use hidden dimension
        for _ in range(num_layers - 1):
            self.convs.append(self.MPNNLayer(hidden_channels, hidden_channels))

class MolMirGIN(MolMirGCN):
    """
    Graph Isomorphism Network (GIN) version of MolMir.
    Implements GIN convolution with an MLP-based update rule.
    """
    class GINLayer(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr='add')
            # MLP for node and neighborhood feature transformation
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            # Epsilon learnable parameter for GIN update rule
            self.eps = nn.Parameter(torch.Tensor([0]))
        
        def forward(self, x, edge_index):
            return self.propagate(edge_index, x=x)
        
        def message(self, x_j):
            # Message is simply the neighboring node features
            return x_j
        
        def update(self, aggr_out, x):
            # GIN update rule: (1 + ε) * x + aggregate
            return self.mlp((1 + self.eps) * x + aggr_out)
    
    def __init__(
        self,
        num_features: int,
        num_node_features: int = 10,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        # Call parent constructor with the same arguments
        super().__init__(
            num_features=num_features,
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            reg_weight=reg_weight,
            cls_weight=cls_weight,
            class_weights=class_weights
        )
        
        # Replace convolution layers with GIN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(self.GINLayer(num_node_features, hidden_channels))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(self.GINLayer(hidden_channels, hidden_channels))


class MolMirAttentiveFP(MolMirGCN):
    """
    AttentiveFP implementation for molecular property prediction.
    Uses attention mechanisms for message passing.
    """
    class AttentiveFPLayer(MessagePassing):
        def __init__(self, in_channels, out_channels, edge_dim=4):
            super().__init__(aggr='add')
            
            # More flexible dimension handling
            self.node_encoder = nn.Linear(in_channels, out_channels)
            
            # Adjust attention network to handle variable input dimensions
            self.attention_network = nn.Sequential(
                nn.Linear(2 * out_channels + edge_dim, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1)
            )
            
            # Message transformation network
            self.message_network = nn.Sequential(
                nn.Linear(out_channels + edge_dim, out_channels),
                nn.ReLU()
            )
        
        def forward(self, x, edge_index, edge_attr=None):
            # Ensure edge_attr exists
            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.size(1), 4), device=x.device)
            
            # Encode node features
            x_encoded = self.node_encoder(x)
            
            # Message passing
            return self.propagate(edge_index, x=x_encoded, edge_attr=edge_attr)
        
        def message(self, x_i, x_j, edge_attr):
            # Concatenate node and edge features
            tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
            
            # Compute attention scores
            attention_score = torch.sigmoid(self.attention_network(tmp))
            
            # Compute messages
            message_input = torch.cat([x_j, edge_attr], dim=1)
            message = self.message_network(message_input)
            
            return attention_score * message

    def __init__(
        self,
        num_features: int,
        num_node_features: int = 10,
        num_edge_features: int = 4,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        # Call parent constructor
        super().__init__(
            num_features=num_features,
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            reg_weight=reg_weight,
            cls_weight=cls_weight,
            class_weights=class_weights
        )
        
        # Recreate convolution layers to handle edge attributes
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(self.AttentiveFPLayer(
            num_node_features, 
            hidden_channels, 
            edge_dim=num_edge_features
        ))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(self.AttentiveFPLayer(
                hidden_channels, 
                hidden_channels, 
                edge_dim=num_edge_features
            ))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_idx: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Interpret inputs as graph components
        x = input_ids  # Node features
        edge_index = attention_mask  # Edge indices
        
        # Ensure edge attributes exist
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), 4), device=x.device)
        
        # Debugging logging
        #logging.info(f"Input node features shape: {x.shape}")
        #logging.info(f"Edge index shape: {edge_index.shape}")
        #logging.info(f"Edge attributes shape: {edge_attr.shape}")
        
        # Graph convolutions with edge attributes
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Shared features
        x = self.shared_layer(x)
        
        # Task-specific predictions
        batch_size = feature_idx.size(0)
        regression_preds = torch.zeros(batch_size, device=x.device)
        classification_preds = torch.zeros(batch_size, device=x.device)
        
        for i in range(batch_size):
            idx = feature_idx[i]
            regression_preds[i] = self.regression_heads[idx](x[i])
            classification_preds[i] = self.classification_heads[idx](x[i])
        
        return regression_preds, classification_preds
