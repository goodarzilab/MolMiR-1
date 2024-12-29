"""
Ensemble model implementation for MolMir.
"""

import logging
from pathlib import Path
import torch
from typing import Dict, Any, Optional, List, Tuple
import json
from sklearn.model_selection import KFold

from . import MolMir
from .graph_models import MolMirGCN, MolMirMPNN, MolMirGIN, MolMirAttentiveFP

class MolMirEnsemble:
    """Ensemble model that combines multiple MolMir models."""
    
    def __init__(
        self,
        model_type: str,
        model_params: Dict[str, Any],
        num_folds: int = 5,
        ensemble_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize MolMir ensemble.

        Args:
            model_type: Type of model ('transformer', 'gcn', 'mpnn', etc.)
            model_params: Parameters for model initialization
            num_folds: Number of folds for cross-validation
            ensemble_dir: Directory for saving/loading ensemble models
            device: Device to use for computation
        """
        self.model_type = model_type
        self.model_params = model_params
        self.num_folds = num_folds
        self.ensemble_dir = Path(ensemble_dir) if ensemble_dir else None
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Map model types to classes
        self.model_classes = {
            'transformer': MolMir,
            'gcn': MolMirGCN,
            'mpnn': MolMirMPNN,
            'gin': MolMirGIN,
            'attentivefp': MolMirAttentiveFP
        }
        
        if model_type not in self.model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model_class = self.model_classes[model_type]
        self.models = []

    def load_models(self) -> None:
        """Load all models in the ensemble."""
        if not self.ensemble_dir:
            raise ValueError("ensemble_dir must be set to load models")
            
        self.models = []
        
        # Suppress the expected RoBERTa initialization warnings
        import warnings
        warnings.filterwarnings('ignore', message='Some weights of RobertaModel were not initialized')
        
        # Load each fold's model
        for fold_idx in range(self.num_folds):
            fold_info_path = self.ensemble_dir / f'fold_{fold_idx}/fold_{fold_idx}_info.json'
            
            if not fold_info_path.exists():
                logging.warning(f"Missing fold info at {fold_info_path}, trying to find checkpoint directly...")
                # Try to find checkpoint file directly
                checkpoint_files = list(self.ensemble_dir.glob(f'fold_{fold_idx}/*.ckpt'))
                if not checkpoint_files:
                    raise ValueError(f"No checkpoint found for fold {fold_idx}")
                checkpoint_path = checkpoint_files[0]  # Use first checkpoint found
            else:
                with open(fold_info_path, 'r') as f:
                    fold_info = json.load(f)
                checkpoint_path = fold_info['model_path']
            
            logging.info(f"Loading model from: {checkpoint_path}")
            
            # Initialize and load model
            model = self.model_class(**self.model_params)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
            model.to(self.device)
            self.models.append(model)
            
        logging.info(f"Loaded {len(self.models)} models for ensemble")

    def save_fold(self, fold_idx: int, model_path: str, metrics: Dict[str, float]) -> None:
        """
        Save information about a trained fold.
        
        Args:
            fold_idx: Index of the fold
            model_path: Path to saved model checkpoint
            metrics: Dictionary of validation metrics
        """
        if not self.ensemble_dir:
            raise ValueError("ensemble_dir must be set to save folds")
            
        fold_dir = self.ensemble_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        fold_info = {
            'fold_idx': fold_idx,
            'model_path': str(model_path),
            'metrics': metrics
        }
        
        # Save fold info
        with open(fold_dir / 'fold_info.json', 'w') as f:
            json.dump(fold_info, f, indent=2)
    
    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        regression_aggregation: str = "mean",
        classification_aggregation: str = "mean"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions using the ensemble."""
        if not self.models:
            raise ValueError("No models loaded in ensemble")
            
        all_reg_preds = []
        all_cls_preds = []
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Get predictions from each model
        for model in self.models:
            with torch.no_grad():
                if self.model_type == 'transformer':
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
                    
                all_reg_preds.append(reg_preds)
                all_cls_preds.append(cls_preds)
        
        # Stack predictions from all models
        all_reg_preds = torch.stack(all_reg_preds)
        all_cls_preds = torch.stack(all_cls_preds)
        
        # Aggregate predictions
        if regression_aggregation == "mean":
            ensemble_reg_preds = torch.mean(all_reg_preds, dim=0)
        elif regression_aggregation == "median":
            ensemble_reg_preds = torch.median(all_reg_preds, dim=0)[0]
        else:
            raise ValueError(f"Unsupported regression aggregation: {regression_aggregation}")
            
        # Handle classification predictions
        if classification_aggregation == "mean":
            # Average probabilities
            ensemble_cls_probs = torch.sigmoid(all_cls_preds).mean(dim=0)
        elif classification_aggregation == "vote":
            # Majority voting
            ensemble_cls_probs = (torch.sigmoid(all_cls_preds) > 0.5).float().mean(dim=0)
        else:
            raise ValueError(f"Unsupported classification aggregation: {classification_aggregation}")
        
        return ensemble_reg_preds, ensemble_cls_probs

    def predict_with_uncertainty(
        self,
        batch: Dict[str, torch.Tensor],
        regression_aggregation: str = "mean",
        classification_aggregation: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty estimates."""
        if not self.models:
            raise ValueError("No models loaded in ensemble")
            
        all_reg_preds = []
        all_cls_probs = []
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Get predictions from each model
        for model in self.models:
            with torch.no_grad():
                if self.model_type == 'transformer':
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
                
                # Store raw regression predictions
                all_reg_preds.append(reg_preds)
                # Store probabilities for classification
                all_cls_probs.append(torch.sigmoid(cls_preds))
        
        # Stack predictions from all models
        all_reg_preds = torch.stack(all_reg_preds)  # Shape: [num_models, batch_size]
        all_cls_probs = torch.stack(all_cls_probs)  # Shape: [num_models, batch_size]
        
        # Calculate mean predictions
        reg_mean = torch.mean(all_reg_preds, dim=0)
        cls_mean = torch.mean(all_cls_probs, dim=0)
        
        # Calculate uncertainties (standard deviation across models)
        reg_std = torch.std(all_reg_preds, dim=0)
        cls_std = torch.std(all_cls_probs, dim=0)
        
        return {
            'regression_pred': reg_mean,
            'regression_std': reg_std,
            'classification_prob': cls_mean,
            'classification_std': cls_std
        }
        
    @staticmethod
    def get_fold_splits(indices: List[int], num_folds: int = 5) -> List[Tuple[List[int], List[int]]]:
        """
        Create k-fold CV splits.
        
        Args:
            indices: List of indices to split
            num_folds: Number of folds
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        splits = []
        
        for train_idx, val_idx in kf.split(indices):
            splits.append((
                [indices[i] for i in train_idx],
                [indices[i] for i in val_idx]
            ))
            
        return splits
