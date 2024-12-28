"""Evaluation metrics for molecular property prediction."""

import torch
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    matthews_corrcoef,
    confusion_matrix
)

def calculate_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate classification metrics for binary predictions.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary containing various classification metrics
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Binary predictions
    y_pred_binary = (y_pred_np >= threshold).astype(int)
    
    # Calculate metrics
    try:
        auroc = roc_auc_score(y_true_np, y_pred_np)
    except ValueError:
        auroc = float('nan')
        
    try:
        auprc = average_precision_score(y_true_np, y_pred_np)
    except ValueError:
        auprc = float('nan')
    
    # Matthews correlation coefficient
    mcc = matthews_corrcoef(y_true_np, y_pred_binary)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_binary).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'mcc': mcc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision': precision,
        'f1_score': f1_score,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

def calculate_regression_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> Dict[str, float]:
    """Calculate regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary containing regression metrics
    """
    # Convert to numpy
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((y_true_np - y_pred_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    
    # R-squared
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate absolute relative error
    relative_errors = np.abs(y_true_np - y_pred_np) / (np.abs(y_true_np) + 1e-8)
    mape = np.mean(relative_errors) * 100  # as percentage
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def get_optimal_threshold(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """Find the optimal classification threshold based on the specified metric.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        metric: Metric to optimize ('f1' or 'mcc')
    
    Returns:
        Tuple of (optimal threshold, best metric value)
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    precisions, recalls, thresholds = precision_recall_curve(y_true_np, y_pred_np)
    
    best_metric = -float('inf')
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_binary = (y_pred_np >= threshold).astype(int)
        
        if metric == 'f1':
            precision = np.sum((y_pred_binary == 1) & (y_true_np == 1)) / (np.sum(y_pred_binary == 1) + 1e-8)
            recall = np.sum((y_pred_binary == 1) & (y_true_np == 1)) / (np.sum(y_true_np == 1) + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            current_metric = f1
        else:  # mcc
            current_metric = matthews_corrcoef(y_true_np, y_pred_binary)
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold
    
    return best_threshold, best_metric

class MetricTracker:
    """Track and accumulate metrics during training/evaluation."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_dict: Dict[str, float], batch_size: int = 1):
        for name, value in metric_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0
                self.counts[name] = 0
            self.metrics[name] += value * batch_size
            self.counts[name] += batch_size
    
    def get_averages(self) -> Dict[str, float]:
        return {
            name: self.metrics[name] / self.counts[name]
            for name in self.metrics
        }
