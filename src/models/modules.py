"""Additional model components and custom modules for MolMir."""

import torch
import torch.nn as nn
from typing import Optional

class MolecularEncoder(nn.Module):
    """Encoder module that processes molecular representations."""
    def __init__(
        self,
        hidden_size: int,
        dropout_prob: float = 0.1,
        num_attention_heads: int = 8,
        intermediate_size: Optional[int] = None
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
            
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        attended, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.layer_norm1(x + attended)
        
        # Feed-forward with residual connection
        x = self.layer_norm2(x + self.feed_forward(x))
        return x

class TaskSpecificHead(nn.Module):
    """Task-specific prediction head with optional multi-task support."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size // 2
            
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_size))
            
        layers.extend([
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size)
        ])
        
        self.head = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class FeatureWiseAttention(nn.Module):
    """Attention mechanism for feature-wise weighting."""
    def __init__(self, hidden_size: int, num_features: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, num_features)
        
    def forward(self, x: torch.Tensor, feature_idx: torch.Tensor) -> torch.Tensor:
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(x), dim=-1)
        
        # Select weights for specific features
        batch_size = feature_idx.size(0)
        selected_weights = attention_weights[torch.arange(batch_size), feature_idx]
        
        return selected_weights.unsqueeze(-1) * x
