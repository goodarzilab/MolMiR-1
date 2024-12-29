"""
Data module for prediction only, without using cached data.
"""

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
import pandas as pd
from typing import Optional, Dict, Any, Union
import logging
from transformers import AutoTokenizer

from src.utils.molecule_utils import smiles_to_graph


class TransformerPredictionDataset(Dataset):
    """Dataset for transformer model predictions."""
    
    def __init__(
        self,
        smiles_df: pd.DataFrame,
        num_features: int,
        model_name: str,
        max_length: int = 512
    ):
        self.smiles_df = smiles_df
        self._n_features = num_features
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Process all SMILES once
        self.tokenized_smiles = []
        self.valid_indices = []
        for idx, row in smiles_df.iterrows():
            try:
                tokens = self.tokenizer(
                    row['Canonical_SMILES'],
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                self.tokenized_smiles.append({
                    'input_ids': tokens['input_ids'].squeeze(0),
                    'attention_mask': tokens['attention_mask'].squeeze(0)
                })
                self.valid_indices.append(idx)
            except Exception as e:
                logging.warning(f"Could not tokenize SMILES: {row['Canonical_SMILES']}, Error: {e}")
                
        if not self.tokenized_smiles:
            raise ValueError("No valid molecules found in input data")
            
        logging.info(f"Successfully processed {len(self.tokenized_smiles)}/{len(smiles_df)} molecules")
        
    def __len__(self) -> int:
        return len(self.tokenized_smiles) * self._n_features
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mol_idx = idx // self._n_features
        feature_idx = idx % self._n_features
        
        tokens = self.tokenized_smiles[mol_idx]
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'feature_idx': torch.tensor(feature_idx, dtype=torch.long),
            'z_score': torch.tensor(0.0),  # Dummy value for prediction
            'is_hit': torch.tensor(0.0),   # Dummy value for prediction
            'mol_idx': torch.tensor(mol_idx, dtype=torch.long)
        }


class GraphPredictionDataset(Dataset):
    """Dataset for graph model predictions."""
    
    def __init__(
        self,
        smiles_df: pd.DataFrame,
        num_features: int
    ):
        super().__init__()
        self.smiles_df = smiles_df
        self._n_features = num_features
        
        # Create graphs for all molecules once
        self.mol_graphs = []
        self.valid_indices = []  # Keep track of which SMILES worked
        for idx, row in smiles_df.iterrows():
            graph = smiles_to_graph(row['Canonical_SMILES'])
            if graph is not None:
                self.mol_graphs.append(graph)
                self.valid_indices.append(idx)
            else:
                logging.warning(f"Could not create graph for SMILES: {row['Canonical_SMILES']}")
        
        if not self.mol_graphs:
            raise ValueError("No valid molecules found in input data")
            
        logging.info(f"Successfully processed {len(self.mol_graphs)}/{len(smiles_df)} molecules")
        
    def __len__(self) -> int:
        return len(self.mol_graphs) * self._n_features
    
    def __getitem__(self, idx: int) -> Data:
        # Calculate which molecule and feature this corresponds to
        mol_idx = idx // self._n_features
        feature_idx = idx % self._n_features
        
        # Get the molecular graph
        mol_graph = self.mol_graphs[mol_idx]
        
        # Create new Data object with all needed attributes
        return Data(
            x=mol_graph.x,
            edge_index=mol_graph.edge_index,
            edge_attr=mol_graph.edge_attr,
            feature_idx=torch.tensor(feature_idx, dtype=torch.long),
            z_score=torch.tensor(0.0),  # Dummy value for prediction
            is_hit=torch.tensor(0.0),   # Dummy value for prediction
            mol_idx=torch.tensor(mol_idx, dtype=torch.long)
        )


class PredictionDataModule(pl.LightningDataModule):
    """DataModule for prediction only."""
    
    def __init__(
        self,
        smiles_file: str,
        num_features: int,
        model_type: str = 'gcn',
        model_name: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 512
    ):
        super().__init__()
        self.smiles_file = smiles_file
        self.num_features = num_features
        self.model_type = model_type
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        
        # Load SMILES data
        self.smiles_df = pd.read_csv(smiles_file, sep='\t')
        required_cols = ['IDNUMBER', 'Canonical_SMILES']
        if not all(col in self.smiles_df.columns for col in required_cols):
            raise ValueError(f"SMILES file must contain columns: {required_cols}")
            
    def setup(self, stage: Optional[str] = None):
        """Set up prediction dataset."""
        if self.model_type == 'transformer':
            if self.model_name is None:
                raise ValueError("model_name must be provided for transformer models")
            self.predict_dataset = TransformerPredictionDataset(
                smiles_df=self.smiles_df,
                num_features=self.num_features,
                model_name=self.model_name,
                max_length=self.max_length
            )
        else:
            self.predict_dataset = GraphPredictionDataset(
                smiles_df=self.smiles_df,
                num_features=self.num_features
            )
    
    def predict_dataloader(self) -> Union[DataLoader, PyGDataLoader]:
        """Create prediction dataloader."""
        if self.model_type == 'transformer':
            return DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
        else:
            return PyGDataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
