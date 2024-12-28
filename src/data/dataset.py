"""
Dataset classes for molecular property prediction.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union, Tuple
import logging
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from src.utils.molecule_utils import smiles_to_graph

class MolecularDataset(Dataset):
    """Dataset for molecular property prediction supporting both SMILES and graph representations."""
    def __init__(
        self,
        smiles_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        model_type: str = 'transformer',  # 'transformer' or 'gcn' or 'mpnn'
        model_name: Optional[str] = 'seyonec/ChemBERTa-zinc-base-v1',
        max_length: int = 512,
        z_score_threshold: float = 2.0
    ):
        super().__init__()
        
        # Store parameters
        self.model_type = model_type
        self.model_name = model_name
        self.max_length = max_length
        self.z_score_threshold = z_score_threshold
        
        # Create mappings
        self.smiles_dict = dict(zip(smiles_data['IDNUMBER'], smiles_data['Canonical_SMILES']))
        self.features = sorted(activity_data['feature'].unique())
        self.feature_to_idx = {f: i for i, f in enumerate(self.features)}
        
        # Initialize tokenizer and graph cache
        self._tokenizer = None
        self.graph_cache = {}
        
        # Process SMILES into graphs if using a graph model
        if model_type in ['gcn', 'mpnn']:
            logging.info("Processing SMILES into molecular graphs...")
            for drug_id, smiles in self.smiles_dict.items():
                graph = smiles_to_graph(smiles)
                if graph is not None:
                    self.graph_cache[drug_id] = graph
                else:
                    logging.warning(f"Could not process SMILES for drug {drug_id}")
            logging.info(f"Successfully processed {len(self.graph_cache)} molecules into graphs")
        
        # Prepare data
        self._prepare_data(smiles_data, activity_data)
        logging.info(f"Created dataset with {len(self.data)} samples")
        logging.info(f"Number of unique drugs: {len(self.smiles_dict)}")
        logging.info(f"Number of features: {len(self.features)}")

    def _prepare_data(self, smiles_data: pd.DataFrame, activity_data: pd.DataFrame):
        """Prepare data samples with more robust graph generation."""
        # Reset graph cache
        self.graph_cache = {}
        
        # Log initial processing
        logging.info(f"Total unique drugs in SMILES data: {len(smiles_data['IDNUMBER'].unique())}")
        logging.info(f"Total unique drugs in Activity data: {len(activity_data['drug'].unique())}")
        
        # Generate graphs for all molecules
        drugs_without_graphs = []
        for drug, smiles in zip(smiles_data['IDNUMBER'], smiles_data['Canonical_SMILES']):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graph_cache[drug] = graph
            else:
                drugs_without_graphs.append(drug)
        
        # Log graph generation results
        logging.info(f"Successfully processed graphs for {len(self.graph_cache)} molecules")
        if drugs_without_graphs:
            logging.warning(f"Failed to generate graphs for {len(drugs_without_graphs)} drugs")
            logging.warning(f"Drugs without graphs (first 10): {drugs_without_graphs[:10]}")
        
        # Prepare data samples
        self.data = []
        
        # Track feature-wise sample counts
        feature_sample_counts = {}
        
        # Process matching drugs
        matching_drugs = list(set(self.graph_cache.keys()) & set(activity_data['drug']))
        
        for drug in matching_drugs:
            # Process drug data for features
            drug_data = activity_data[activity_data['drug'] == drug]
            
            for feature in self.features:
                feature_data = drug_data[drug_data['feature'] == feature]
                
                if not feature_data.empty:
                    z_score = feature_data['z.score'].iloc[0]
                    is_hit = float(z_score >= self.z_score_threshold)
                    
                    self.data.append({
                        'drug': drug,
                        'feature': feature,
                        'z_score': z_score,
                        'is_hit': is_hit
                    })
                    
                    # Track feature sample counts
                    feature_sample_counts[feature] = feature_sample_counts.get(feature, 0) + 1
        
        # Logging feature-wise sample distribution
        logging.info("\nFeature-wise Sample Distribution:")
        sorted_features = sorted(feature_sample_counts.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features[:20]:  # Top 20 features
            logging.info(f"{feature}: {count} samples")
        
        # Track hit vs non-hit distribution
        hit_counts = [d['is_hit'] for d in self.data]
        total_hits = sum(hit_counts)
        
        logging.info(f"\nTotal samples: {len(self.data)}")
        logging.info(f"Total hits: {total_hits} ({total_hits/len(self.data)*100:.2f}%)")
        logging.info(f"Total non-hits: {len(self.data) - total_hits} ({(len(self.data) - total_hits)/len(self.data)*100:.2f}%)")

    @property
    def tokenizer(self):
        """Lazy load tokenizer for transformer models."""
        if self._tokenizer is None and self.model_type == 'transformer':
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        return self._tokenizer

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        item = self.data[idx]
        drug = item['drug']
        
        if self.model_type == 'transformer':
            # Prepare transformer input
            smiles = self.smiles_dict[drug]
            encoding = self.tokenizer(
                smiles,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'feature_idx': torch.tensor(self.feature_to_idx[item['feature']]),
                'z_score': torch.tensor(item['z_score'], dtype=torch.float),
                'is_hit': torch.tensor(item['is_hit'], dtype=torch.float)
            }
        
        else:  # graph model
            # Get cached graph
            graph = self.graph_cache[drug]
            
            return {
                'x': graph.x,
                'edge_index': graph.edge_index,
                'edge_attr': graph.edge_attr,
                'num_nodes': graph.num_nodes,
                'feature_idx': torch.tensor(self.feature_to_idx[item['feature']]),
                'z_score': torch.tensor(item['z_score'], dtype=torch.float),
                'is_hit': torch.tensor(item['is_hit'], dtype=torch.float)
            }
    
