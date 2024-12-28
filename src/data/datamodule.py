"""
PyTorch Lightning DataModule for molecular property prediction.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import pandas as pd
from pathlib import Path
import logging
import json
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from .dataset import MolecularDataset

def collate_graphs(batch):
    """Custom collate function for graph batches with error handling."""
    graphs = {
        'x': [],
        'edge_index': [],
        'edge_attr': [],
        'batch_idx': [],
        'feature_idx': [],
        'z_score': [],
        'is_hit': []
    }
    
    # Process each item in batch
    batch_size = len(batch)
    cumsum_nodes = 0
    valid_samples = []
    
    for i, data in enumerate(batch):
        try:
            # Verify required keys exist
            if not all(key in data for key in ['x', 'edge_index']):
                logging.warning(f"Sample {i} missing required keys. Keys present: {data.keys()}")
                continue
                
            num_nodes = data['x'].size(0)
            
            # Node features
            graphs['x'].append(data['x'])
            
            # Edge attributes (ensure tensor exists)
            edge_attr = data.get('edge_attr', torch.zeros((data['edge_index'].size(1), 4), dtype=torch.float))
            graphs['edge_attr'].append(edge_attr)
            
            # Shift edge indices by cumsum of previous nodes
            edge_index = data['edge_index'] + cumsum_nodes
            graphs['edge_index'].append(edge_index)
            
            # Create batch assignment for each node
            graphs['batch_idx'].extend([i] * num_nodes)
            
            # Task-specific data
            graphs['feature_idx'].append(data['feature_idx'])
            graphs['z_score'].append(data['z_score'])
            graphs['is_hit'].append(data['is_hit'])
            
            cumsum_nodes += num_nodes
            valid_samples.append(i)
            
        except Exception as e:
            logging.error(f"Error processing sample {i}: {str(e)}")
            continue
    
    # Only proceed if we have valid samples
    if not valid_samples:
        raise RuntimeError("No valid samples in batch")
    
    try:
        # Concatenate everything
        return {
            'x': torch.cat(graphs['x'], dim=0),
            'edge_index': torch.cat(graphs['edge_index'], dim=1),
            'edge_attr': torch.cat(graphs['edge_attr'], dim=0),
            'batch': torch.tensor(graphs['batch_idx'], dtype=torch.long),
            'feature_idx': torch.stack(graphs['feature_idx']),
            'z_score': torch.stack(graphs['z_score']),
            'is_hit': torch.stack(graphs['is_hit'])
        }
    except Exception as e:
        logging.error(f"Error concatenating batch: {str(e)}")
        logging.error(f"Graph shapes - x: {[g.shape for g in graphs['x']]}, "
                     f"edge_index: {[g.shape for g in graphs['edge_index']]}")
        raise

class MolecularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        smiles_path: str,
        activity_path: str,
        model_type: str = 'transformer',
        model_name: Optional[str] = 'seyonec/ChemBERTa-zinc-base-v1',
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 512,
        z_score_threshold: float = 2.0,
        train_size: float = 0.8,
        val_size: float = 0.05,  # Test size will be 1 - train_size - val_size
        random_seed: int = 42,
        auto_weight: bool = True,
        cache_only: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize attributes
        self.cache_only = cache_only
        self.num_features = None
        self.class_weights = None
        self.cls_weight = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Create cache directory
        Path("cache").mkdir(exist_ok=True)

    def _get_cache_paths(self) -> Dict[str, Path]:
        """Get paths for various cache files."""
        cache_dir = Path("cache")
        # Simplified path construction - just use model_type for graph models
        cache_path = cache_dir / f"processed_dataset_{self.hparams.model_type}.pt"
    
        logging.info(f"Looking for cache at: {cache_path}")
    
        return {
            'dataset': cache_path,
            'split_info': cache_dir / "split_info.json",
            'weights': cache_dir / f"class_weights_z{self.hparams.z_score_threshold}.pt"
        }

    def _split_drugs(self, dataset: MolecularDataset) -> Tuple[List[str], List[str], List[str]]:
        """Split drugs into train/val/test sets."""
        # Get unique drugs
        unique_drugs = sorted(list(set(item['drug'] for item in dataset.data)))
        logging.info(f"\nTotal number of unique drugs: {len(unique_drugs)}")
        
        # First split into train and temp
        train_drugs, remaining = train_test_split(
            unique_drugs,
            train_size=self.hparams.train_size,
            random_state=self.hparams.random_seed
        )
        
        # Split remaining into val and test
        remaining_total = len(remaining)
        val_size_relative = self.hparams.val_size / (1 - self.hparams.train_size)
        val_drugs, test_drugs = train_test_split(
            remaining,
            train_size=val_size_relative,
            random_state=self.hparams.random_seed
        )
        
        # Calculate actual split sizes
        total = len(unique_drugs)
        test_size = 1 - self.hparams.train_size - self.hparams.val_size
        
        # Log split information
        logging.info("\nDrug Split:")
        logging.info(f"Train: {len(train_drugs)} drugs ({self.hparams.train_size*100:.1f}%)")
        logging.info(f"Val: {len(val_drugs)} drugs ({self.hparams.val_size*100:.1f}%)")
        logging.info(f"Test: {len(test_drugs)} drugs ({test_size*100:.1f}%)")

        # Save split information
        split_info = {
            'train_drugs': train_drugs,
            'val_drugs': val_drugs,
            'test_drugs': test_drugs,
            'total_drugs': total,
            'train_size': self.hparams.train_size,
            'val_size': self.hparams.val_size,
            'test_size': test_size,
            'random_seed': self.hparams.random_seed
        }
        
        # Ensure cache directory exists
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Save split information to JSON
        split_info_path = cache_dir / "split_info.json"
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=4)
        
        logging.info(f"\nSplit information saved to {split_info_path}")
        
        return train_drugs, val_drugs, test_drugs

    def setup(self, stage: Optional[str] = None):
        """Set up the data module."""
        cache_paths = self._get_cache_paths()
        
        try:
            if not cache_paths['dataset'].exists():
                if self.cache_only:
                    raise RuntimeError(
                        "Cache-only mode is set but no cached dataset found. "
                        "Please run training first to generate the cache."
                    )
                
                # Load and process data
                logging.info(f"Loading SMILES data from {self.hparams.smiles_path}")
                logging.info(f"Loading activity data from {self.hparams.activity_path}")
                
                smiles_data = pd.read_csv(self.hparams.smiles_path, sep='\t')
                activity_data = pd.read_csv(self.hparams.activity_path)
                
                logging.info("Creating dataset...")
                dataset = MolecularDataset(
                    smiles_data=smiles_data,
                    activity_data=activity_data,
                    model_type=self.hparams.model_type,
                    model_name=self.hparams.model_name,
                    max_length=self.hparams.max_length,
                    z_score_threshold=self.hparams.z_score_threshold
                )
                self.num_features = len(dataset.features)
                
                # Cache dataset
                logging.info("Saving dataset to cache...")
                torch.save({
                    'dataset': dataset,
                    'num_features': self.num_features,
                    'model_type': self.hparams.model_type
                }, cache_paths['dataset'])
                logging.info("Dataset cached successfully")
                
            else:
                # Load from cache
                logging.info(f"Loading preprocessed dataset from cache: {cache_paths['dataset']}")
                cached_data = torch.load(cache_paths['dataset'])

                dataset = cached_data['dataset']
                self.num_features = cached_data['num_features']
                
                # Verify model type matches
                #if cached_data['model_type'] != self.hparams.model_type:
                #    raise ValueError(
                #        f"Cached dataset is for {cached_data['model_type']} models "
                #        f"but trying to use with {self.hparams.model_type} model"
                #    )
                    
                logging.info(f"Loaded dataset with {len(dataset)} samples and {self.num_features} features")
            
            # Split drugs and create dataset splits
            train_drugs, val_drugs, test_drugs = self._split_drugs(dataset)
            
            # Create indices for each split
            train_indices = []
            val_indices = []
            test_indices = []
            
            # Track feature coverage
            feature_counts = {split: {feature: 0 for feature in dataset.features} 
                          for split in ['train', 'val', 'test']}
            
            # Assign data points based on drugs
            for idx, item in enumerate(dataset.data):
                drug = item['drug']
                feature = item['feature']
                
                if drug in test_drugs:
                    test_indices.append(idx)
                    feature_counts['test'][feature] += 1
                elif drug in val_drugs:
                    val_indices.append(idx)
                    feature_counts['val'][feature] += 1
                else:
                    train_indices.append(idx)
                    feature_counts['train'][feature] += 1
            
            # Create dataset splits
            self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
            self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
            self.test_dataset = torch.utils.data.Subset(dataset, test_indices)
            
            # Calculate class weights if needed
            if self.hparams.auto_weight and not cache_paths['weights'].exists():
                logging.info("Calculating class weights...")
                all_hits = torch.tensor([item['is_hit'] for item in dataset.data])
                pos_count = torch.sum(all_hits)
                neg_count = len(all_hits) - pos_count
                
                pos_weight = len(all_hits) / (2 * pos_count)
                neg_weight = len(all_hits) / (2 * neg_count)
                
                self.class_weights = torch.tensor([neg_weight, pos_weight])
                self.cls_weight = torch.mean(self.class_weights).item()
                
                # Cache weights
                torch.save({
                    'class_weights': self.class_weights,
                    'cls_weight': self.cls_weight,
                    'pos_count': pos_count,
                    'neg_count': neg_count
                }, cache_paths['weights'])
            
            # Log split information
            logging.info("\nDataset Split Summary:")
            for split_name, indices in [('Train', train_indices), ('Val', val_indices), ('Test', test_indices)]:
                logging.info(f"{split_name}: {len(indices)} samples")
                features_with_data = sum(1 for count in feature_counts[split_name.lower()].values() if count > 0)
                logging.info(f"{split_name} features with data: {features_with_data}/{self.num_features}")
            
        except Exception as e:
            logging.error(f"Error during data setup: {e}")
            raise

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=None if self.hparams.model_type == 'transformer' else collate_graphs
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=None if self.hparams.model_type == 'transformer' else collate_graphs
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=None if self.hparams.model_type == 'transformer' else collate_graphs
        )
