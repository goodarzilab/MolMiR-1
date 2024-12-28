"""Utility functions for molecular graph processing."""

import torch
from torch_geometric.data import Data
from rdkit import Chem
import logging
import numpy as np
from typing import Optional, Dict, List

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric graph with comprehensive error handling."""
    if not isinstance(smiles, str):
        logging.error(f"Invalid SMILES input type: {type(smiles)}")
        return None
        
    try:
        # Attempt multiple parsing strategies
        mol = None
        parsing_methods = [
            lambda s: Chem.MolFromSmiles(s, sanitize=True),
            lambda s: Chem.MolFromSmiles(s, sanitize=False),
            lambda s: Chem.MolFromSmiles(s)
        ]
        
        for parse_method in parsing_methods:
            try:
                mol = parse_method(smiles)
                if mol is not None:
                    break
            except Exception as e:
                logging.debug(f"Parsing method failed: {str(e)}")
                continue
        
        if mol is None:
            logging.warning(f"Could not parse SMILES: {smiles}")
            return None

        # Ensure molecule has atoms
        if mol.GetNumAtoms() == 0:
            logging.warning(f"Molecule has no atoms: {smiles}")
            return None

        # Feature generation with verification
        x = []
        try:
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),           # 0. Atomic number
                    atom.GetDegree(),              # 1. Degree
                    atom.GetFormalCharge(),        # 2. Formal charge
                    atom.GetNumImplicitHs(),       # 3. Implicit hydrogens
                    int(atom.GetIsAromatic()),     # 4. Aromaticity
                    int(atom.IsInRing()),          # 5. In ring
                    int(atom.IsInRingSize(5)),     # 6. In 5-membered ring
                    int(atom.IsInRingSize(6)),     # 7. In 6-membered ring
                    int(atom.GetHybridization()),  # 8. Hybridization (as int)
                    atom.GetTotalNumHs(),          # 9. Total hydrogens
                ]
                x.append(features)
        except Exception as e:
            logging.error(f"Error generating atom features for {smiles}: {str(e)}")
            return None
            
        if not x:
            logging.warning(f"No valid atoms found in molecule: {smiles}")
            return None

        # Convert to tensor with explicit dtype
        x = torch.tensor(x, dtype=torch.float)
        
        # Edge processing with verification
        edge_indices = []
        edge_features = []
        
        # Handle molecules with no bonds
        if mol.GetNumBonds() == 0:
            # Create self-loops for isolated atoms
            for i in range(mol.GetNumAtoms()):
                edge_indices.append([i, i])
                edge_features.append([1.0, 0, 0, 0])  # Single bond features
        else:
            try:
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    
                    # Verify indices are valid
                    if max(i, j) >= mol.GetNumAtoms():
                        raise ValueError(f"Invalid atom indices: {i}, {j}")
                    
                    # Add both directions
                    edge_indices.extend([[i, j], [j, i]])
                    
                    # Edge features
                    features = [
                        float(bond.GetBondTypeAsDouble()),
                        float(bond.GetIsConjugated()),
                        float(bond.IsInRing()),
                        float(bond.GetIsAromatic())
                    ]
                    edge_features.extend([features, features])
            except Exception as e:
                logging.error(f"Error processing bonds for {smiles}: {str(e)}")
                return None

        # Create tensors with explicit dtypes
        if not edge_indices:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Verify tensor shapes
        if edge_index.shape[0] != 2:
            logging.error(f"Invalid edge_index shape: {edge_index.shape}")
            return None
            
        # Create and verify the graph
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Verify graph integrity
        if not graph.validate():
            logging.error(f"Invalid graph created for SMILES: {smiles}")
            return None
            
        return graph
        
    except Exception as e:
        logging.error(f"Unexpected error converting SMILES to graph: {smiles}")
        logging.error(f"Error details: {str(e)}")
        return None
