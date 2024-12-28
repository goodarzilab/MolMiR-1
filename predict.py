"""
Batch inference script for making predictions with MolMir model.
"""

import os
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
from transformers import AutoTokenizer
import json
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.models import MolMir

def setup_logging(output_dir: str = "predictions") -> None:
    """Set up logging and create output directory."""
    logging.basicConfig(level=logging.INFO)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def load_model(checkpoint_path: str) -> Tuple[MolMir, str]:
    """Load model and get model name from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Loading model from {checkpoint_path}")
    model = MolMir.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    model_name = model.hparams.model_name
    return model, model_name

def load_features(activity_path: str) -> List[str]:
    """Load and sort feature names from activity data."""
    activity_data = pd.read_csv(activity_path)
    return sorted(activity_data['feature'].unique())

def load_smiles(smiles_path: str) -> List[str]:
    """Load SMILES from file, one per line."""
    with open(smiles_path) as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(smiles_list)} SMILES strings")
    return smiles_list

def process_smiles_batch(
    smiles_list: List[str],
    model_name: str,
    max_length: int,
    batch_size: int = 32
) -> List[Dict[str, torch.Tensor]]:
    """Process a list of SMILES strings into batched model inputs."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    batched_inputs = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        encodings = tokenizer(
            batch_smiles,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        batched_inputs.append(encodings)
    
    return batched_inputs

def make_predictions(
    model: MolMir,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    features: List[str]
) -> pd.DataFrame:
    """Make predictions for all features for a batch of compounds."""
    model.eval()
    results = []
    batch_size = input_ids.size(0)
    
    # Move inputs to model's device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    with torch.no_grad():
        # Create feature indices tensor
        feature_indices = torch.arange(len(features), dtype=torch.long, device=model.device)
        
        # Get predictions for each feature
        for idx in feature_indices:
            # Expand feature index for batch
            batch_idx = idx.repeat(batch_size)
            
            reg_pred, cls_pred = model(
                input_ids,
                attention_mask,
                batch_idx
            )
            
            probs = torch.sigmoid(cls_pred)
            
            for b in range(batch_size):
                results.append({
                    'smiles_idx': b,
                    'feature': features[idx],
                    'z_score': reg_pred[b].item(),
                    'probability': probs[b].item()
                })
    
    return pd.DataFrame(results)

def save_predictions(
    predictions_df: pd.DataFrame,
    smiles_list: List[str],
    model_name: str,
    output_dir: str = "predictions"
) -> None:
    """Save predictions to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = f"{output_dir}/predictions_{model_name}_{timestamp}"
    
    # Add SMILES column
    predictions_df['smiles'] = predictions_df['smiles_idx'].map(lambda x: smiles_list[x])
    
    # Reshape data for better visualization
    pivot_df = predictions_df.pivot(
        index='smiles',
        columns='feature',
        values=['z_score', 'probability']
    )
    
    # Save detailed predictions
    predictions_csv = f"{results_base}_detailed.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    
    # Save pivoted predictions
    pivot_csv = f"{results_base}_pivot.csv"
    pivot_df.to_csv(pivot_csv)
    
    # Save as Excel with formatting
    excel_path = f"{results_base}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Save detailed view
        predictions_df.to_excel(writer, sheet_name='Detailed', index=False)
        
        # Save z-scores
        z_scores = pivot_df['z_score']
        z_scores.to_excel(writer, sheet_name='Z-scores')
        
        # Save probabilities with conditional formatting
        probs = pivot_df['probability']
        probs.to_excel(writer, sheet_name='Probabilities')
        
        # Add conditional formatting to probabilities
        worksheet = writer.sheets['Probabilities']
        from openpyxl.formatting.rule import ColorScaleRule
        color_scale = ColorScaleRule(
            start_type='min',
            start_color='FFFFFF',
            end_type='max',
            end_color='FF0000'
        )
        worksheet.conditional_formatting.add(
            f"B2:{chr(65 + len(features))}{len(smiles_list)+1}",
            color_scale
        )
    
    # Save metadata
    metadata = {
        'num_compounds': len(smiles_list),
        'num_features': len(set(predictions_df['feature'])),
        'model_name': model_name,
        'prediction_time': timestamp,
        'files': {
            'detailed_csv': predictions_csv,
            'pivot_csv': pivot_csv,
            'excel': excel_path
        }
    }
    
    metadata_path = f"{results_base}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved detailed predictions to {predictions_csv}")
    logging.info(f"Saved pivoted predictions to {pivot_csv}")
    logging.info(f"Saved formatted Excel to {excel_path}")
    logging.info(f"Saved metadata to {metadata_path}")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main inference function."""
    # Setup
    setup_logging()
    
    # Load model
    model, model_name = load_model(cfg.predict.checkpoint_path)
    
    # Load features
    features = load_features(cfg.data.activity_path)
    
    # Load SMILES
    smiles_list = load_smiles(cfg.predict.smiles_file)
    
    # Process SMILES in batches
    logging.info("Processing SMILES and making predictions...")
    all_predictions = []
    batched_inputs = process_smiles_batch(
        smiles_list,
        model_name,
        max_length=cfg.model.architecture.max_length,
        batch_size=cfg.training.batch_size
    )
    
    for batch in tqdm(batched_inputs, desc="Processing batches"):
        batch_predictions = make_predictions(
            model,
            batch['input_ids'],
            batch['attention_mask'],
            features
        )
        all_predictions.append(batch_predictions)
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Save results
    save_predictions(predictions_df, smiles_list, model_name.split('/')[-1])
    
    # Print summary statistics
    logging.info("\nSummary Statistics:")
    logging.info(f"Processed {len(smiles_list)} compounds")
    logging.info(f"Made predictions for {len(features)} features")
    logging.info(f"Mean Z-score: {predictions_df['z_score'].mean():.3f} ± {predictions_df['z_score'].std():.3f}")
    logging.info(f"Mean Probability: {predictions_df['probability'].mean():.3f} ± {predictions_df['probability'].std():.3f}")
    logging.info(f"\nResults saved in predictions directory")

if __name__ == '__main__':
    main()

#python predict.py predict.checkpoint_path=/path/to/checkpoint.ckpt predict.smiles_file=/path/to/smiles.txt
