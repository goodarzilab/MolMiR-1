# MolMiR-1

MolMiR-1 is a machine learning framework for molecular miRNA target prediction. It combines transformer-based and graph-based neural networks to predict miRNA-small molecule interactions.

## Features

- Multiple model architectures:
  - Transformer-based models (using ChemBERTa)
  - Graph Neural Networks (GCN, MPNN, GIN, AttentiveFP)
- Multi-task learning for both regression and classification
- Ensemble models with uncertainty estimation
  - K-fold cross-validation ensemble training
  - Model uncertainty quantification
  - Ensemble prediction aggregation
- Comprehensive evaluation metrics
- Hydra configuration system
- PyTorch Lightning training framework

## Installation

```bash
# Clone the repository
git clone https://github.com/goodarzilab/MolMiR-1.git
cd MolMiR-1

# Create and activate a conda environment
conda env create -f environment.yml
conda activate molmir
```

## Project Structure

```
MolMiR-1/
├── cache/                      # Cached preprocessed data
│   ├── processed_dataset_*.pt  # Model-specific processed datasets
│   └── split_info.json        # Train/val/test split information
├── checkpoints/               # Model checkpoints
│   └── ensemble/             # Ensemble model checkpoints
├── configs/                   # Hydra configuration files
│   ├── config.yaml           # Main configuration
│   ├── model/                # Model-specific configs
│   ├── test/                 # Testing configs
│   ├── training/             # Training configs
│   └── wandb/               # Weights & Biases configs
├── src/                      # Source code
│   ├── data/                # Data loading and processing
│   ├── models/              # Model architectures
│   │   └── ensemble.py     # Ensemble model implementation
│   └── utils/               # Utility functions
├── test_results/            # Test evaluation results
├── predictions/             # Model predictions output
├── environment.yml          # Conda environment specification
├── train.py                 # Single model training script
├── test.py                  # Single model testing script
├── predict.py               # Single model prediction script
├── train_ensemble.py        # Ensemble training script
├── test_ensemble.py         # Ensemble testing script
└── predict_ensemble.py      # Ensemble prediction script
```

## Usage

### Single Model Training

```bash
# Train transformer model
python train.py model.architecture.type=transformer model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM

# Train GCN model
python train.py model.architecture.type=gcn

# Train MPNN model
python train.py model.architecture.type=mpnn
```

### Ensemble Model Training

```bash
# Train transformer ensemble
python train_ensemble.py model.architecture.type=transformer \
    model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM \
    training.ensemble.enabled=true

# Train GCN ensemble
python train_ensemble.py model.architecture.type=gcn \
    training.ensemble.enabled=true
```

### Testing

```bash
# Test single transformer model
python test.py model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM \
    "+test.checkpoint_path='checkpoints/your_checkpoint.ckpt'"

# Test transformer ensemble
python test_ensemble.py model.architecture.type=transformer \
    test.ensemble.model_dir=checkpoints/ensemble/ChemBERTa-77M-MLM \
    test.ensemble.enabled=true

# Test graph models (single and ensemble)
python test.py model.architecture.type=gcn "+test.checkpoint_path='checkpoints/your_checkpoint.ckpt'"
python test_ensemble.py model.architecture.type=gcn test.ensemble.model_dir=checkpoints/ensemble/gcn_model test.ensemble.enabled=true
```

### Prediction

```bash
# Single model prediction
python predict.py model.architecture.type=transformer \
    predict.checkpoint_path=checkpoints/your_checkpoint.ckpt \
    predict.smiles_file=test/test_SMILES.tsv

# Ensemble prediction with uncertainty
python predict_ensemble.py model.architecture.type=transformer \
    predict.ensemble.model_dir=checkpoints/ensemble/ChemBERTa-77M-MLM \
    predict.ensemble.enabled=true \
    predict.smiles_file=test/test_SMILES.tsv
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, please contact:
- Hani Goodarzi - [hani@arcinstitute.org](mailto:hani@arcinstitute.org)
