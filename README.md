# MolMiR-1

MolMiR-1 is a machine learning framework for molecular miRNA target prediction. It combines transformer-based and graph-based neural networks to predict miRNA-small molecule interactions.

## Features

- Multiple model architectures:
  - Transformer-based models (using ChemBERTa)
  - Graph Neural Networks (GCN, MPNN, GIN, AttentiveFP)
- Multi-task learning for both regression and classification
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
├── configs/                   # Hydra configuration files
│   ├── config.yaml           # Main configuration
│   ├── model/                # Model-specific configs
│   ├── test/                 # Testing configs
│   ├── training/             # Training configs
│   └── wandb/               # Weights & Biases configs
├── src/                      # Source code
│   ├── data/                # Data loading and processing
│   ├── models/              # Model architectures
│   └── utils/               # Utility functions
├── test_results/            # Test evaluation results
├── environment.yml          # Conda environment specification
├── train.py                 # Training script
├── test.py                  # Testing script
└── predict.py               # Prediction script
```

## Usage

### Training

```bash
# Train transformer model
python train.py model.architecture.type=transformer model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM

# Train GCN model
python train.py model.architecture.type=gcn

# Train MPNN model
python train.py model.architecture.type=mpnn
```

### Testing

```bash
# Test transformer model
python test.py model.architecture.foundation_model=DeepChem/ChemBERTa-77M-MLM "+test.checkpoint_path='checkpoints/your_checkpoint.ckpt'"

# Test graph models
python test.py model.architecture.type=gcn "+test.checkpoint_path='checkpoints/your_checkpoint.ckpt'"
python test.py model.architecture.type=mpnn "+test.checkpoint_path='checkpoints/your_checkpoint.ckpt'"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, please contact:
- Hani Goodarzi - [hani@arcinstitute.org](mailto:hani@arcinstitute.org)
