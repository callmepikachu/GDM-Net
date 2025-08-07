# GDM-Net: Graph-Augmented Dual Memory Network

A PyTorch implementation of Graph-Augmented Dual Memory Network for Multi-Document Understanding.

## Project Structure

```
GDM-Net/
├── src/                    # Source code
│   ├── model/             # Model components
│   ├── dataloader/        # Data loading utilities
│   ├── train/             # Training scripts
│   ├── evaluate/          # Evaluation utilities
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── data/                  # Dataset files
├── logs/                  # Training logs
└── README.md             # This file
```

## Model Components

### Core Architecture
- **Graph Memory Module**: Maintains structured knowledge representation
- **Dual Memory System**: Combines episodic and semantic memory
- **Multi-hop Reasoning**: Performs iterative reasoning over graph structures
- **Document Encoder**: BERT-based text encoding

### Key Features
- Multi-document understanding
- Graph-based knowledge representation
- Attention-based fusion mechanisms
- End-to-end trainable architecture

## Usage

### Training
```bash
python src/train/train.py --config config/default_config.yaml
```

### Evaluation
```bash
python src/evaluate/evaluate.py --model_path checkpoints/best_model.pt
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Lightning
- PyTorch Geometric
- Transformers
- Other dependencies in requirements.txt

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gdmnet2024,
  title={Graph-Augmented Dual Memory Network for Multi-Document Understanding},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
