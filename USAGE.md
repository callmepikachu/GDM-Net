# GDM-Net Usage Guide

This guide provides detailed instructions on how to use the GDM-Net (Graph-Augmented Dual Memory Network) for multi-document understanding tasks.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd GDM-Net

# Install dependencies and setup
make setup
```

### 2. Test Installation

```bash
# Run installation tests
make test
```

### 3. Create Sample Data and Train

```bash
# Train with synthetic data
make train
```

### 4. Run Examples

```bash
# Run usage examples
make example
```

## Detailed Usage

### Data Format

GDM-Net expects data in the following JSON format:

```json
{
  "document": "Apple Inc. is a technology company founded by Steve Jobs. Tim Cook is the current CEO.",
  "query": "Who is the CEO of Apple?",
  "entities": [
    {"span": [0, 9], "type": "ORGANIZATION"},
    {"span": [50, 60], "type": "PERSON"},
    {"span": [62, 70], "type": "PERSON"}
  ],
  "relations": [
    {"head": 0, "tail": 2, "type": "CEO_OF"},
    {"head": 0, "tail": 1, "type": "FOUNDED_BY"}
  ],
  "label": 0
}
```

### Model Configuration

Edit `config/model_config.yaml` to customize the model:

```yaml
model:
  bert_model_name: "bert-base-uncased"
  hidden_size: 768
  num_entities: 10
  num_relations: 20
  num_classes: 5
  gnn_type: "rgcn"  # or "gat"
  fusion_method: "gate"  # "concat", "add", "gate", "attention"
```

### Training

#### Basic Training

```bash
python train/train.py --config config/model_config.yaml --mode train
```

#### With Custom Data

```bash
# Prepare your data files
# data/train.json, data/val.json, data/test.json

python train/train.py --config config/model_config.yaml --mode train
```

#### Advanced Training Options

```bash
# Create synthetic data first
python train/train.py --config config/model_config.yaml --create_synthetic

# Train with custom configuration
python train/train.py --config my_config.yaml --mode train
```

### Evaluation

```bash
# Evaluate a trained model
python train/train.py \
  --config config/model_config.yaml \
  --mode eval \
  --model_path checkpoints/gdmnet-epoch=05-val_loss=0.25.ckpt
```

### Inference

#### Programmatic Usage

```python
from gdmnet import GDMNet
from transformers import BertTokenizer

# Load trained model
model = GDMNet.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare input
document = "Apple Inc. is a technology company founded by Steve Jobs."
query = "Who founded Apple?"

doc_encoding = tokenizer(
    document,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

query_encoding = tokenizer(
    query,
    max_length=64,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Inference
with torch.no_grad():
    outputs = model(
        input_ids=doc_encoding['input_ids'],
        attention_mask=doc_encoding['attention_mask'],
        query=query_encoding['input_ids'],
        return_intermediate=True
    )

# Get results
prediction = torch.argmax(outputs['logits'], dim=-1)
entities = outputs['entities']
relations = outputs['relations']
```

## Model Components

### 1. Document Encoder
- Uses BERT for encoding input documents
- Configurable model size and dropout

### 2. Structure Extractor
- Extracts entities and relations from documents
- Supports both token-level and span-level extraction

### 3. Graph Memory
- Maintains graph structure using RGCN or GAT
- Supports dynamic memory updates

### 4. Path Finder
- Discovers reasoning paths in the graph
- Multi-hop reasoning capabilities

### 5. Graph Reader
- Reads relevant information from graph memory
- Attention-based information retrieval

### 6. Reasoning Fusion
- Combines document and graph representations
- Multiple fusion strategies available

## Configuration Options

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `bert_model_name` | Pre-trained BERT model | `bert-base-uncased` |
| `hidden_size` | Hidden dimension | `768` |
| `num_entities` | Number of entity types | `10` |
| `num_relations` | Number of relation types | `20` |
| `num_classes` | Number of output classes | `5` |
| `gnn_type` | Graph neural network type | `rgcn` |
| `fusion_method` | Representation fusion method | `gate` |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_epochs` | Maximum training epochs | `10` |
| `batch_size` | Training batch size | `8` |
| `learning_rate` | Learning rate | `2e-5` |
| `dropout_rate` | Dropout rate | `0.1` |

## Advanced Features

### Custom Entity and Relation Types

```python
# Define custom types
entity_types = {
    'O': 0,
    'PERSON': 1,
    'COMPANY': 2,
    'PRODUCT': 3
}

relation_types = {
    'NO_RELATION': 0,
    'WORKS_FOR': 1,
    'FOUNDED': 2,
    'PRODUCES': 3
}

# Use in dataset
dataset = GDMNetDataset(
    data_path="data/train.json",
    entity_types=entity_types,
    relation_types=relation_types
)
```

### Multi-GPU Training

```yaml
# In config file
training:
  accelerator: "gpu"
  devices: 2  # Use 2 GPUs
  strategy: "ddp"  # Distributed training
```

### Experiment Tracking

```yaml
# Use Weights & Biases
logging:
  type: "wandb"
  project: "gdmnet-experiments"
  name: "experiment-1"
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable 16-bit precision

2. **Slow training**
   - Increase number of workers
   - Use faster storage (SSD)
   - Enable mixed precision training

3. **Poor performance**
   - Check data quality and format
   - Adjust learning rate
   - Increase model size or training epochs

### Performance Tips

1. **Data Loading**
   ```yaml
   training:
     num_workers: 8  # Increase for faster data loading
     pin_memory: true
   ```

2. **Memory Optimization**
   ```yaml
   training:
     precision: 16  # Use half precision
     accumulate_grad_batches: 4  # Gradient accumulation
   ```

3. **Model Optimization**
   ```yaml
   model:
     dropout_rate: 0.1  # Prevent overfitting
   advanced:
     encoder:
       freeze_bert: true  # Freeze BERT for faster training
   ```

## Examples

See the `examples/` directory for complete usage examples:

- `example_usage.py`: Basic model usage and training
- More examples coming soon...

## Support

For questions and issues:
1. Check this documentation
2. Review the example code
3. Check the GitHub issues
4. Create a new issue with detailed information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.
