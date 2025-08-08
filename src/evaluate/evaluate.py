#!/usr/bin/env python3

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from typing import Dict, Any, List

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import GDMNet
from src.dataloader import HotpotQADataset, GDMNetDataCollator
from src.utils import load_config, setup_logger
from src.evaluate.metrics import MetricsCalculator, compute_metrics


def load_model(model_path: str, config: Dict[str, Any]) -> GDMNet:
    """Load trained model from checkpoint."""
    
    model = GDMNet(**config['model'])
    
    # Load state dict
    if model_path.endswith('.pt') or model_path.endswith('.pth'):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        # Load from PyTorch Lightning checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
    
    return model


def evaluate_model(
    model: GDMNet,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate model on dataset."""
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_outputs = []
    
    logger = setup_logger("Evaluation")
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                elif hasattr(value, 'to'):  # For graph data
                    batch[key] = value.to(device)
            
            # Forward pass
            outputs = model(
                query_input_ids=batch['query_input_ids'],
                query_attention_mask=batch['query_attention_mask'],
                doc_input_ids=batch['doc_input_ids'],
                doc_attention_mask=batch['doc_attention_mask'],
                entity_spans=batch['entity_spans']
            )
            
            # Get predictions
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(batch['labels'].cpu().numpy().tolist())
            
            # Store detailed outputs for analysis
            batch_outputs = {
                'predictions': predictions.cpu().numpy().tolist(),
                'labels': batch['labels'].cpu().numpy().tolist(),
                'logits': logits.cpu().numpy().tolist(),
                'metadata': batch.get('metadata', [])
            }
            all_outputs.append(batch_outputs)
    
    # Compute metrics
    predictions_tensor = torch.tensor(all_predictions)
    labels_tensor = torch.tensor(all_labels)
    
    metrics = compute_metrics(
        predictions_tensor, 
        labels_tensor, 
        config['model']['num_classes']
    )
    
    logger.info("Evaluation completed")
    
    return {
        'metrics': metrics,
        'predictions': all_predictions,
        'labels': all_labels,
        'detailed_outputs': all_outputs
    }


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to file."""
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if key == 'detailed_outputs':
            # Skip detailed outputs for main results file
            continue
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for sub_key, sub_value in value.items():
                if hasattr(sub_value, 'tolist'):
                    serializable_results[key][sub_key] = sub_value.tolist()
                else:
                    serializable_results[key][sub_key] = sub_value
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description='Evaluate GDM-Net model')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to evaluation data (overrides config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for evaluation (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data path if provided
    if args.data_path:
        config['data']['test_path'] = args.data_path
    else:
        config['data']['test_path'] = config['data'].get('test_path', config['data']['val_path'])
    
    # Override batch size if provided
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Setup logging
    logger = setup_logger("GDMNet-Evaluation")
    logger.info("Starting GDM-Net evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {config['data']['test_path']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path, config)
    logger.info("Model loaded successfully")
    
    # Create dataset and dataloader
    logger.info("Loading dataset...")
    dataset = HotpotQADataset(
        data_path=config['data']['test_path'],
        tokenizer_name=config['model']['bert_model_name'],
        max_length=config['data']['max_length'],
        max_query_length=config['data']['max_query_length'],
        num_entities=config['model']['num_entities']
    )
    
    data_collator = GDMNetDataCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        collate_fn=data_collator,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Evaluate model
    results = evaluate_model(model, dataloader, device, config)
    
    # Print metrics
    metrics_calculator = MetricsCalculator(config['model']['num_classes'])
    metrics_calculator.print_metrics(results['metrics'])
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    save_results(results, results_path)
    logger.info(f"Results saved to: {results_path}")
    
    # Save detailed outputs
    detailed_path = os.path.join(args.output_dir, 'detailed_outputs.json')
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results['detailed_outputs'], f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed outputs saved to: {detailed_path}")
    
    return results


if __name__ == "__main__":
    main()
