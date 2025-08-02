"""
Example Usage of GDM-Net

This script demonstrates how to use the GDM-Net model for various tasks
including initialization, training, and inference.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import BertTokenizer
import json
import numpy as np
from typing import List, Dict, Any

# Import GDM-Net components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdmnet import GDMNet


class ExampleDataset(Dataset):
    """
    Example dataset for demonstrating GDM-Net usage.
    
    This dataset simulates multi-document reasoning tasks with entities and relations.
    """
    
    def __init__(self, tokenizer, max_length=512, num_samples=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic data for demonstration."""
        data = []
        
        # Sample documents and queries
        sample_docs = [
            "Apple Inc. is a technology company founded by Steve Jobs. Tim Cook is the current CEO.",
            "Microsoft Corporation was founded by Bill Gates. Satya Nadella is the current CEO.",
            "Google was founded by Larry Page and Sergey Brin. Sundar Pichai is the current CEO.",
            "Tesla Inc. was founded by Elon Musk. The company focuses on electric vehicles.",
            "Amazon was founded by Jeff Bezos. Andy Jassy is the current CEO."
        ]
        
        sample_queries = [
            "Who is the CEO of Apple?",
            "Which company was founded by Bill Gates?",
            "What does Tesla focus on?",
            "Who founded Google?",
            "Who is the current CEO of Amazon?"
        ]
        
        # Generate samples
        for i in range(self.num_samples):
            doc_idx = i % len(sample_docs)
            query_idx = i % len(sample_queries)
            
            sample = {
                'document': sample_docs[doc_idx],
                'query': sample_queries[query_idx],
                'label': i % 5,  # Random label for demonstration
                'entities': [
                    {'span': [0, 5], 'type': 1},  # Company
                    {'span': [50, 60], 'type': 2}  # Person
                ],
                'relations': [
                    {'head': 0, 'tail': 1, 'type': 1}  # CEO_OF relation
                ]
            }
            
            data.append(sample)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize document and query
        doc_encoding = self.tokenizer(
            sample['document'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        query_encoding = self.tokenizer(
            sample['query'],
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create entity labels (simplified)
        entity_labels = torch.full((self.max_length,), -1, dtype=torch.long)
        for entity in sample['entities']:
            start, end = entity['span']
            if start < self.max_length:
                entity_labels[start:min(end, self.max_length)] = entity['type']
        
        # Create relation labels (simplified)
        relation_labels = torch.full((self.max_length, self.max_length), -1, dtype=torch.long)
        for relation in sample['relations']:
            head_pos = sample['entities'][relation['head']]['span'][0]
            tail_pos = sample['entities'][relation['tail']]['span'][0]
            if head_pos < self.max_length and tail_pos < self.max_length:
                relation_labels[head_pos, tail_pos] = relation['type']
        
        return {
            'input_ids': doc_encoding['input_ids'].squeeze(0),
            'attention_mask': doc_encoding['attention_mask'].squeeze(0),
            'query': query_encoding['input_ids'].squeeze(0),
            'labels': torch.tensor(sample['label'], dtype=torch.long),
            'entity_labels': entity_labels,
            'relation_labels': relation_labels
        }


def example_model_initialization():
    """Example of model initialization with different configurations."""
    print("=== Model Initialization Examples ===")
    
    # Basic initialization
    model_basic = GDMNet(
        bert_model_name='bert-base-uncased',
        hidden_size=768,
        num_entities=10,
        num_relations=20,
        num_classes=5
    )
    print(f"Basic model parameters: {sum(p.numel() for p in model_basic.parameters()):,}")
    
    # Advanced configuration
    model_advanced = GDMNet(
        bert_model_name='bert-base-uncased',
        hidden_size=768,
        num_entities=50,
        num_relations=100,
        num_classes=10,
        gnn_type='gat',
        num_gnn_layers=3,
        num_reasoning_hops=4,
        fusion_method='attention',
        learning_rate=1e-5,
        dropout_rate=0.2
    )
    print(f"Advanced model parameters: {sum(p.numel() for p in model_advanced.parameters()):,}")
    
    return model_basic, model_advanced


def example_forward_pass():
    """Example of forward pass with synthetic data."""
    print("\n=== Forward Pass Example ===")
    
    # Initialize model
    model = GDMNet(
        bert_model_name='bert-base-uncased',
        hidden_size=768,
        num_entities=10,
        num_relations=20,
        num_classes=5
    )
    model.eval()
    
    # Create synthetic input
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    query = torch.randn(batch_size, 768)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query=query,
            return_intermediate=True
        )
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Entity logits shape: {outputs['entity_logits'].shape}")
    print(f"Relation logits shape: {outputs['relation_logits'].shape}")
    print(f"Number of extracted entities: {len(outputs['entities'][0])}")
    print(f"Number of extracted relations: {len(outputs['relations'][0])}")
    print(f"Number of reasoning paths: {outputs['path_info']['num_paths']}")
    
    return outputs


def example_training():
    """Example of model training."""
    print("\n=== Training Example ===")
    
    # Initialize tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = ExampleDataset(tokenizer, num_samples=100)
    val_dataset = ExampleDataset(tokenizer, num_samples=20)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = GDMNet(
        bert_model_name='bert-base-uncased',
        hidden_size=768,
        num_entities=10,
        num_relations=20,
        num_classes=5,
        learning_rate=2e-5
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=2,  # Short training for demonstration
        accelerator='auto',
        devices=1,
        precision=16,
        log_every_n_steps=10,
        enable_checkpointing=False,
        logger=False
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training completed!")
    
    return model, trainer


def example_inference():
    """Example of model inference."""
    print("\n=== Inference Example ===")
    
    # Initialize model and tokenizer
    model = GDMNet(
        bert_model_name='bert-base-uncased',
        hidden_size=768,
        num_entities=10,
        num_relations=20,
        num_classes=5
    )
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Example documents and queries
    documents = [
        "Apple Inc. is a technology company founded by Steve Jobs. Tim Cook is the current CEO.",
        "Microsoft Corporation was founded by Bill Gates. Satya Nadella is the current CEO."
    ]
    
    queries = [
        "Who is the CEO of Apple?",
        "Which company was founded by Bill Gates?"
    ]
    
    # Process each document-query pair
    for doc, query in zip(documents, queries):
        print(f"\nDocument: {doc}")
        print(f"Query: {query}")
        
        # Tokenize
        doc_encoding = tokenizer(
            doc,
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
        
        # Get predictions
        logits = outputs['logits']
        probabilities = F.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        
        print(f"Prediction: {prediction.item()}")
        print(f"Confidence: {probabilities.max().item():.3f}")
        print(f"Entities found: {len(outputs['entities'][0])}")
        print(f"Relations found: {len(outputs['relations'][0])}")


def main():
    """Main function to run all examples."""
    print("GDM-Net Usage Examples")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Model initialization
        model_basic, model_advanced = example_model_initialization()
        
        # Forward pass
        outputs = example_forward_pass()
        
        # Training (commented out to avoid long execution)
        # model, trainer = example_training()
        
        # Inference
        example_inference()
        
        print("\n=== All examples completed successfully! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
