#!/usr/bin/env python3
"""
Quick test script to verify model functionality.
"""

import torch
import json
import os
from src.model import GDMNet
from src.dataloader import HotpotQADataset, GDMNetDataCollator
from src.utils import load_config
from torch.utils.data import DataLoader

# Set environment variables for Chinese mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'


def test_model_forward():
    """Test model forward pass with detailed debugging."""
    print("Testing model forward pass...")

    try:
        # Load config
        config = load_config('config/default_config.yaml')
        print(f"  Config hidden_size: {config['model']['hidden_size']}")

        # Initialize model components step by step
        print("  Initializing DocumentEncoder...")
        from src.model.bert_encoder import DocumentEncoder
        doc_encoder = DocumentEncoder(
            model_name=config['model']['bert_model_name'],
            hidden_size=config['model']['hidden_size']
        )
        print(f"    BERT config hidden_size: {doc_encoder.config.hidden_size}")
        print(f"    Target hidden_size: {doc_encoder.hidden_size}")
        print(f"    Need projection: {doc_encoder.need_projection}")

        # Test document encoding
        batch_size = 2
        seq_len = 128
        dummy_input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        dummy_attention_mask = torch.ones(batch_size, seq_len)

        print("  Testing document encoding...")
        doc_pooled, doc_sequence = doc_encoder.encode_document(dummy_input_ids, dummy_attention_mask)
        print(f"    doc_pooled shape: {doc_pooled.shape}")
        print(f"    doc_sequence shape: {doc_sequence.shape}")

        # Test structure extraction
        print("  Testing structure extraction...")
        from src.model.bert_encoder import StructureExtractor
        struct_extractor = StructureExtractor(
            hidden_size=config['model']['hidden_size'],
            num_entity_types=config['model']['num_entities'],
            num_relation_types=config['model']['num_relations']
        )

        num_entities = config['model']['num_entities']
        entity_spans = torch.zeros(batch_size, num_entities, 2, dtype=torch.long)
        for b in range(batch_size):
            for e in range(num_entities):
                start = torch.randint(0, seq_len - 10, (1,)).item()
                end = start + torch.randint(1, 10, (1,)).item()
                entity_spans[b, e] = torch.tensor([start, min(end, seq_len - 1)])

        entity_logits, relation_logits, entities_batch, relations_batch = struct_extractor(
            doc_sequence, dummy_attention_mask, entity_spans
        )
        print(f"    entity_logits shape: {entity_logits.shape}")
        print(f"    entities_batch length: {len(entities_batch)}")
        for i, entities in enumerate(entities_batch):
            print(f"      Batch {i}: {len(entities)} entities")
            for j, entity in enumerate(entities[:3]):  # Show first 3 entities
                print(f"        Entity {j}: type={entity['type']}, span={entity['span']}, repr_shape={entity['representation'].shape}")

        # Test graph writer
        print("  Testing graph writer...")
        from src.model.graph_memory import GraphWriter
        graph_writer = GraphWriter(
            hidden_size=config['model']['hidden_size'],
            num_entity_types=config['model']['num_entities'],
            num_relation_types=config['model']['num_relations']
        )

        node_features, edge_index, edge_type, batch_indices = graph_writer(
            entities_batch, relations_batch, doc_sequence
        )
        print(f"    node_features shape: {node_features.shape}")
        print(f"    edge_index shape: {edge_index.shape}")
        print(f"    edge_type shape: {edge_type.shape}")
        print(f"    batch_indices shape: {batch_indices.shape}")

        # Test full model
        print("  Testing full model...")
        model = GDMNet(**config['model'])
        model.eval()

        dummy_batch = {
            'query_input_ids': torch.randint(1, 1000, (batch_size, 64)),
            'query_attention_mask': torch.ones(batch_size, 64),
            'doc_input_ids': dummy_input_ids,
            'doc_attention_mask': dummy_attention_mask,
            'entity_spans': entity_spans,
            'labels': torch.randint(0, config['model']['num_classes'], (batch_size,))
        }

        # Forward pass
        with torch.no_grad():
            outputs = model(**dummy_batch)

        print(f"✓ Model forward pass successful")
        print(f"  Output logits shape: {outputs['logits'].shape}")
        print(f"  Expected shape: ({batch_size}, {config['model']['num_classes']})")

        return True

    except Exception as e:
        import traceback
        print(f"✗ Model forward pass failed: {str(e)}")
        print("  Full traceback:")
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    
    # Load config
    config = load_config('config/default_config.yaml')
    
    # Test with a small subset
    try:
        dataset = HotpotQADataset(
            data_path=config['data']['train_path'],
            tokenizer_name=config['model']['bert_model_name'],
            max_length=config['data']['max_length'],
            max_query_length=config['data']['max_query_length'],
            num_entities=config['model']['num_entities']
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Dataset size: {len(dataset)}")
        
        # Test single sample
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        
        # Test dataloader
        collator = GDMNetDataCollator()
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=collator
        )
        
        batch = next(iter(dataloader))
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Batch size: {batch['labels'].shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {str(e)}")
        return False


def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    try:
        from src.train import GDMNetTrainer
        
        # Load config
        config = load_config('config/default_config.yaml')
        
        # Initialize trainer
        trainer = GDMNetTrainer(config)
        trainer.eval()
        
        # Create dummy batch (same as in test_model_forward)
        batch_size = 2
        seq_len = 128
        num_entities = config['model']['num_entities']
        
        # Create valid entity spans
        entity_spans = torch.zeros(batch_size, num_entities, 2, dtype=torch.long)
        for b in range(batch_size):
            for e in range(num_entities):
                start = torch.randint(0, seq_len - 10, (1,)).item()
                end = start + torch.randint(1, 10, (1,)).item()
                entity_spans[b, e] = torch.tensor([start, min(end, seq_len - 1)])

        dummy_batch = {
            'query_input_ids': torch.randint(1, 1000, (batch_size, 64)),
            'query_attention_mask': torch.ones(batch_size, 64),
            'doc_input_ids': torch.randint(1, 1000, (batch_size, seq_len)),
            'doc_attention_mask': torch.ones(batch_size, seq_len),
            'entity_spans': entity_spans,
            'labels': torch.randint(0, config['model']['num_classes'], (batch_size,))
        }
        
        # Test training step
        with torch.no_grad():
            outputs = trainer.forward(dummy_batch)
            loss_dict = trainer.model.compute_loss(outputs, dummy_batch['labels'])
            loss = loss_dict['total_loss']

        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")

        return True

    except Exception as e:
        import traceback
        print(f"✗ Training step failed: {str(e)}")
        print("  Full traceback:")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("GDM-Net Model Testing")
    print("=" * 50)
    
    tests = [
        test_model_forward,
        test_dataset_loading,
        test_training_step
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Model is ready for training.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    main()
