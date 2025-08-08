#!/usr/bin/env python3
"""
Quick test script to verify model functionality.
"""

import torch
import json
from src.model import GDMNet
from src.dataloader import HotpotQADataset, GDMNetDataCollator
from src.utils import load_config
from torch.utils.data import DataLoader


def test_model_forward():
    """Test model forward pass."""
    print("Testing model forward pass...")
    
    # Load config
    config = load_config('config/default_config.yaml')
    
    # Initialize model
    model = GDMNet(**config['model'])
    model.eval()
    
    # Create dummy batch
    batch_size = 2
    seq_len = 128
    num_entities = config['model']['num_entities']
    
    dummy_batch = {
        'query_input_ids': torch.randint(0, 1000, (batch_size, 64)),
        'query_attention_mask': torch.ones(batch_size, 64),
        'doc_input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'doc_attention_mask': torch.ones(batch_size, seq_len),
        'entity_spans': torch.randint(0, seq_len, (batch_size, num_entities, 2)),
        'labels': torch.randint(0, config['model']['num_classes'], (batch_size,))
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**dummy_batch)
    
    print(f"✓ Model forward pass successful")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Expected shape: ({batch_size}, {config['model']['num_classes']})")
    
    return True


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
        
        dummy_batch = {
            'query_input_ids': torch.randint(0, 1000, (batch_size, 64)),
            'query_attention_mask': torch.ones(batch_size, 64),
            'doc_input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'doc_attention_mask': torch.ones(batch_size, seq_len),
            'entity_spans': torch.randint(0, seq_len, (batch_size, num_entities, 2)),
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
        print(f"✗ Training step failed: {str(e)}")
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
