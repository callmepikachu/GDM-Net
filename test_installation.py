"""
Test script to verify GDM-Net installation and basic functionality.
"""

import torch
import sys
import os

def test_imports():
    """Test if all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from gdmnet import GDMNet
        print("✓ GDMNet imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GDMNet: {e}")
        return False
    
    try:
        from gdmnet.encoder import DocumentEncoder
        from gdmnet.extractor import StructureExtractor
        from gdmnet.graph_memory import GraphMemory, GraphWriter
        from gdmnet.reasoning import PathFinder, GraphReader, ReasoningFusion
        print("✓ All submodules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import submodules: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if the model can be created successfully."""
    print("\nTesting model creation...")
    
    try:
        from gdmnet import GDMNet
        
        model = GDMNet(
            bert_model_name='bert-base-uncased',
            hidden_size=768,
            num_entities=10,
            num_relations=20,
            num_classes=5
        )
        
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True, model
        
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False, None

def test_forward_pass(model):
    """Test if the model can perform a forward pass."""
    print("\nTesting forward pass...")
    
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 128
        
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        query = torch.randn(batch_size, 768)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                query=query,
                return_intermediate=True
            )
        
        print("✓ Forward pass successful")
        print(f"  - Output logits shape: {outputs['logits'].shape}")
        print(f"  - Entity logits shape: {outputs['entity_logits'].shape}")
        print(f"  - Relation logits shape: {outputs['relation_logits'].shape}")
        print(f"  - Entities extracted: {len(outputs['entities'][0])}")
        print(f"  - Relations extracted: {len(outputs['relations'][0])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML')
    ]
    
    all_available = True
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name} available")
        except ImportError:
            print(f"✗ {display_name} not available")
            all_available = False
    
    return all_available

def test_config_loading():
    """Test if configuration can be loaded."""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        
        config_path = "config/model_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration loaded successfully")
            print(f"  - Model type: {config['model']['bert_model_name']}")
            print(f"  - Hidden size: {config['model']['hidden_size']}")
            print(f"  - Batch size: {config['training']['batch_size']}")
            return True
        else:
            print(f"✗ Configuration file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

def main():
    """Run all tests."""
    print("GDM-Net Installation Test")
    print("=" * 50)
    
    # Test results
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test dependencies
    results.append(test_dependencies())
    
    # Test model creation
    model_success, model = test_model_creation()
    results.append(model_success)
    
    # Test forward pass (only if model creation succeeded)
    if model_success and model is not None:
        results.append(test_forward_pass(model))
    else:
        results.append(False)
    
    # Test configuration loading
    results.append(test_config_loading())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    test_names = [
        "Imports",
        "Dependencies", 
        "Model Creation",
        "Forward Pass",
        "Configuration Loading"
    ]
    
    passed = sum(results)
    total = len(results)
    
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GDM-Net is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
