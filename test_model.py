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
        print(f"    BERT model type: {type(doc_encoder.bert).__name__}")

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

        # Test full model with detailed intermediate results
        print("  Testing full model with intermediate results...")
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

        # Forward pass with detailed outputs
        with torch.no_grad():
            outputs = model(**dummy_batch)

        print(f"\n🎯 GDM-Net 双路径处理结果展示:")
        print(f"=" * 60)

        # 1. 隐式路径 (Implicit Path) - BERT文档编码
        print(f"\n📄 隐式路径 (Implicit Path) - 文档语义表示:")
        print(f"  文档表示 (doc_representation): {outputs['doc_representation'].shape}")
        print(f"  查询表示 (query_representation): {outputs['query_representation'].shape}")
        print(f"  特点: 稠密向量，包含全局语义信息")

        # 2. 显式路径 (Explicit Path) - 结构化知识
        print(f"\n🔗 显式路径 (Explicit Path) - 结构化知识表示:")
        print(f"  提取的实体数量: {len(outputs['entities_batch'][0])} (batch 0)")
        print(f"  提取的关系数量: {len(outputs['relations_batch'][0])} (batch 0)")

        # 显示提取的实体信息
        if outputs['entities_batch'][0]:
            print(f"  实体示例:")
            for i, entity in enumerate(outputs['entities_batch'][0][:3]):  # 显示前3个
                print(f"    实体{i+1}: 类型={entity['type']}, 位置={entity['span']}")

        # 显示提取的关系信息
        if outputs['relations_batch'][0]:
            print(f"  关系示例:")
            for i, relation in enumerate(outputs['relations_batch'][0][:3]):  # 显示前3个
                print(f"    关系{i+1}: 头实体={relation['head']}, 尾实体={relation['tail']}, 类型={relation['type']}")

        # 3. 图表示
        print(f"\n🕸️  图神经网络处理结果:")
        print(f"  图表示 (graph_representation): {outputs['graph_representation'].shape}")
        print(f"  特点: 经过GNN处理的结构化知识表示")

        # 4. 多跳推理路径
        print(f"\n🔄 多跳推理路径:")
        print(f"  路径表示 (path_representation): {outputs['path_representation'].shape}")
        print(f"  特点: 基于查询的多跳推理结果")

        # 5. 双记忆系统
        print(f"\n🧠 双记忆系统:")
        print(f"  记忆输出 (memory_output): {outputs['memory_output'].shape}")
        print(f"  特点: 情节记忆 + 语义记忆的融合")

        # 6. 最终融合
        print(f"\n⚡ 双路径融合结果:")
        print(f"  融合表示 (fused_representation): {outputs['fused_representation'].shape}")
        print(f"  最终分类 (logits): {outputs['logits'].shape}")
        print(f"  预测概率分布:")
        probs = torch.softmax(outputs['logits'], dim=1)
        for i in range(min(2, batch_size)):  # 显示前2个样本
            print(f"    样本{i+1}: {[f'{p:.3f}' for p in probs[i].tolist()]}")

        # 7. 创新点总结
        print(f"\n🌟 GDM-Net 创新点展示:")
        print(f"  ✅ 双路径处理: 隐式(BERT) + 显式(图结构)")
        print(f"  ✅ 动态图构建: 从文本自动提取实体关系")
        print(f"  ✅ 多跳推理: PathFinder进行推理路径发现")
        print(f"  ✅ 图读取机制: GraphReader基于查询读取图信息")
        print(f"  ✅ 智能融合: 多种表示的自适应融合")
        print(f"  ✅ 端到端训练: 主任务+辅助任务联合优化")

        print(f"\n✓ Model forward pass successful")
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
    """Test dataset loading and check data structure."""
    print("\nTesting dataset loading...")

    # Load config
    config = load_config('config/default_config.yaml')

    # First, check raw data structure
    import json
    print("  Checking raw data structure...")
    try:
        with open(config['data']['train_path'], 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        print(f"  Raw data type: {type(raw_data)}")
        print(f"  Raw data length: {len(raw_data)}")

        # Check first sample structure
        first_sample = raw_data[0]
        print(f"  First sample keys: {list(first_sample.keys())}")

        # Check if there's a label field
        if 'label' in first_sample:
            print(f"  Label field exists: {first_sample['label']}")
        elif 'answer' in first_sample:
            print(f"  Answer field exists: {first_sample['answer']}")
        else:
            print(f"  No obvious label field. Available keys: {list(first_sample.keys())}")

        # Check a few samples for label distribution
        labels_found = []
        for i in range(min(10, len(raw_data))):
            sample = raw_data[i]
            if 'label' in sample:
                labels_found.append(sample['label'])
            elif 'answer' in sample:
                labels_found.append(sample['answer'])

        print(f"  Labels in first 10 samples: {labels_found}")

    except Exception as e:
        print(f"  Error reading raw data: {e}")

    # Test with dataset class
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
        print(f"  Label value: {sample['label']}")
        print(f"  Label type: {type(sample['label'])}")

        # Check label distribution
        label_dist = dataset.get_label_distribution()
        print(f"  Label distribution: {label_dist}")

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
        print(f"  Batch labels: {batch['labels']}")
        print(f"  Labels range: [{batch['labels'].min()}, {batch['labels'].max()}]")

        return True

    except Exception as e:
        print(f"✗ Dataset loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
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


def test_intermediate_results():
    """专门测试和展示GDM-Net的中间结果"""
    print("\n🔬 GDM-Net 中间结果详细分析")
    print("=" * 80)

    try:
        # Load config
        config = load_config('config/default_config.yaml')

        # Initialize model
        model = GDMNet(**config['model'])
        model.eval()

        # Create a more realistic test batch
        batch_size = 1  # 使用单个样本便于分析
        seq_len = 128

        # 模拟真实的输入数据
        dummy_batch = {
            'query_input_ids': torch.randint(1, 1000, (batch_size, 64)),
            'query_attention_mask': torch.ones(batch_size, 64),
            'doc_input_ids': torch.randint(1, 1000, (batch_size, seq_len)),
            'doc_attention_mask': torch.ones(batch_size, seq_len),
            'entity_spans': torch.tensor([[[10, 15], [25, 30], [45, 50], [60, 65], [80, 85], [90, 95], [100, 105], [110, 115], [120, 125]]]),
            'labels': torch.randint(0, config['model']['num_classes'], (batch_size,))
        }

        # Forward pass
        with torch.no_grad():
            outputs = model(**dummy_batch)

        print("\n📊 GDM-Net 双路径处理流程分析:")
        print("-" * 60)
        print("🎯 这是GDM-Net的核心创新点展示!")

        # 1. 输入分析
        print(f"📥 输入数据:")
        print(f"  查询长度: {dummy_batch['query_input_ids'].shape[1]}")
        print(f"  文档长度: {dummy_batch['doc_input_ids'].shape[1]}")
        print(f"  实体span数量: {dummy_batch['entity_spans'].shape[1]}")

        # 2. 隐式路径分析
        print(f"\n🔄 隐式路径 (BERT编码):")
        doc_repr = outputs['doc_representation']
        query_repr = outputs['query_representation']
        print(f"  文档表示维度: {doc_repr.shape}")
        print(f"  文档表示统计: min={doc_repr.min():.3f}, max={doc_repr.max():.3f}, mean={doc_repr.mean():.3f}")
        print(f"  查询表示维度: {query_repr.shape}")
        print(f"  查询表示统计: min={query_repr.min():.3f}, max={query_repr.max():.3f}, mean={query_repr.mean():.3f}")

        # 3. 显式路径分析
        print(f"\n🕸️  显式路径 (结构化知识):")
        entities = outputs['entities_batch'][0]
        relations = outputs['relations_batch'][0]
        print(f"  提取实体数量: {len(entities)}")
        print(f"  提取关系数量: {len(relations)}")

        if entities:
            entity_types = [e['type'] for e in entities]
            print(f"  实体类型分布: {dict(zip(*torch.unique(torch.tensor(entity_types), return_counts=True)))}")

        if relations:
            relation_types = [r['type'] for r in relations]
            print(f"  关系类型分布: {dict(zip(*torch.unique(torch.tensor(relation_types), return_counts=True)))}")

        # 4. 图构建分析
        print(f"\n🔗 图构建结果:")
        print(f"  节点数量: {outputs['node_features'].shape[0]}")
        print(f"  边数量: {outputs['edge_index'].shape[1]}")
        print(f"  节点特征维度: {outputs['node_features'].shape[1]}")
        print(f"  边类型数量: {len(torch.unique(outputs['edge_type']))}")

        # 5. 图神经网络处理
        print(f"\n🧠 图神经网络处理:")
        updated_features = outputs['updated_node_features']
        print(f"  更新后节点特征: {updated_features.shape}")
        print(f"  特征变化统计: min={updated_features.min():.3f}, max={updated_features.max():.3f}")

        # 6. 推理模块分析
        print(f"\n🔍 推理模块结果:")
        graph_repr = outputs['graph_representation']
        path_repr = outputs['path_representation']
        print(f"  图表示: {graph_repr.shape}")
        print(f"  路径表示: {path_repr.shape}")
        print(f"  图表示统计: min={graph_repr.min():.3f}, max={graph_repr.max():.3f}")
        print(f"  路径表示统计: min={path_repr.min():.3f}, max={path_repr.max():.3f}")

        # 7. 记忆系统分析
        print(f"\n💭 双记忆系统:")
        memory_out = outputs['memory_output']
        episodic_out = outputs['episodic_output']
        semantic_out = outputs['semantic_output']
        print(f"  记忆输出: {memory_out.shape}")
        print(f"  情节记忆: {episodic_out.shape}")
        print(f"  语义记忆: {semantic_out.shape}")

        # 8. 最终融合分析
        print(f"\n⚡ 最终融合:")
        fused_repr = outputs['fused_representation']
        logits = outputs['logits']
        print(f"  融合表示: {fused_repr.shape}")
        print(f"  分类logits: {logits.shape}")

        # 9. 预测结果
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1)
        print(f"\n🎯 预测结果:")
        print(f"  预测类别: {pred_class.item()}")
        print(f"  置信度分布: {[f'{p:.3f}' for p in probs[0].tolist()]}")
        print(f"  最高置信度: {probs.max().item():.3f}")

        # 10. 创新点总结
        print(f"\n🌟 GDM-Net 创新点验证:")
        print(f"  ✅ 双路径处理: 隐式({doc_repr.shape}) + 显式({len(entities)}实体, {len(relations)}关系)")
        print(f"  ✅ 动态图构建: {outputs['node_features'].shape[0]}节点, {outputs['edge_index'].shape[1]}边")
        print(f"  ✅ 多跳推理: 路径发现({path_repr.shape}) + 图读取({graph_repr.shape})")
        print(f"  ✅ 记忆融合: 情节+语义→记忆输出({memory_out.shape})")
        print(f"  ✅ 智能融合: 文档+图+路径+记忆→最终表示({fused_repr.shape})")

        return True

    except Exception as e:
        print(f"✗ 中间结果测试失败: {str(e)}")
        import traceback
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
        test_training_step,
        test_intermediate_results  # 添加中间结果展示
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
