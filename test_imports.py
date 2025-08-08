#!/usr/bin/env python3
"""
测试新项目结构中的导入是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """测试所有关键组件的导入"""
    print("🧪 测试项目导入...")
    
    try:
        # 测试模型导入
        print("📦 测试模型组件导入...")
        from src.model.model import GDMNet
        print("✅ GDMNet 导入成功")
        
        from src.model.encoder import DocumentEncoder
        print("✅ DocumentEncoder 导入成功")
        
        from src.model.extractor import StructureExtractor
        print("✅ StructureExtractor 导入成功")
        
        from src.model.graph_memory import GraphMemory, GraphWriter
        print("✅ GraphMemory, GraphWriter 导入成功")
        
        from src.model.reasoning import PathFinder, GraphReader, ReasoningFusion
        print("✅ PathFinder, GraphReader, ReasoningFusion 导入成功")
        
        # 测试数据加载器导入
        print("\n📊 测试数据加载器导入...")
        from src.dataloader.dataset import GDMNetDataset, create_data_loaders, create_synthetic_dataset
        print("✅ Dataset 组件导入成功")
        
        print("\n🎉 所有导入测试通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")
    
    try:
        from src.model.model import GDMNet
        
        # 创建模型实例
        model = GDMNet(
            bert_model_name='bert-base-uncased',
            hidden_size=256,
            num_entities=9,
            num_relations=10,
            num_classes=5,
            num_gnn_layers=1,
            num_reasoning_hops=1
        )
        
        print("✅ 模型创建成功")
        print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GDM-Net 项目结构测试")
    print("=" * 40)
    
    # 测试导入
    import_success = test_imports()
    
    if import_success:
        # 测试模型创建
        model_success = test_model_creation()
        
        if model_success:
            print("\n🎉 所有测试通过！项目结构正常工作。")
        else:
            print("\n⚠️ 模型创建测试失败，但导入正常。")
    else:
        print("\n❌ 导入测试失败，需要修复导入路径。")
