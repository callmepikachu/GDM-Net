"""
简单的 GDM-Net 测试脚本
直接在当前环境中运行测试
"""

def test_pytorch():
    """测试 PyTorch"""
    print("🧪 测试 PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"❌ PyTorch 导入失败: {e}")
        return False


def test_dependencies():
    """测试依赖包"""
    print("\n🧪 测试依赖包...")
    
    dependencies = [
        ('transformers', 'Transformers'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML')
    ]
    
    all_ok = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} 未安装")
            all_ok = False
    
    return all_ok


def test_gdmnet_import():
    """测试 GDM-Net 导入"""
    print("\n🧪 测试 GDM-Net 导入...")
    try:
        from gdmnet import GDMNet
        print("✅ GDM-Net 导入成功")
        return True
    except ImportError as e:
        print(f"❌ GDM-Net 导入失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n🧪 测试模型创建...")
    try:
        from gdmnet import GDMNet
        
        model = GDMNet(
            bert_model_name='bert-base-uncased',
            hidden_size=768,
            num_entities=5,
            num_relations=10,
            num_classes=3
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型创建成功")
        print(f"✅ 模型参数数量: {param_count:,}")
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n🧪 测试前向传播...")
    try:
        import torch
        from gdmnet import GDMNet
        
        model = GDMNet(
            bert_model_name='bert-base-uncased',
            hidden_size=768,
            num_entities=5,
            num_relations=10,
            num_classes=3
        )
        model.eval()
        
        # 创建测试输入
        batch_size = 2
        seq_len = 64
        
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        query = torch.randn(batch_size, 768)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                query=query,
                return_intermediate=True
            )
        
        print(f"✅ 前向传播成功")
        print(f"✅ 输出 logits 形状: {outputs['logits'].shape}")
        print(f"✅ 实体 logits 形状: {outputs['entity_logits'].shape}")
        print(f"✅ 关系 logits 形状: {outputs['relation_logits'].shape}")
        print(f"✅ 提取的实体数量: {len(outputs['entities'][0])}")
        print(f"✅ 提取的关系数量: {len(outputs['relations'][0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """测试配置加载"""
    print("\n🧪 测试配置加载...")
    try:
        import yaml
        import os
        
        config_path = "config/model_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ 配置文件加载成功")
            print(f"✅ 模型类型: {config['model']['bert_model_name']}")
            print(f"✅ 隐藏层大小: {config['model']['hidden_size']}")
            return True
        else:
            print(f"❌ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧠 GDM-Net 安装测试")
    print("=" * 50)
    
    tests = [
        ("PyTorch", test_pytorch),
        ("依赖包", test_dependencies),
        ("GDM-Net 导入", test_gdmnet_import),
        ("模型创建", test_model_creation),
        ("前向传播", test_forward_pass),
        ("配置加载", test_config_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！GDM-Net 已准备就绪。")
        print("\n后续操作:")
        print("1. 激活环境: conda activate gdmnet")
        print("2. 创建数据: python train/dataset.py")
        print("3. 开始训练: python train/train.py --config config/model_config.yaml --mode train")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        
        # 提供修复建议
        if passed >= 3:  # 如果大部分测试通过
            print("\n💡 修复建议:")
            print("大部分功能正常，可以尝试继续使用。")
            print("如果训练时遇到问题，请检查具体的错误信息。")
        
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
