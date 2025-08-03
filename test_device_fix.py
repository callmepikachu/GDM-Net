"""
测试设备修复是否成功
"""

import torch
from gdmnet import GDMNet


def test_device_fix():
    """测试设备修复"""
    print("🧪 测试设备修复...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU测试")
        device = 'cpu'
    else:
        print("✅ CUDA可用，使用GPU测试")
        device = 'cuda'
    
    try:
        # 创建模型
        model = GDMNet(
            bert_model_name='bert-base-uncased',
            hidden_size=768,
            num_entities=9,  # 根据训练日志调整
            num_relations=10,
            num_classes=5
        )
        
        # 移动到设备
        model = model.to(device)
        model.eval()
        
        print(f"✅ 模型创建成功，设备: {device}")
        
        # 创建测试数据
        batch_size = 2
        seq_len = 128
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        query = torch.randint(0, 1000, (batch_size, 32), device=device)
        
        # 创建测试实体和关系标签
        entity_labels = torch.randint(0, 9, (batch_size, seq_len), device=device)
        relation_labels = torch.randint(0, 10, (batch_size, seq_len), device=device)
        
        print("✅ 测试数据创建成功")
        
        # 前向传播测试
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                query=query,
                entity_labels=entity_labels,
                relation_labels=relation_labels
            )
        
        print("✅ 前向传播成功")
        print(f"✅ 输出形状: {outputs['logits'].shape}")
        print(f"✅ 输出设备: {outputs['logits'].device}")

        # 测试验证步骤（这是出错的地方）
        print("🧪 测试验证步骤...")
        model.train()  # 切换到训练模式以测试validation_step

        # 创建标签
        labels = torch.randint(0, 5, (batch_size,), device=device)

        # 模拟验证步骤
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'query': query,
            'entity_labels': entity_labels,
            'relation_labels': relation_labels,
            'labels': labels
        }

        loss = model.validation_step(batch, 0)
        print(f"✅ 验证步骤成功，损失: {loss:.4f}")

        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_device_fix()
    if success:
        print("\n🎉 设备修复成功！可以继续训练了。")
    else:
        print("\n❌ 设备修复失败，需要进一步调试。")
