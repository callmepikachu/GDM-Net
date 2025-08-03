"""
最终设备问题修复脚本
解决所有CUDA/CPU设备不匹配问题
"""

import torch
import os
import gc


def clear_gpu_memory():
    """清理GPU内存"""
    print("🧹 清理GPU内存...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU内存已清理")


def fix_rgcn_device_issues():
    """修复RGCN设备问题"""
    print("🔧 修复RGCN设备问题...")
    
    graph_memory_path = "gdmnet/graph_memory.py"
    
    try:
        with open(graph_memory_path, "r") as f:
            content = f.read()
        
        # 检查是否已经修复
        if "gnn_layer.lin_rel.weight.to(device)" in content:
            print("✅ RGCN设备问题已经修复")
            return True
        
        # 修复RGCN设备问题
        old_pattern = """            # Apply GNN layer
            if self.gnn_type == 'rgcn':
                h = gnn_layer(h, edge_index, edge_type)
            else:  # GAT
                h = gnn_layer(h, edge_index)"""
        
        new_pattern = """            # Apply GNN layer with device synchronization
            if self.gnn_type == 'rgcn':
                # 确保RGCN层的所有参数都在正确设备上
                if hasattr(gnn_layer, 'lin_rel') and hasattr(gnn_layer.lin_rel, 'weight'):
                    gnn_layer.lin_rel.weight = gnn_layer.lin_rel.weight.to(device)
                if hasattr(gnn_layer, 'lin_root') and hasattr(gnn_layer.lin_root, 'weight'):
                    gnn_layer.lin_root.weight = gnn_layer.lin_root.weight.to(device)
                if hasattr(gnn_layer, 'bias') and gnn_layer.bias is not None:
                    gnn_layer.bias = gnn_layer.bias.to(device)
                
                h = gnn_layer(h, edge_index, edge_type)
            else:  # GAT
                h = gnn_layer(h, edge_index)"""
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            
            with open(graph_memory_path, "w") as f:
                f.write(content)
            
            print("✅ RGCN设备问题已修复")
            return True
        else:
            print("⚠️ 未找到需要修复的RGCN代码模式")
            return False
            
    except Exception as e:
        print(f"❌ 修复RGCN设备问题失败: {e}")
        return False


def fix_all_device_issues():
    """修复所有设备问题"""
    print("🛠️ 修复所有设备问题...")
    
    # 1. 清理内存
    clear_gpu_memory()
    
    # 2. 修复RGCN问题
    rgcn_ok = fix_rgcn_device_issues()
    
    # 3. 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("✅ 环境变量设置完成")
    
    return rgcn_ok


def create_minimal_config():
    """创建最小化配置以避免设备问题"""
    print("📝 创建最小化配置...")
    
    minimal_config = """
# Minimal Configuration to Avoid Device Issues
seed: 42

model:
  bert_model_name: "bert-base-uncased"
  hidden_size: 384  # 减少模型大小
  num_entities: 9
  num_relations: 10
  num_classes: 5
  gnn_type: "rgcn"
  num_gnn_layers: 1  # 只用1层避免复杂性
  num_reasoning_hops: 1  # 减少推理跳数
  fusion_method: "gate"
  learning_rate: 3e-5
  dropout_rate: 0.1

data:
  train_path: "data/hotpotqa_official_train.json"
  val_path: "data/hotpotqa_official_val.json"
  test_path: "data/hotpotqa_official_val.json"
  max_length: 128  # 大幅减少序列长度
  max_query_length: 32

training:
  max_epochs: 3  # 减少训练轮数
  batch_size: 1
  num_workers: 0
  accelerator: "gpu"
  devices: 1
  precision: 32
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4  # 减少累积
  val_check_interval: 1.0  # 每个epoch验证一次
  log_every_n_steps: 100
  checkpoint_dir: "checkpoints"
  early_stopping: true
  patience: 2

logging:
  type: "tensorboard"
  save_dir: "logs"
  name: "gdmnet-minimal"
"""
    
    with open('config/minimal_config.yaml', 'w') as f:
        f.write(minimal_config.strip())
    
    print("✅ 最小化配置已创建: config/minimal_config.yaml")


def test_model_creation():
    """测试模型创建"""
    print("🧪 测试模型创建...")
    
    try:
        from gdmnet import GDMNet
        
        # 创建最小模型
        model = GDMNet(
            bert_model_name='bert-base-uncased',
            hidden_size=384,
            num_entities=9,
            num_relations=10,
            num_classes=5,
            num_gnn_layers=1,
            num_reasoning_hops=1
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ 模型已移动到GPU")
        
        # 测试前向传播
        batch_size = 1
        seq_len = 64
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        query = torch.randint(0, 1000, (batch_size, 16))
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            query = query.cuda()
        
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                query=query
            )
        
        print("✅ 模型前向传播测试成功")
        print(f"  输出形状: {outputs['logits'].shape}")
        print(f"  输出设备: {outputs['logits'].device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🛠️ 最终设备问题修复")
    print("=" * 40)
    
    # 1. 修复所有设备问题
    device_ok = fix_all_device_issues()
    print()
    
    # 2. 创建最小化配置
    create_minimal_config()
    print()
    
    # 3. 测试模型
    model_ok = test_model_creation()
    print()
    
    # 4. 给出建议
    if device_ok and model_ok:
        print("🎉 所有设备问题已修复！")
        print("💡 建议使用最小化配置重新训练:")
        print("python train/train.py --config config/minimal_config.yaml --mode train")
    else:
        print("⚠️ 仍有问题需要解决")
        print("💡 建议:")
        print("1. 重启Colab运行时")
        print("2. 重新运行所有单元格")
        print("3. 使用最小化配置")


if __name__ == "__main__":
    main()
