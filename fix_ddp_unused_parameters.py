"""
修复DDP未使用参数问题
解决多GPU训练中的参数同步错误
"""

import torch
import os
import sys


def analyze_unused_parameters():
    """分析未使用的参数"""
    print("🔍 分析DDP未使用参数问题...")
    
    # 从错误信息中提取的未使用参数索引
    unused_params_rank0 = [207, 208, 209, 210, 217, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 270, 271]
    unused_params_rank1 = [207, 208, 209, 210, 217, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 270, 271]
    
    print(f"📊 Rank 0 未使用参数: {len(unused_params_rank0)} 个")
    print(f"📊 Rank 1 未使用参数: {len(unused_params_rank1)} 个")
    print(f"🎯 参数索引范围: {min(unused_params_rank0)} - {max(unused_params_rank0)}")
    
    # 这些参数可能来自：
    print("\n💡 可能的未使用参数来源:")
    print("  - 图记忆模块的某些层")
    print("  - 推理模块的某些组件")
    print("  - 条件性使用的参数（当图为空时）")


def setup_ddp_debug_env():
    """设置DDP调试环境"""
    print("🔧 设置DDP调试环境...")
    
    # DDP调试环境变量
    debug_env = {
        'TORCH_DISTRIBUTED_DEBUG': 'DETAIL',
        'NCCL_DEBUG': 'INFO',
        'CUDA_LAUNCH_BLOCKING': '1',  # 同步执行以便调试
        'TORCH_SHOW_CPP_STACKTRACES': '1',
        'TORCH_USE_CUDA_DSA': '1'
    }
    
    for key, value in debug_env.items():
        os.environ[key] = value
        print(f"  {key} = {value}")
    
    print("✅ DDP调试环境设置完成")


def create_ddp_safe_config():
    """创建DDP安全配置"""
    print("📝 创建DDP安全配置...")
    
    safe_config = """
# DDP Safe Configuration
# 专门解决未使用参数问题的配置

seed: 42

model:
  bert_model_name: "bert-base-uncased"
  hidden_size: 768
  num_entities: 9
  num_relations: 10
  num_classes: 5
  gnn_type: "rgcn"
  num_gnn_layers: 2
  num_reasoning_hops: 3
  fusion_method: "gate"
  learning_rate: 2e-5
  dropout_rate: 0.1

data:
  train_path: "data/hotpotqa_official_train.json"
  val_path: "data/hotpotqa_official_val.json"
  test_path: "data/hotpotqa_official_val.json"
  max_length: 384  # 适中的序列长度
  max_query_length: 64

training:
  max_epochs: 5
  batch_size: 2  # 较小的批次大小确保稳定性
  num_workers: 0  # 禁用多进程避免复杂性
  accelerator: "gpu"
  devices: 2  # 使用2个GPU
  strategy: "ddp_find_unused_parameters_true"  # 启用未使用参数检测
  precision: 32
  gradient_clip_val: 1.0
  accumulate_grad_batches: 2
  val_check_interval: 0.5
  log_every_n_steps: 10
  checkpoint_dir: "checkpoints"
  early_stopping: true
  patience: 3
  sync_batchnorm: true  # 同步批归一化

logging:
  type: "tensorboard"
  save_dir: "logs"
  name: "gdmnet-ddp-safe"
"""
    
    with open('config/ddp_safe_config.yaml', 'w') as f:
        f.write(safe_config.strip())
    
    print("✅ DDP安全配置已创建: config/ddp_safe_config.yaml")


def test_model_forward():
    """测试模型前向传播"""
    print("🧪 测试模型前向传播...")
    
    try:
        from gdmnet import GDMNet
        
        # 创建模型
        model = GDMNet(
            bert_model_name='bert-base-uncased',
            hidden_size=768,
            num_entities=9,
            num_relations=10,
            num_classes=5
        )
        
        model.eval()
        
        # 创建测试输入
        batch_size = 2
        seq_len = 128
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        query = torch.randint(0, 1000, (batch_size, 32))
        entity_labels = torch.randint(0, 9, (batch_size, seq_len))
        relation_labels = torch.randint(0, 10, (batch_size, seq_len))
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                query=query,
                entity_labels=entity_labels,
                relation_labels=relation_labels
            )
        
        print("✅ 模型前向传播测试成功")
        print(f"  输出形状: {outputs['logits'].shape}")
        
        # 检查所有参数是否有梯度
        model.train()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query=query,
            entity_labels=entity_labels,
            relation_labels=relation_labels
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # 统计有梯度的参数
        params_with_grad = 0
        params_without_grad = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
            else:
                params_without_grad += 1
                print(f"  无梯度参数: {name}")
        
        print(f"📊 参数梯度统计:")
        print(f"  有梯度: {params_with_grad}")
        print(f"  无梯度: {params_without_grad}")
        
        if params_without_grad == 0:
            print("✅ 所有参数都有梯度")
            return True
        else:
            print("⚠️ 存在无梯度参数")
            return False
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False


def comprehensive_ddp_fix():
    """综合DDP问题修复"""
    print("🛠️ DDP未使用参数问题综合修复")
    print("=" * 50)
    
    # 1. 分析问题
    analyze_unused_parameters()
    print()
    
    # 2. 设置调试环境
    setup_ddp_debug_env()
    print()
    
    # 3. 创建安全配置
    create_ddp_safe_config()
    print()
    
    # 4. 测试模型
    model_ok = test_model_forward()
    print()
    
    # 5. 给出建议
    if model_ok:
        print("🎉 模型测试通过！")
        print("💡 建议:")
        print("1. 使用DDP安全配置重新训练:")
        print("   python train/train.py --config config/ddp_safe_config.yaml --mode train")
        print("2. 监控DDP调试输出")
        print("3. 如果仍有问题，切换到单GPU模式")
    else:
        print("⚠️ 模型仍有问题")
        print("💡 建议:")
        print("1. 检查模型代码中的条件分支")
        print("2. 确保所有参数都参与前向传播")
        print("3. 使用单GPU模式作为备选")
    
    return model_ok


def quick_single_gpu_switch():
    """快速切换到单GPU模式"""
    print("⚡ 快速切换到单GPU模式")
    
    # 强制单GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print("✅ 已切换到单GPU模式")
    print("💡 使用命令:")
    print("python train/train.py --config config/single_gpu_fallback_config.yaml --mode train")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--single-gpu':
        quick_single_gpu_switch()
    else:
        success = comprehensive_ddp_fix()
        
        if not success:
            print("\n⚡ 建议切换到单GPU模式...")
            quick_single_gpu_switch()
