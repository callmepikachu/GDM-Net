"""
快速修复PyTorch Lightning策略名称问题
"""

import os
import torch


def show_available_strategies():
    """显示可用的策略"""
    print("📋 PyTorch Lightning可用策略:")
    
    available_strategies = [
        'ddp', 'ddp_find_unused_parameters_false', 'ddp_fork', 
        'ddp_spawn', 'ddp_spawn_find_unused_parameters_false',
        'ddp_sharded', 'ddp_fully_sharded', 'deepspeed',
        'single_device', 'dp'
    ]
    
    for strategy in available_strategies:
        print(f"  ✅ {strategy}")
    
    print(f"\n🎯 推荐策略:")
    print(f"  - 多GPU: ddp")
    print(f"  - 单GPU: single_device")
    print(f"  - 调试: ddp_spawn")


def create_working_multi_gpu_config():
    """创建可工作的多GPU配置"""
    print("📝 创建可工作的多GPU配置...")
    
    working_config = """
# Working Multi-GPU Configuration
# 使用正确的PyTorch Lightning策略

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
  max_length: 384
  max_query_length: 64

training:
  max_epochs: 5
  batch_size: 4  # 每个GPU的批次大小
  num_workers: 2
  accelerator: "gpu"
  devices: 2  # 使用2个GPU
  strategy: "ddp"  # 使用标准DDP策略
  precision: 32
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  val_check_interval: 0.5
  log_every_n_steps: 50
  checkpoint_dir: "checkpoints"
  early_stopping: true
  patience: 3

logging:
  type: "tensorboard"
  save_dir: "logs"
  name: "gdmnet-working-multi-gpu"
"""
    
    with open('config/working_multi_gpu_config.yaml', 'w') as f:
        f.write(working_config.strip())
    
    print("✅ 可工作的多GPU配置已创建: config/working_multi_gpu_config.yaml")


def create_conservative_single_gpu_config():
    """创建保守的单GPU配置"""
    print("📝 创建保守的单GPU配置...")
    
    conservative_config = """
# Conservative Single GPU Configuration
# 最稳定的训练配置

seed: 42

model:
  bert_model_name: "bert-base-uncased"
  hidden_size: 512  # 减少模型大小
  num_entities: 9
  num_relations: 10
  num_classes: 5
  gnn_type: "rgcn"
  num_gnn_layers: 1  # 减少层数
  num_reasoning_hops: 2  # 减少推理跳数
  fusion_method: "gate"
  learning_rate: 3e-5
  dropout_rate: 0.1

data:
  train_path: "data/hotpotqa_official_train.json"
  val_path: "data/hotpotqa_official_val.json"
  test_path: "data/hotpotqa_official_val.json"
  max_length: 256  # 减少序列长度
  max_query_length: 32

training:
  max_epochs: 5
  batch_size: 1  # 最小批次大小
  num_workers: 0  # 禁用多进程
  accelerator: "gpu"
  devices: 1  # 单GPU
  strategy: "single_device"  # 单设备策略
  precision: 32
  gradient_clip_val: 1.0
  accumulate_grad_batches: 8  # 通过累积增加有效批次大小
  val_check_interval: 0.5
  log_every_n_steps: 50
  checkpoint_dir: "checkpoints"
  early_stopping: true
  patience: 3

logging:
  type: "tensorboard"
  save_dir: "logs"
  name: "gdmnet-conservative-single-gpu"
"""
    
    with open('config/conservative_single_gpu_config.yaml', 'w') as f:
        f.write(conservative_config.strip())
    
    print("✅ 保守的单GPU配置已创建: config/conservative_single_gpu_config.yaml")


def test_gpu_setup():
    """测试GPU设置"""
    print("🔍 测试GPU设置...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return 0
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ 检测到 {num_gpus} 个GPU")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return num_gpus


def main():
    """主函数"""
    print("🛠️ PyTorch Lightning策略修复")
    print("=" * 50)
    
    # 1. 显示可用策略
    show_available_strategies()
    print()
    
    # 2. 测试GPU设置
    num_gpus = test_gpu_setup()
    print()
    
    # 3. 创建配置文件
    if num_gpus > 1:
        create_working_multi_gpu_config()
        print()
        print("🚀 多GPU训练命令:")
        print("python train/train.py --config config/working_multi_gpu_config.yaml --mode train")
    
    create_conservative_single_gpu_config()
    print()
    print("🔧 单GPU训练命令:")
    print("python train/train.py --config config/conservative_single_gpu_config.yaml --mode train")
    
    # 4. 给出建议
    print("\n💡 建议:")
    if num_gpus > 1:
        print("1. 先尝试多GPU配置")
        print("2. 如果有问题，使用单GPU配置")
    else:
        print("1. 使用单GPU配置")
    print("3. 监控GPU内存使用情况")
    print("4. 根据需要调整批次大小")


if __name__ == "__main__":
    main()
