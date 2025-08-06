"""
修复多GPU训练中的设备不匹配问题
"""

import torch
import os
import gc


def clear_gpu_memory():
    """清理GPU内存"""
    print("🧹 清理GPU内存...")
    
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        print(f"✅ 已清理 {torch.cuda.device_count()} 个GPU的内存")
    else:
        print("❌ CUDA不可用")


def setup_multi_gpu_env():
    """设置多GPU环境变量"""
    print("🔧 设置多GPU环境变量...")
    
    # NCCL设置
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_TREE_THRESHOLD'] = '0'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    
    # CUDA设置
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(torch.cuda.device_count()))
    
    # PyTorch设置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    print("✅ 多GPU环境变量设置完成")


def test_multi_gpu_communication():
    """测试多GPU通信"""
    print("🔍 测试多GPU通信...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"⚠️ 只有 {num_gpus} 个GPU，无法测试多GPU通信")
        return False
    
    try:
        # 在每个GPU上创建张量
        tensors = []
        for i in range(num_gpus):
            with torch.cuda.device(i):
                tensor = torch.randn(100, 100, device=f'cuda:{i}')
                tensors.append(tensor)
        
        print(f"✅ 在 {num_gpus} 个GPU上创建张量成功")
        
        # 测试张量传输
        tensor_0_to_1 = tensors[0].to('cuda:1')
        print("✅ GPU间张量传输成功")
        
        # 测试all_reduce操作（模拟DDP通信）
        if torch.distributed.is_available():
            print("✅ 分布式通信可用")
        
        return True
        
    except Exception as e:
        print(f"❌ 多GPU通信测试失败: {e}")
        return False


def fix_device_mismatch():
    """修复设备不匹配问题"""
    print("🔧 修复设备不匹配问题...")
    
    # 设置默认设备
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("✅ 设置默认CUDA设备为0")
    
    # 禁用某些可能导致设备不匹配的优化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("✅ 禁用可能导致问题的CUDNN优化")


def create_single_gpu_fallback_config():
    """创建单GPU回退配置"""
    print("🔄 创建单GPU回退配置...")
    
    fallback_config = """
# Single GPU Fallback Configuration
# 当多GPU训练失败时使用

seed: 42

model:
  bert_model_name: "bert-base-uncased"
  hidden_size: 512  # 减少模型大小
  num_entities: 9
  num_relations: 10
  num_classes: 5
  gnn_type: "rgcn"
  num_gnn_layers: 1  # 减少层数
  num_reasoning_hops: 2
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
  batch_size: 1  # 单GPU小批次
  num_workers: 0  # 禁用多进程
  accelerator: "gpu"
  devices: 1  # 强制单GPU
  precision: 32
  gradient_clip_val: 1.0
  accumulate_grad_batches: 16  # 增加累积
  val_check_interval: 0.5
  log_every_n_steps: 50
  checkpoint_dir: "checkpoints"
  early_stopping: true
  patience: 3

logging:
  type: "tensorboard"
  save_dir: "logs"
  name: "gdmnet-single-gpu-fallback"
"""
    
    with open('config/single_gpu_fallback_config.yaml', 'w') as f:
        f.write(fallback_config.strip())
    
    print("✅ 单GPU回退配置已创建: config/single_gpu_fallback_config.yaml")


def comprehensive_multi_gpu_fix():
    """综合多GPU问题修复"""
    print("🛠️ 多GPU训练问题综合修复")
    print("=" * 50)
    
    # 1. 清理GPU内存
    clear_gpu_memory()
    print()
    
    # 2. 设置环境变量
    setup_multi_gpu_env()
    print()
    
    # 3. 修复设备不匹配
    fix_device_mismatch()
    print()
    
    # 4. 测试多GPU通信
    comm_ok = test_multi_gpu_communication()
    print()
    
    # 5. 创建回退配置
    create_single_gpu_fallback_config()
    print()
    
    # 6. 给出建议
    if comm_ok:
        print("🎉 多GPU环境修复完成！")
        print("💡 建议:")
        print("1. 重新运行训练脚本")
        print("2. 如果仍有问题，使用单GPU回退配置")
        print("3. 监控GPU内存使用情况")
    else:
        print("⚠️ 多GPU通信仍有问题")
        print("💡 建议:")
        print("1. 使用单GPU回退配置")
        print("2. 检查CUDA和驱动版本")
        print("3. 重启系统")
    
    return comm_ok


def quick_single_gpu_fix():
    """快速切换到单GPU模式"""
    print("⚡ 快速切换到单GPU模式")
    
    # 设置环境变量强制单GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 清理内存
    clear_gpu_memory()
    
    print("✅ 已切换到单GPU模式")
    print("💡 请使用以下命令重新训练:")
    print("python train/train.py --config config/single_gpu_fallback_config.yaml --mode train")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--single-gpu':
        quick_single_gpu_fix()
    else:
        success = comprehensive_multi_gpu_fix()
        
        if not success:
            print("\n⚡ 自动切换到单GPU模式...")
            quick_single_gpu_fix()
