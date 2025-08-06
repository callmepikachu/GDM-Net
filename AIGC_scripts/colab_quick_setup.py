"""
Google Colab 快速设置脚本
一键完成环境配置和训练准备
"""

import os
import json
import subprocess
import sys


def install_dependencies():
    """安装依赖包"""
    print("📦 安装依赖包...")
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "torch-geometric",
        "transformers>=4.20.0",
        "pytorch-lightning>=1.7.0",
        "datasets>=2.0.0",
        "PyYAML>=6.0",
        "tensorboard>=2.8.0",
        "wandb>=0.12.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + package.split(), 
                         check=True, capture_output=True)
            print(f"✅ {package.split()[0]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package.split()[0]}: {e}")
    
    print("✅ 依赖安装完成")


def check_gpu():
    """检查GPU可用性"""
    print("🔍 检查GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️ No GPU available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def create_directories():
    """创建项目目录"""
    print("📁 创建项目目录...")
    
    directories = [
        'gdmnet', 'train', 'config', 'data', 
        'checkpoints', 'logs', 'examples'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ {dir_name}/")


def create_colab_config():
    """创建Colab优化配置"""
    print("⚙️ 创建配置文件...")
    
    config = {
        'seed': 42,
        'model': {
            'bert_model_name': 'bert-base-uncased',
            'hidden_size': 768,
            'num_entities': 8,
            'num_relations': 4,
            'num_classes': 5,
            'gnn_type': 'rgcn',
            'num_gnn_layers': 2,
            'num_reasoning_hops': 3,
            'fusion_method': 'gate',
            'learning_rate': 2e-5,
            'dropout_rate': 0.1
        },
        'data': {
            'train_path': 'data/hotpotqa_train.json',
            'val_path': 'data/hotpotqa_val.json',
            'test_path': 'data/hotpotqa_val.json',
            'max_length': 512,
            'max_query_length': 64
        },
        'training': {
            'max_epochs': 10,
            'batch_size': 8,
            'num_workers': 2,
            'accelerator': 'gpu',
            'devices': 1,
            'precision': 16,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'val_check_interval': 0.5,
            'log_every_n_steps': 50,
            'checkpoint_dir': 'checkpoints',
            'early_stopping': True,
            'patience': 3
        },
        'logging': {
            'type': 'tensorboard',
            'save_dir': 'logs',
            'name': 'gdmnet-colab'
        }
    }
    
    import yaml
    with open('config/colab_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ 配置文件: config/colab_config.yaml")


def create_sample_data():
    """创建示例数据（如果没有真实数据）"""
    print("📊 创建示例数据...")
    
    if os.path.exists('data/hotpotqa_train.json'):
        print("✅ 训练数据已存在")
        return
    
    # 创建最小示例数据
    sample_data = []
    for i in range(20):
        sample = {
            "document": f"This is sample document {i}. It contains information about topic {i}.",
            "query": f"What is the main topic of document {i}?",
            "entities": [
                {"span": [0, 4], "type": 0, "text": "This"},
                {"span": [8, 14], "type": 1, "text": "sample"}
            ],
            "relations": [
                {"head": 0, "tail": 1, "type": 0}
            ],
            "label": i % 5,
            "metadata": {"source": "sample", "id": f"sample_{i}"}
        }
        sample_data.append(sample)
    
    # 保存训练和验证数据
    train_data = sample_data[:16]
    val_data = sample_data[16:]
    
    with open('data/hotpotqa_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('data/hotpotqa_val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✅ 示例数据: {len(train_data)} 训练, {len(val_data)} 验证")


def check_files():
    """检查必要文件"""
    print("🔍 检查项目文件...")
    
    required_files = [
        'gdmnet/__init__.py',
        'gdmnet/model.py',
        'train/train.py',
        'config/colab_config.yaml',
        'data/hotpotqa_train.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ 缺少 {len(missing_files)} 个文件")
        print("请上传完整的项目代码")
        return False
    else:
        print("✅ 所有必要文件都存在")
        return True


def show_next_steps():
    """显示后续步骤"""
    print("\n🎉 Colab环境设置完成！")
    print("=" * 50)
    print("📋 后续步骤:")
    print("1. 上传项目代码文件（如果还没有）")
    print("2. 上传真实数据文件到 data/ 目录")
    print("3. 运行训练:")
    print("   !python train/train.py --config config/colab_config.yaml --mode train")
    print("4. 监控训练:")
    print("   %load_ext tensorboard")
    print("   %tensorboard --logdir logs/")
    print("\n🚀 开始您的GDM-Net训练之旅！")


def main():
    """主函数"""
    print("🧠 GDM-Net Google Colab 快速设置")
    print("=" * 60)
    
    # 执行设置步骤
    install_dependencies()
    print()
    
    check_gpu()
    print()
    
    create_directories()
    print()
    
    create_colab_config()
    print()
    
    create_sample_data()
    print()
    
    files_ok = check_files()
    print()
    
    show_next_steps()
    
    return files_ok


if __name__ == "__main__":
    main()
