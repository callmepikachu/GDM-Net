"""
GDM-Net 优化指南
解决GPU加速、内存占用和真实数据集接入问题
"""

import os
import json
import torch
import yaml
from pathlib import Path


def check_system_info():
    """检查系统信息"""
    print("🔍 系统信息检查")
    print("=" * 40)
    
    # PyTorch信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ 未检测到CUDA支持")
        print("💡 解决方案:")
        print("1. 检查是否有NVIDIA GPU")
        print("2. 安装CUDA驱动")
        print("3. 重新安装支持CUDA的PyTorch")
    
    # 内存信息
    import psutil
    memory = psutil.virtual_memory()
    print(f"\n💾 系统内存:")
    print(f"总内存: {memory.total / (1024**3):.1f} GB")
    print(f"可用内存: {memory.available / (1024**3):.1f} GB")
    print(f"使用率: {memory.percent}%")


def create_gpu_environment():
    """创建支持GPU的环境配置"""
    print("\n🚀 创建GPU环境配置")
    print("=" * 40)
    
    gpu_env_config = {
        'name': 'gdmnet-gpu',
        'channels': ['pytorch', 'nvidia', 'conda-forge', 'defaults'],
        'dependencies': [
            'python=3.9',
            'pytorch>=1.12.0',
            'torchvision',
            'torchaudio',
            'pytorch-cuda=11.8',  # 或 11.7
            'cudatoolkit',
            'pip',
            {
                'pip': [
                    'torch-geometric>=2.1.0',
                    'transformers>=4.20.0',
                    'pytorch-lightning>=1.7.0',
                    'datasets>=2.0.0',
                    'numpy>=1.21.0',
                    'pandas>=1.3.0',
                    'PyYAML>=6.0',
                    'tensorboard>=2.8.0',
                    'wandb>=0.12.0',
                    'tqdm>=4.64.0',
                    'scikit-learn>=1.1.0',
                    'matplotlib>=3.5.0',
                    'seaborn>=0.11.0',
                    'psutil'  # 用于内存监控
                ]
            }
        ]
    }
    
    with open('environment_gpu.yml', 'w', encoding='utf-8') as f:
        yaml.dump(gpu_env_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ 已创建 environment_gpu.yml")
    print("💡 使用方法:")
    print("conda env create -f environment_gpu.yml")
    print("conda activate gdmnet-gpu")


def create_optimized_config():
    """创建优化的训练配置"""
    print("\n⚡ 创建优化配置")
    print("=" * 40)
    
    # 检查是否有GPU
    has_gpu = torch.cuda.is_available()
    
    optimized_config = {
        'seed': 42,
        'model': {
            'bert_model_name': 'bert-base-uncased',
            'hidden_size': 768,
            'num_entities': 10,
            'num_relations': 20,
            'num_classes': 5,
            'gnn_type': 'rgcn',
            'num_gnn_layers': 2,
            'num_reasoning_hops': 3,
            'fusion_method': 'gate',
            'learning_rate': 2e-5,
            'dropout_rate': 0.1
        },
        'data': {
            'train_path': 'data/train.json',
            'val_path': 'data/val.json',
            'test_path': 'data/test.json',
            'max_length': 256,  # 减少序列长度以节省内存
            'max_query_length': 32
        },
        'training': {
            'max_epochs': 5,  # 减少epoch数
            'batch_size': 2 if not has_gpu else 8,  # CPU用小batch size
            'num_workers': 2,  # 减少worker数量
            'accelerator': 'gpu' if has_gpu else 'cpu',
            'devices': 1,
            'precision': 32,  # 使用32位精度避免问题
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 4 if not has_gpu else 1,  # CPU用梯度累积
            'val_check_interval': 1.0,
            'log_every_n_steps': 10,
            'checkpoint_dir': 'checkpoints',
            'early_stopping': True,
            'patience': 3
        },
        'logging': {
            'type': 'tensorboard',
            'save_dir': 'logs',
            'name': 'gdmnet-optimized'
        }
    }
    
    with open('config/optimized_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ 已创建 config/optimized_config.yaml")
    print(f"🎯 配置特点:")
    print(f"- 设备: {'GPU' if has_gpu else 'CPU'}")
    print(f"- 批次大小: {optimized_config['training']['batch_size']}")
    print(f"- 序列长度: {optimized_config['data']['max_length']}")
    print(f"- 梯度累积: {optimized_config['training']['accumulate_grad_batches']}")


def create_real_dataset_template():
    """创建真实数据集模板和转换脚本"""
    print("\n📊 创建真实数据集模板")
    print("=" * 40)
    
    # 创建数据集模板
    template = {
        "document": "Your document text here",
        "query": "Your query here", 
        "entities": [
            {
                "span": [0, 5],  # 字符位置 [start, end]
                "type": "PERSON",  # 实体类型
                "text": "Apple"  # 实体文本（可选）
            }
        ],
        "relations": [
            {
                "head": 0,  # 头实体索引
                "tail": 1,  # 尾实体索引
                "type": "CEO_OF"  # 关系类型
            }
        ],
        "label": 0,  # 分类标签
        "metadata": {
            "source": "dataset_name",
            "id": "sample_001"
        }
    }
    
    # 保存模板
    with open('data_template.json', 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    # 创建数据转换脚本
    converter_script = '''"""
真实数据集转换脚本
将您的数据转换为GDM-Net格式
"""

import json
import pandas as pd
from typing import List, Dict, Any


def convert_from_csv(csv_path: str, output_path: str):
    """从CSV文件转换数据"""
    df = pd.read_csv(csv_path)
    
    converted_data = []
    for _, row in df.iterrows():
        sample = {
            "document": row['document'],
            "query": row.get('query', ''),
            "entities": json.loads(row.get('entities', '[]')),
            "relations": json.loads(row.get('relations', '[]')),
            "label": int(row.get('label', 0)),
            "metadata": {"source": "csv", "id": str(row.get('id', ''))}
        }
        converted_data.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成: {len(converted_data)} 个样本 -> {output_path}")


def convert_from_jsonl(jsonl_path: str, output_path: str):
    """从JSONL文件转换数据"""
    converted_data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # 根据您的数据格式调整这里
                sample = {
                    "document": data.get('text', data.get('document', '')),
                    "query": data.get('query', ''),
                    "entities": data.get('entities', []),
                    "relations": data.get('relations', []),
                    "label": data.get('label', 0),
                    "metadata": data.get('metadata', {})
                }
                converted_data.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成: {len(converted_data)} 个样本 -> {output_path}")


def convert_docred_format(docred_path: str, output_path: str):
    """转换DocRED格式数据"""
    with open(docred_path, 'r', encoding='utf-8') as f:
        docred_data = json.load(f)
    
    converted_data = []
    for item in docred_data:
        # 合并句子
        document = ' '.join([' '.join(sent) for sent in item['sents']])
        
        # 转换实体
        entities = []
        for entity in item['vertexSet']:
            for mention in entity:
                entities.append({
                    "span": [mention['pos'][0], mention['pos'][1]],
                    "type": mention.get('type', 'MISC'),
                    "text": mention['name']
                })
        
        # 转换关系
        relations = []
        for relation in item.get('labels', []):
            relations.append({
                "head": relation['h'],
                "tail": relation['t'],
                "type": relation['r']
            })
        
        sample = {
            "document": document,
            "query": "",  # DocRED没有查询
            "entities": entities,
            "relations": relations,
            "label": 0,  # 默认标签
            "metadata": {"source": "docred", "title": item.get('title', '')}
        }
        converted_data.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成: {len(converted_data)} 个样本 -> {output_path}")


if __name__ == "__main__":
    print("数据转换脚本")
    print("请根据您的数据格式选择相应的转换函数")
    print("1. convert_from_csv() - CSV格式")
    print("2. convert_from_jsonl() - JSONL格式") 
    print("3. convert_docred_format() - DocRED格式")
'''
    
    with open('convert_dataset.py', 'w', encoding='utf-8') as f:
        f.write(converter_script)
    
    print("✅ 已创建数据集模板和转换脚本:")
    print("- data_template.json: 数据格式模板")
    print("- convert_dataset.py: 数据转换脚本")


def create_memory_monitor():
    """创建内存监控脚本"""
    print("\n💾 创建内存监控脚本")
    print("=" * 40)
    
    monitor_script = '''"""
内存监控脚本
监控训练过程中的内存使用情况
"""

import psutil
import torch
import time
import matplotlib.pyplot as plt
from collections import deque
import threading


class MemoryMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.cpu_memory = deque(maxlen=max_points)
        self.gpu_memory = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        self.monitoring = False
        
    def start_monitoring(self, interval=1.0):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("内存监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        print("内存监控已停止")
    
    def _monitor_loop(self, interval):
        """监控循环"""
        start_time = time.time()
        
        while self.monitoring:
            current_time = time.time() - start_time
            
            # CPU内存
            cpu_mem = psutil.virtual_memory()
            cpu_usage = cpu_mem.used / (1024**3)  # GB
            
            # GPU内存
            gpu_usage = 0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            self.cpu_memory.append(cpu_usage)
            self.gpu_memory.append(gpu_usage)
            self.timestamps.append(current_time)
            
            time.sleep(interval)
    
    def plot_memory_usage(self, save_path='memory_usage.png'):
        """绘制内存使用图"""
        if not self.timestamps:
            print("没有监控数据")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(list(self.timestamps), list(self.cpu_memory), 'b-', label='CPU Memory')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (GB)')
        plt.title('CPU Memory Usage')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if torch.cuda.is_available():
            plt.plot(list(self.timestamps), list(self.gpu_memory), 'r-', label='GPU Memory')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory Usage (GB)')
            plt.title('GPU Memory Usage')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No GPU Available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('GPU Memory Usage')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"内存使用图已保存到: {save_path}")
    
    def get_current_usage(self):
        """获取当前内存使用情况"""
        cpu_mem = psutil.virtual_memory()
        cpu_usage = cpu_mem.used / (1024**3)
        cpu_percent = cpu_mem.percent
        
        gpu_usage = 0
        gpu_percent = 0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_percent = (gpu_usage / gpu_total) * 100 if gpu_total > 0 else 0
        
        return {
            'cpu_usage_gb': cpu_usage,
            'cpu_percent': cpu_percent,
            'gpu_usage_gb': gpu_usage,
            'gpu_percent': gpu_percent
        }


def optimize_memory_usage():
    """内存优化建议"""
    print("💡 内存优化建议:")
    print("1. 减少batch_size")
    print("2. 减少max_length")
    print("3. 使用gradient_checkpointing")
    print("4. 使用混合精度训练")
    print("5. 定期清理GPU缓存: torch.cuda.empty_cache()")


if __name__ == "__main__":
    monitor = MemoryMonitor()
    
    # 显示当前内存使用
    usage = monitor.get_current_usage()
    print(f"当前内存使用:")
    print(f"CPU: {usage['cpu_usage_gb']:.1f} GB ({usage['cpu_percent']:.1f}%)")
    print(f"GPU: {usage['gpu_usage_gb']:.1f} GB ({usage['gpu_percent']:.1f}%)")
    
    optimize_memory_usage()
'''
    
    with open('memory_monitor.py', 'w', encoding='utf-8') as f:
        f.write(monitor_script)
    
    print("✅ 已创建 memory_monitor.py")


def main():
    """主函数"""
    print("🛠️  GDM-Net 优化指南")
    print("=" * 50)
    
    # 检查系统信息
    check_system_info()
    
    # 创建GPU环境配置
    create_gpu_environment()
    
    # 创建优化配置
    create_optimized_config()
    
    # 创建数据集模板
    create_real_dataset_template()
    
    # 创建内存监控
    create_memory_monitor()
    
    print("\n🎉 优化指南创建完成！")
    print("\n📋 后续操作:")
    print("1. GPU加速: 如果有GPU，运行 conda env create -f environment_gpu.yml")
    print("2. 内存优化: 使用 config/optimized_config.yaml 训练")
    print("3. 真实数据: 参考 data_template.json 准备数据，使用 convert_dataset.py 转换")
    print("4. 内存监控: 运行 python memory_monitor.py")


if __name__ == "__main__":
    main()
