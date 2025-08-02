# GDM-Net Google Colab 训练指南

本指南将帮助您在Google Colab上训练GDM-Net模型，充分利用免费的GPU资源。

## 🚀 快速开始

### 1. 打开Google Colab

访问 [Google Colab](https://colab.research.google.com/) 并创建新的笔记本。

### 2. 检查GPU可用性

```python
# 检查GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("No GPU available, using CPU")

# 检查系统信息
!nvidia-smi
```

### 3. 安装依赖

```python
# 安装必要的包
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install torch-geometric
!pip install transformers>=4.20.0
!pip install pytorch-lightning>=1.7.0
!pip install datasets>=2.0.0
!pip install PyYAML>=6.0
!pip install tensorboard>=2.8.0
!pip install wandb>=0.12.0
!pip install tqdm>=4.64.0
!pip install scikit-learn>=1.1.0
!pip install matplotlib>=3.5.0
!pip install seaborn>=0.11.0

print("✅ 依赖安装完成")
```

### 4. 克隆或上传项目代码

```python
# 方法1: 从GitHub克隆（如果您的代码在GitHub上）
# !git clone https://github.com/your-username/GDM-Net.git
# %cd GDM-Net

# 方法2: 上传代码文件
# 使用Colab的文件上传功能上传您的项目文件

# 方法3: 从Google Drive挂载
from google.colab import drive
drive.mount('/content/drive')

# 如果代码在Google Drive中
# %cd /content/drive/MyDrive/GDM-Net

print("✅ 项目代码准备完成")
```

## 📁 项目结构设置

```python
# 创建项目目录结构
import os

# 创建必要的目录
directories = [
    'gdmnet',
    'train', 
    'config',
    'data',
    'checkpoints',
    'logs',
    'examples'
]

for dir_name in directories:
    os.makedirs(dir_name, exist_ok=True)
    print(f"✅ 创建目录: {dir_name}")
```

## 🔧 Colab优化配置

```python
# 创建Colab优化的配置文件
colab_config = """
# GDM-Net Colab Configuration - GPU Optimized

seed: 42

model:
  bert_model_name: "bert-base-uncased"
  hidden_size: 768
  num_entities: 8
  num_relations: 4
  num_classes: 5
  gnn_type: "rgcn"
  num_gnn_layers: 2
  num_reasoning_hops: 3
  fusion_method: "gate"
  learning_rate: 2e-5
  dropout_rate: 0.1

data:
  train_path: "data/hotpotqa_train.json"
  val_path: "data/hotpotqa_val.json"
  test_path: "data/hotpotqa_val.json"
  max_length: 512
  max_query_length: 64

training:
  max_epochs: 10
  batch_size: 8  # GPU可以使用更大的batch size
  num_workers: 2  # Colab推荐值
  accelerator: "gpu"
  devices: 1
  precision: 16  # 混合精度训练，节省GPU内存
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  val_check_interval: 0.5  # 每半个epoch验证一次
  log_every_n_steps: 50
  checkpoint_dir: "checkpoints"
  early_stopping: true
  patience: 3

logging:
  type: "tensorboard"
  save_dir: "logs"
  name: "gdmnet-colab"
"""

# 保存配置文件
with open('config/colab_config.yaml', 'w') as f:
    f.write(colab_config)

print("✅ Colab配置文件创建完成")
```

## 📊 数据准备

```python
# 下载和准备HotpotQA数据集
import json
import requests
from tqdm import tqdm

def download_hotpotqa_data():
    """下载HotpotQA数据集"""
    
    # 如果您已经有数据文件，可以跳过下载
    if os.path.exists('data/hotpotqa_train.json'):
        print("✅ 数据文件已存在")
        return
    
    print("📥 下载HotpotQA数据集...")
    
    # 这里使用您已经准备好的数据
    # 或者从您的Google Drive复制
    
    # 示例：从Drive复制数据
    if os.path.exists('/content/drive/MyDrive/GDM-Net/data/'):
        !cp -r /content/drive/MyDrive/GDM-Net/data/* ./data/
        print("✅ 从Google Drive复制数据完成")
    else:
        print("⚠️ 请上传数据文件到data/目录")

# 执行数据准备
download_hotpotqa_data()

# 检查数据
if os.path.exists('data/hotpotqa_train.json'):
    with open('data/hotpotqa_train.json', 'r') as f:
        data = json.load(f)
    print(f"📊 训练数据: {len(data)} 样本")
    
    # 显示数据样本
    print("📋 数据样本:")
    sample = data[0]
    print(f"文档: {sample['document'][:100]}...")
    print(f"查询: {sample['query']}")
    print(f"实体数量: {len(sample['entities'])}")
    print(f"关系数量: {len(sample['relations'])}")
```

## 🧠 模型代码部署

```python
# 如果需要直接在Colab中定义模型代码
# 这里可以复制您的gdmnet模块代码

# 或者从文件导入
import sys
sys.path.append('/content')  # 添加当前目录到Python路径

# 测试导入
try:
    from gdmnet import GDMNet
    print("✅ GDM-Net模型导入成功")
except ImportError as e:
    print(f"❌ 模型导入失败: {e}")
    print("请确保所有模型文件都已上传")
```

## 🏋️ 开始训练

```python
# 设置训练环境
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免tokenizer警告

# 开始训练
!python train/train.py --config config/colab_config.yaml --mode train

print("🎉 训练启动完成！")
```

## 📈 监控训练进度

```python
# 在新的cell中运行TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs/

# 或者使用命令行
# !tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

```python
# 实时查看训练日志
import time
import glob

def monitor_training():
    """监控训练进度"""
    while True:
        # 检查最新的日志文件
        log_files = glob.glob('logs/gdmnet-colab/version_*/events.out.tfevents.*')
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            print(f"📊 最新日志: {latest_log}")
        
        # 检查检查点
        checkpoints = glob.glob('checkpoints/*.ckpt')
        if checkpoints:
            latest_ckpt = max(checkpoints, key=os.path.getctime)
            print(f"💾 最新检查点: {latest_ckpt}")
        
        time.sleep(30)  # 每30秒检查一次

# 在后台运行监控（可选）
# monitor_training()
```

## 💾 保存和下载结果

```python
# 训练完成后，保存重要文件到Google Drive
def save_results_to_drive():
    """保存训练结果到Google Drive"""
    
    # 创建结果目录
    result_dir = '/content/drive/MyDrive/GDM-Net-Results'
    os.makedirs(result_dir, exist_ok=True)
    
    # 复制检查点
    !cp -r checkpoints/* /content/drive/MyDrive/GDM-Net-Results/
    
    # 复制日志
    !cp -r logs/* /content/drive/MyDrive/GDM-Net-Results/
    
    print("✅ 结果已保存到Google Drive")

# 下载最佳模型
def download_best_model():
    """下载最佳模型到本地"""
    import glob
    from google.colab import files
    
    # 找到最佳检查点
    checkpoints = glob.glob('checkpoints/*.ckpt')
    if checkpoints:
        best_model = min(checkpoints, key=lambda x: float(x.split('val_loss=')[1].split('-')[0]))
        print(f"📥 下载最佳模型: {best_model}")
        files.download(best_model)
    else:
        print("❌ 未找到检查点文件")

# 执行保存
save_results_to_drive()
```

## 🧪 模型测试和推理

```python
# 加载训练好的模型进行测试
import torch
from gdmnet import GDMNet

def test_trained_model():
    """测试训练好的模型"""
    
    # 找到最佳检查点
    import glob
    checkpoints = glob.glob('checkpoints/*.ckpt')
    if not checkpoints:
        print("❌ 未找到检查点文件")
        return
    
    best_model_path = min(checkpoints, key=lambda x: float(x.split('val_loss=')[1].split('-')[0]))
    print(f"🧠 加载模型: {best_model_path}")
    
    # 加载模型
    model = GDMNet.load_from_checkpoint(best_model_path)
    model.eval()
    
    print("✅ 模型加载成功")
    print(f"📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

# 测试模型
trained_model = test_trained_model()
```

```python
# 运行推理示例
def run_inference_example(model):
    """运行推理示例"""
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 示例输入
    document = "Apple Inc. is a technology company founded by Steve Jobs. Tim Cook is the current CEO."
    query = "Who is the CEO of Apple?"
    
    # 编码输入
    doc_encoding = tokenizer(
        document,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    query_encoding = tokenizer(
        query,
        max_length=64,
        padding='max_length', 
        truncation=True,
        return_tensors='pt'
    )
    
    # 推理
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()
            doc_encoding = {k: v.cuda() for k, v in doc_encoding.items()}
            query_encoding = {k: v.cuda() for k, v in query_encoding.items()}
        
        outputs = model(
            input_ids=doc_encoding['input_ids'],
            attention_mask=doc_encoding['attention_mask'],
            query=query_encoding['input_ids']
        )
    
    # 显示结果
    prediction = torch.argmax(outputs, dim=-1)
    confidence = torch.softmax(outputs, dim=-1).max()
    
    print(f"🎯 预测结果: {prediction.item()}")
    print(f"📊 置信度: {confidence.item():.3f}")

# 运行推理
if 'trained_model' in locals():
    run_inference_example(trained_model)
```

## ⚠️ Colab注意事项

### 1. 会话管理
```python
# 防止会话超时
import time
from IPython.display import Javascript

def keep_alive():
    """保持Colab会话活跃"""
    display(Javascript('''
        function ClickConnect(){
            console.log("Working");
            document.querySelector("colab-toolbar-button#connect").click()
        }
        setInterval(ClickConnect,60000)
    '''))

# 运行保活脚本（可选）
# keep_alive()
```

### 2. 内存管理
```python
# 清理GPU内存
import gc
import torch

def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ 内存清理完成")

# 在需要时调用
clear_memory()
```

### 3. 检查点恢复
```python
# 从检查点恢复训练
def resume_training():
    """从检查点恢复训练"""
    import glob
    
    checkpoints = glob.glob('checkpoints/*.ckpt')
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"🔄 从检查点恢复: {latest_ckpt}")
        
        # 修改训练命令以包含resume_from_checkpoint
        !python train/train.py --config config/colab_config.yaml --mode train --resume_from_checkpoint {latest_ckpt}
    else:
        print("❌ 未找到检查点，开始新的训练")
        !python train/train.py --config config/colab_config.yaml --mode train

# 如果需要恢复训练
# resume_training()
```

## 🎉 完整训练流程

将以上所有代码按顺序在Colab中执行，您就可以成功在GPU上训练GDM-Net模型了！

### 预期性能
- **GPU训练速度**: 比CPU快10-20倍
- **内存使用**: 约4-6GB GPU内存
- **训练时间**: 完整数据集约1-2小时

### 最终检查清单
- ✅ GPU可用性确认
- ✅ 依赖包安装
- ✅ 数据集准备
- ✅ 模型代码部署
- ✅ 配置文件优化
- ✅ 训练启动
- ✅ 进度监控
- ✅ 结果保存

祝您训练顺利！🚀
