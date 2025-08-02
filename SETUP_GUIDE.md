# GDM-Net 环境设置和训练指南

本指南将帮助您快速设置 GDM-Net 环境并开始训练。

## 🚀 一键设置和训练

### 方法一：自动化脚本（推荐）

```bash
# 1. 克隆项目（如果还没有）
git clone <your-repo-url>
cd GDM-Net

# 2. 运行自动化设置脚本
bash setup_and_train.sh
```

这个脚本会自动完成：
- ✅ 检查系统要求（conda、CUDA）
- ✅ 创建 conda 环境
- ✅ 安装所有依赖
- ✅ 测试安装
- ✅ 生成合成数据
- ✅ 开始训练

### 方法二：手动步骤

如果您喜欢手动控制每个步骤：

#### 1. 创建 Conda 环境

```bash
# 使用 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate gdmnet
```

#### 2. 安装 GDM-Net 包

```bash
# 安装包（开发模式）
pip install -e .
```

#### 3. 测试安装

```bash
# 运行安装测试
python test_installation.py
```

#### 4. 创建训练数据

```bash
# 生成合成数据集
python train/dataset.py
```

#### 5. 开始训练

```bash
# 使用默认配置训练
python train/train.py --config config/model_config.yaml --mode train
```

## 📋 系统要求

### 必需
- **Python**: 3.8-3.10
- **Conda**: Anaconda 或 Miniconda
- **内存**: 至少 8GB RAM
- **存储**: 至少 5GB 可用空间

### 推荐
- **GPU**: NVIDIA GPU with CUDA 11.7/11.8
- **内存**: 16GB+ RAM
- **存储**: SSD 存储

### CUDA 版本说明

根据您的 CUDA 版本修改 `environment.yml`：

```yaml
# CUDA 11.7
- pytorch-cuda=11.7

# CUDA 11.8  
- pytorch-cuda=11.8

# CPU 版本（无GPU）
# 删除 pytorch-cuda 行
```

## 🔧 环境配置详解

### Conda 环境内容

创建的 `gdmnet` 环境包含：

```
📦 核心框架
├── PyTorch 1.12+          # 深度学习框架
├── PyTorch Geometric      # 图神经网络
├── PyTorch Lightning      # 训练框架
└── Transformers          # BERT 模型

📦 数据处理
├── NumPy                  # 数值计算
├── Pandas                 # 数据处理
└── Datasets               # 数据集工具

📦 实验工具
├── TensorBoard            # 可视化
├── Weights & Biases       # 实验跟踪
└── scikit-learn          # 评估指标
```

### 环境管理命令

```bash
# 激活环境
conda activate gdmnet

# 停用环境
conda deactivate

# 查看环境信息
conda info --envs

# 删除环境（如果需要重新创建）
conda env remove -n gdmnet
```

## 🏋️ 训练配置

### 基础训练

```bash
# 使用默认配置
python train/train.py --config config/model_config.yaml --mode train
```

### 自定义配置

编辑 `config/model_config.yaml`：

```yaml
# 模型配置
model:
  hidden_size: 768          # 隐藏层大小
  num_entities: 10          # 实体类型数
  num_relations: 20         # 关系类型数
  num_classes: 5            # 输出类别数

# 训练配置
training:
  max_epochs: 10            # 最大训练轮数
  batch_size: 8             # 批次大小
  learning_rate: 2e-5       # 学习率
  accelerator: "auto"       # 自动选择设备
```

### GPU 训练

```yaml
training:
  accelerator: "gpu"        # 使用GPU
  devices: 1                # GPU数量
  precision: 16             # 混合精度训练
```

### 多GPU 训练

```yaml
training:
  accelerator: "gpu"
  devices: 2                # 使用2个GPU
  strategy: "ddp"           # 分布式训练
```

## 📊 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 在浏览器中打开
# http://localhost:6006
```

### Weights & Biases

```yaml
# 在配置文件中启用
logging:
  type: "wandb"
  project: "gdmnet-experiments"
  name: "my-experiment"
```

## 🔍 故障排除

### 常见问题

#### 1. CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvidia-smi

# 重新安装对应版本的 PyTorch
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### 2. 内存不足

```yaml
# 减少批次大小
training:
  batch_size: 4             # 从8减少到4
  accumulate_grad_batches: 2 # 梯度累积
```

#### 3. 依赖冲突

```bash
# 重新创建环境
conda env remove -n gdmnet
conda env create -f environment.yml
```

#### 4. 训练速度慢

```yaml
# 优化配置
training:
  num_workers: 8            # 增加数据加载进程
  pin_memory: true          # 固定内存
  precision: 16             # 混合精度
```

### 性能优化

#### CPU 优化

```bash
# 设置线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

#### GPU 优化

```bash
# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 优化 CUDA 缓存
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 📁 项目结构

训练完成后的项目结构：

```
GDM-Net/
├── data/                   # 训练数据
│   ├── train.json
│   ├── val.json
│   └── test.json
├── checkpoints/            # 模型检查点
│   └── gdmnet-epoch=XX-val_loss=X.XX.ckpt
├── logs/                   # 训练日志
│   └── gdmnet/
└── wandb/                  # W&B 日志（如果启用）
```

## 🎯 下一步

训练完成后，您可以：

1. **评估模型**：
   ```bash
   python train/train.py --config config/model_config.yaml --mode eval --model_path checkpoints/best_model.ckpt
   ```

2. **运行推理**：
   ```bash
   python examples/example_usage.py
   ```

3. **自定义数据**：
   - 准备您自己的数据集
   - 修改配置文件
   - 重新训练

4. **实验跟踪**：
   - 使用 TensorBoard 查看训练曲线
   - 使用 W&B 进行实验管理

## 💡 提示

- 首次运行会下载 BERT 模型，需要网络连接
- 建议在 SSD 上运行以获得更好的性能
- 使用 `screen` 或 `tmux` 进行长时间训练
- 定期备份重要的检查点文件

## 📞 获取帮助

如果遇到问题：

1. 检查错误日志
2. 查看 [故障排除](#故障排除) 部分
3. 运行 `python test_installation.py` 诊断问题
4. 提交 GitHub Issue 并附上详细错误信息
