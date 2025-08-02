# GDM-Net 命令参考

## 🚀 快速开始

### 一键设置和训练

```bash
# Linux/Mac - 完整设置
bash setup_and_train.sh

# Linux/Mac - 快速设置  
bash quick_start.sh

# Windows
setup_and_train.bat

# 使用 Make (推荐)
make env-setup        # 完整设置
make quick-start      # 快速设置
```

## 📦 环境管理

### 创建环境

```bash
# 方法1: 使用 conda
conda env create -f environment.yml

# 方法2: 使用 Make
make env-create
```

### 环境操作

```bash
# 激活环境
conda activate gdmnet

# 停用环境
conda deactivate

# 查看环境
conda env list

# 删除环境
conda env remove -n gdmnet
```

## 🔧 安装和测试

### 安装包

```bash
# 激活环境后安装
conda activate gdmnet
pip install -e .

# 或使用 Make
make install
```

### 测试安装

```bash
# 运行测试
python test_installation.py

# 或使用 Make
make test
```

## 📊 数据准备

### 创建合成数据

```bash
# 生成训练数据
python train/dataset.py

# 或使用 Make
make data
```

### 数据格式

```json
{
  "document": "文档内容",
  "query": "查询问题", 
  "entities": [{"span": [0, 5], "type": "PERSON"}],
  "relations": [{"head": 0, "tail": 1, "type": "WORKS_FOR"}],
  "label": 0
}
```

## 🏋️ 训练模型

### 基础训练

```bash
# 使用默认配置
python train/train.py --config config/model_config.yaml --mode train

# 或使用 Make
make train
```

### 自定义训练

```bash
# 创建合成数据并训练
python train/train.py --config config/model_config.yaml --mode train --create_synthetic

# 使用自定义配置
python train/train.py --config my_config.yaml --mode train
```

## 📈 评估模型

### 模型评估

```bash
# 评估训练好的模型
python train/train.py \
  --config config/model_config.yaml \
  --mode eval \
  --model_path checkpoints/gdmnet-epoch=05-val_loss=0.25.ckpt

# 使用 Make (需要设置 MODEL_PATH)
make eval MODEL_PATH=checkpoints/best_model.ckpt
```

## 🔍 监控和可视化

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 在浏览器中查看
# http://localhost:6006
```

### 查看训练进度

```bash
# 实时查看日志
tail -f logs/gdmnet/version_0/events.out.tfevents.*

# 查看检查点
ls -la checkpoints/
```

## 🧪 运行示例

### 基础示例

```bash
# 运行使用示例
python examples/example_usage.py

# 或使用 Make
make example
```

## 🛠 开发命令

### 代码格式化

```bash
# 格式化代码
make format

# 检查代码风格
make lint
```

### 清理文件

```bash
# 清理生成的文件
make clean

# 手动清理
rm -rf data/ checkpoints/ logs/ wandb/
```

## ⚙️ 配置修改

### 模型配置

编辑 `config/model_config.yaml`:

```yaml
model:
  hidden_size: 768          # 隐藏层大小
  num_entities: 10          # 实体类型数
  num_relations: 20         # 关系类型数
  gnn_type: "rgcn"         # GNN类型: rgcn/gat

training:
  batch_size: 8            # 批次大小
  max_epochs: 10           # 训练轮数
  learning_rate: 2e-5      # 学习率
  accelerator: "auto"      # 设备: auto/gpu/cpu
```

### GPU 配置

```yaml
training:
  accelerator: "gpu"       # 使用GPU
  devices: 1               # GPU数量
  precision: 16            # 混合精度
```

## 🐛 故障排除

### 常见问题

```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 PyTorch 安装
python -c "import torch; print(torch.cuda.is_available())"

# 重新创建环境
conda env remove -n gdmnet
conda env create -f environment.yml

# 清理缓存
pip cache purge
conda clean --all
```

### 性能优化

```bash
# 设置环境变量
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

# 减少批次大小
# 在配置文件中修改 batch_size: 4
```

## 📁 重要文件位置

```
GDM-Net/
├── config/model_config.yaml    # 模型配置
├── data/                       # 训练数据
├── checkpoints/                # 模型检查点
├── logs/                       # 训练日志
├── environment.yml             # Conda环境
├── requirements.txt            # Python依赖
└── examples/                   # 使用示例
```

## 🎯 完整工作流程

```bash
# 1. 设置环境
conda env create -f environment.yml
conda activate gdmnet

# 2. 安装包
pip install -e .

# 3. 测试安装
python test_installation.py

# 4. 创建数据
python train/dataset.py

# 5. 训练模型
python train/train.py --config config/model_config.yaml --mode train

# 6. 监控训练
tensorboard --logdir logs/

# 7. 评估模型
python train/train.py --config config/model_config.yaml --mode eval --model_path checkpoints/best_model.ckpt

# 8. 运行示例
python examples/example_usage.py
```

## 💡 提示

- 首次运行会下载 BERT 模型 (~400MB)
- 建议使用 SSD 存储以提高性能
- 长时间训练建议使用 `screen` 或 `tmux`
- 定期备份重要的检查点文件
- 使用 `make help` 查看所有可用命令
