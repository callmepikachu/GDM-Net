# GDM-Net 使用指南

## 快速开始

### 1. 环境准备

确保您的系统满足以下要求：
- Python 3.8+
- CUDA 11.0+ (推荐用于GPU训练)
- 48GB GPU内存 (推荐配置)

### 2. 安装依赖

```bash
# 使用快速启动脚本安装
python quick_start.py --install

# 或手动安装
pip install -r requirements.txt
```

### 3. 数据准备

数据文件已放置在 `data/` 目录下：
- `hotpotqa_official_train.json` - 训练数据
- `hotpotqa_official_val.json` - 验证数据

### 4. 模型测试

在训练前，建议先运行测试确保模型正常工作：

```bash

# 或直接运行测试脚本
python test_model.py
```

### 5. 训练模型

```

#### 完整训练
```bash

# 或直接使用训练脚本
python src/train/train.py --config config/default_config.yaml
```

#### 从检查点恢复训练
```bash
python src/train/train.py --config config/default_config.yaml --resume_from_checkpoint checkpoints/last.ckpt
```

### 6. 模型评估

```bash
# 自动找到最新模型进行评估
python quick_start.py --evaluate

# 指定模型路径评估
python src/evaluate/evaluate.py --model_path checkpoints/best_model.pt --config config/default_config.yaml
```

### 7. 完整流程

运行完整的训练和评估流程：

```bash
python quick_start.py --all
```

## 配置说明

### 主要配置参数 (config/default_config.yaml)

```yaml
model:
  bert_model_name: "bert-base-uncased"  # BERT模型
  hidden_size: 768                      # 隐藏层大小
  num_entities: 9                       # 实体数量
  num_relations: 10                     # 关系数量
  num_classes: 5                        # 分类类别数
  gnn_type: "rgcn"                      # 图神经网络类型
  num_gnn_layers: 3                     # GNN层数
  num_reasoning_hops: 4                 # 推理跳数
  learning_rate: 2e-5                   # 学习率

training:
  max_epochs: 10                        # 最大训练轮数
  batch_size: 16                        # 批次大小
  num_workers: 4                        # 数据加载进程数
  accelerator: "gpu"                    # 加速器类型
  devices: 1                            # GPU设备数
```

### 性能优化配置

对于48GB GPU，当前配置已优化：
- `batch_size: 16` - 充分利用GPU内存
- `precision: 32` - 完整精度训练
- `accumulate_grad_batches: 1` - 无需梯度累积

如果GPU内存不足，可以调整：
- 减少 `batch_size` 到 8 或 4
- 增加 `accumulate_grad_batches` 到 2 或 4
- 使用 `precision: 16` 进行混合精度训练

## 模型架构

GDM-Net包含以下核心组件：

1. **BERT编码器** - 文档和查询编码
2. **图记忆模块** - 结构化知识表示
3. **双记忆系统** - 情节记忆和语义记忆
4. **多跳推理模块** - 迭代推理机制

## 输出文件

### 训练输出
- `checkpoints/` - 模型检查点
- `logs/` - 训练日志和TensorBoard文件

### 评估输出
- `evaluation_results/evaluation_results.json` - 评估指标
- `evaluation_results/detailed_outputs.json` - 详细预测结果

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用梯度累积
   - 启用混合精度训练

2. **数据加载错误**
   - 检查数据文件路径
   - 确认数据格式正确

3. **模型收敛慢**
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量

### 性能监控

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir logs/
```

## 扩展和自定义

### 添加新的数据集

1. 继承 `HotpotQADataset` 类
2. 实现数据加载逻辑
3. 更新配置文件

### 修改模型架构

1. 编辑 `src/model/` 下的相应模块
2. 更新配置参数
3. 重新训练模型

### 自定义评估指标

1. 在 `src/evaluate/metrics.py` 中添加新指标
2. 更新评估脚本

## 技术支持

如遇到问题，请检查：
1. 依赖版本是否正确
2. GPU驱动和CUDA版本
3. 数据文件完整性
4. 配置参数合理性
