#!/bin/bash

# GDM-Net 快速开始脚本
# 最简化的设置和训练流程

echo "🧠 GDM-Net 快速开始"
echo "=================="

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "❌ 请先安装 Anaconda 或 Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ 检测到 conda: $(conda --version)"

# 创建环境
echo "📦 创建 conda 环境..."
conda env create -f environment.yml -y

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gdmnet

# 安装包
echo "📥 安装 GDM-Net..."
pip install -e .

# 测试
echo "🧪 测试安装..."
python test_installation.py

# 创建数据
echo "📊 创建训练数据..."
mkdir -p data checkpoints logs
python train/dataset.py

# 开始训练
echo "🚀 开始训练..."
python train/train.py --config config/model_config.yaml --mode train

echo ""
echo "🎉 完成！"
echo ""
echo "后续操作："
echo "1. 激活环境: conda activate gdmnet"
echo "2. 查看日志: tensorboard --logdir logs/"
echo "3. 运行示例: python examples/example_usage.py"
