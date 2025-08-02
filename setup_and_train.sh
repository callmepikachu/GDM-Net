#!/bin/bash

# GDM-Net 环境设置和训练脚本
# 使用方法: bash setup_and_train.sh

set -e  # 遇到错误时退出

echo "🚀 开始设置 GDM-Net 环境和训练..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查conda是否安装
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda 未安装。请先安装 Anaconda 或 Miniconda。"
        echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_status "Conda 已安装: $(conda --version)"
}

# 检查CUDA版本
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "检测到 NVIDIA GPU:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
        
        # 获取CUDA版本
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
            print_status "CUDA 版本: $CUDA_VERSION"
        else
            print_warning "nvcc 未找到，将使用默认CUDA版本"
        fi
    else
        print_warning "未检测到 NVIDIA GPU，将使用 CPU 版本"
    fi
}

# 创建conda环境
create_environment() {
    print_step "1. 创建 conda 环境..."
    
    # 检查环境是否已存在
    if conda env list | grep -q "gdmnet"; then
        print_warning "环境 'gdmnet' 已存在"
        read -p "是否删除并重新创建? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "删除现有环境..."
            conda env remove -n gdmnet -y
        else
            print_status "使用现有环境..."
            return 0
        fi
    fi
    
    # 创建环境
    print_status "从 environment.yml 创建环境..."
    conda env create -f environment.yml
    
    print_status "✅ Conda 环境创建完成"
}

# 激活环境并安装包
install_package() {
    print_step "2. 激活环境并安装 GDM-Net 包..."
    
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gdmnet
    
    # 安装包
    print_status "安装 GDM-Net 包..."
    pip install -e .
    
    print_status "✅ 包安装完成"
}

# 测试安装
test_installation() {
    print_step "3. 测试安装..."
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gdmnet
    
    print_status "运行安装测试..."
    python test_installation.py
    
    if [ $? -eq 0 ]; then
        print_status "✅ 安装测试通过"
    else
        print_error "❌ 安装测试失败"
        exit 1
    fi
}

# 创建合成数据
create_data() {
    print_step "4. 创建训练数据..."
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gdmnet
    
    # 创建数据目录
    mkdir -p data
    
    print_status "生成合成数据集..."
    python train/dataset.py
    
    print_status "✅ 数据创建完成"
    print_status "数据文件位置:"
    echo "  - 训练集: data/train.json (800 samples)"
    echo "  - 验证集: data/val.json (100 samples)"
    echo "  - 测试集: data/test.json (100 samples)"
}

# 开始训练
start_training() {
    print_step "5. 开始训练模型..."
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gdmnet
    
    # 创建必要的目录
    mkdir -p checkpoints logs
    
    print_status "启动训练..."
    print_status "配置文件: config/model_config.yaml"
    print_status "检查点保存位置: checkpoints/"
    print_status "日志保存位置: logs/"
    
    # 开始训练
    python train/train.py --config config/model_config.yaml --mode train
    
    print_status "✅ 训练完成"
}

# 显示训练后的操作指南
show_next_steps() {
    print_step "🎉 设置和训练完成!"
    
    echo ""
    echo "📋 后续操作指南:"
    echo ""
    echo "1. 激活环境:"
    echo "   conda activate gdmnet"
    echo ""
    echo "2. 查看训练日志:"
    echo "   tensorboard --logdir logs/"
    echo ""
    echo "3. 评估模型 (替换为实际的检查点路径):"
    echo "   python train/train.py --config config/model_config.yaml --mode eval --model_path checkpoints/gdmnet-epoch=XX-val_loss=X.XX.ckpt"
    echo ""
    echo "4. 运行示例:"
    echo "   python examples/example_usage.py"
    echo ""
    echo "5. 使用 Makefile 命令:"
    echo "   make train    # 重新训练"
    echo "   make test     # 测试安装"
    echo "   make example  # 运行示例"
    echo "   make clean    # 清理文件"
    echo ""
    echo "📁 重要文件位置:"
    echo "   - 配置文件: config/model_config.yaml"
    echo "   - 训练数据: data/"
    echo "   - 模型检查点: checkpoints/"
    echo "   - 训练日志: logs/"
    echo ""
}

# 主函数
main() {
    echo "🧠 GDM-Net (Graph-Augmented Dual Memory Network) 设置脚本"
    echo "================================================================"
    
    # 检查系统要求
    check_conda
    check_cuda
    
    # 执行设置步骤
    create_environment
    install_package
    test_installation
    create_data
    start_training
    
    # 显示后续步骤
    show_next_steps
}

# 错误处理
trap 'print_error "脚本执行失败，请检查错误信息"; exit 1' ERR

# 运行主函数
main "$@"
