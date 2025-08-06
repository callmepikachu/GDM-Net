#!/usr/bin/env python3
"""
自动化 GDM-Net 环境设置和测试脚本
自动检测系统配置并选择最佳安装方案
"""

import os
import sys
import subprocess
import platform
import shutil
import yaml
from pathlib import Path


class Colors:
    """控制台颜色"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_status(message, status="INFO"):
    """打印带颜色的状态信息"""
    colors = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "STEP": Colors.CYAN
    }
    color = colors.get(status, Colors.WHITE)
    print(f"{color}[{status}]{Colors.END} {message}")


def run_command(command, check=True, capture_output=True):
    """运行命令并返回结果"""
    try:
        if isinstance(command, str):
            command = command.split()
        
        result = subprocess.run(
            command, 
            check=check, 
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print_status(f"命令执行失败: {' '.join(command)}", "ERROR")
            print_status(f"错误信息: {e.stderr if e.stderr else e.stdout}", "ERROR")
        return None
    except FileNotFoundError:
        print_status(f"命令未找到: {command[0]}", "ERROR")
        return None


def check_conda():
    """检查 conda 是否安装"""
    print_status("检查 conda 安装...", "STEP")
    
    result = run_command("conda --version", check=False)
    if result and result.returncode == 0:
        print_status(f"找到 conda: {result.stdout.strip()}", "SUCCESS")
        return True
    else:
        print_status("未找到 conda，请先安装 Anaconda 或 Miniconda", "ERROR")
        print_status("下载地址: https://docs.conda.io/en/latest/miniconda.html", "INFO")
        return False


def check_cuda():
    """检查 CUDA 是否可用"""
    print_status("检查 CUDA 支持...", "STEP")
    
    # 检查 nvidia-smi
    result = run_command("nvidia-smi", check=False)
    if result and result.returncode == 0:
        print_status("检测到 NVIDIA GPU", "SUCCESS")
        
        # 尝试获取 CUDA 版本
        try:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print_status(f"CUDA 版本: {cuda_version}", "INFO")
                    return True, cuda_version
        except:
            pass
        
        return True, "unknown"
    else:
        print_status("未检测到 NVIDIA GPU 或 CUDA，将使用 CPU 版本", "WARNING")
        return False, None


def create_environment_yml(use_cuda=False, cuda_version=None):
    """创建适合的 environment.yml 文件"""
    print_status("创建环境配置文件...", "STEP")
    
    env_config = {
        'name': 'gdmnet',
        'channels': ['pytorch', 'pyg', 'conda-forge', 'defaults'],
        'dependencies': [
            'python=3.9',
            'pytorch>=1.12.0',
            'torchvision',
            'torchaudio',
        ]
    }
    
    if use_cuda and cuda_version and cuda_version != "unknown":
        # 尝试使用 CUDA 版本
        major_version = cuda_version.split('.')[0]
        minor_version = cuda_version.split('.')[1]
        
        if major_version == "11":
            env_config['channels'].insert(1, 'nvidia')
            if minor_version in ["7", "8"]:
                env_config['dependencies'].append(f'pytorch-cuda={major_version}.{minor_version}')
            else:
                env_config['dependencies'].append('cpuonly')
                print_status(f"CUDA {cuda_version} 不支持，使用 CPU 版本", "WARNING")
        else:
            env_config['dependencies'].append('cpuonly')
            print_status(f"CUDA {cuda_version} 不支持，使用 CPU 版本", "WARNING")
    else:
        # 使用 CPU 版本
        env_config['dependencies'].append('cpuonly')
        print_status("使用 CPU 版本的 PyTorch", "INFO")
    
    # 添加其他依赖
    env_config['dependencies'].extend([
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
                'seaborn>=0.11.0'
            ]
        }
    ])
    
    # 保存配置文件
    with open('environment.yml', 'w', encoding='utf-8') as f:
        yaml.dump(env_config, f, default_flow_style=False, allow_unicode=True)
    
    print_status("环境配置文件已创建", "SUCCESS")


def setup_environment():
    """设置 conda 环境"""
    print_status("设置 conda 环境...", "STEP")
    
    # 检查环境是否已存在
    result = run_command("conda env list", check=False)
    if result and "gdmnet" in result.stdout:
        print_status("环境 'gdmnet' 已存在，将删除并重新创建", "WARNING")
        run_command("conda env remove -n gdmnet -y")
    
    # 创建环境
    print_status("创建 conda 环境（这可能需要几分钟）...", "INFO")
    result = run_command("conda env create -f environment.yml")
    
    if result and result.returncode == 0:
        print_status("环境创建成功", "SUCCESS")
        return True
    else:
        print_status("环境创建失败", "ERROR")
        return False


def install_package():
    """安装 GDM-Net 包"""
    print_status("安装 GDM-Net 包...", "STEP")
    
    # 激活环境并安装包
    if platform.system() == "Windows":
        activate_cmd = "conda activate gdmnet && pip install -e ."
        result = run_command(f'cmd /c "{activate_cmd}"', check=False)
    else:
        activate_cmd = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate gdmnet && pip install -e ."
        result = run_command(f'bash -c "{activate_cmd}"', check=False)
    
    if result and result.returncode == 0:
        print_status("包安装成功", "SUCCESS")
        return True
    else:
        print_status("包安装失败，尝试备用方法...", "WARNING")
        
        # 备用方法：直接使用 pip
        try:
            import subprocess
            env = os.environ.copy()
            env['CONDA_DEFAULT_ENV'] = 'gdmnet'
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "."
            ], env=env, check=True, capture_output=True, text=True)
            
            print_status("包安装成功（备用方法）", "SUCCESS")
            return True
        except:
            print_status("包安装失败", "ERROR")
            return False


def test_installation():
    """测试安装"""
    print_status("测试安装...", "STEP")
    
    try:
        # 测试导入
        sys.path.insert(0, os.getcwd())
        
        # 测试基本导入
        import torch
        print_status(f"PyTorch 版本: {torch.__version__}", "INFO")
        print_status(f"CUDA 可用: {torch.cuda.is_available()}", "INFO")
        
        # 测试 GDM-Net 导入
        from gdmnet import GDMNet
        print_status("GDM-Net 导入成功", "SUCCESS")
        
        # 创建模型测试
        model = GDMNet(
            bert_model_name='bert-base-uncased',
            hidden_size=768,
            num_entities=5,
            num_relations=10,
            num_classes=3
        )
        print_status("模型创建成功", "SUCCESS")
        
        return True
        
    except Exception as e:
        print_status(f"安装测试失败: {str(e)}", "ERROR")
        return False


def create_data():
    """创建训练数据"""
    print_status("创建训练数据...", "STEP")
    
    # 创建必要目录
    for dir_name in ['data', 'checkpoints', 'logs']:
        os.makedirs(dir_name, exist_ok=True)
    
    try:
        # 运行数据创建脚本
        sys.path.insert(0, os.path.join(os.getcwd(), 'train'))
        from dataset import create_synthetic_dataset
        
        create_synthetic_dataset('data/train.json', num_samples=800)
        create_synthetic_dataset('data/val.json', num_samples=100)
        create_synthetic_dataset('data/test.json', num_samples=100)
        
        print_status("训练数据创建成功", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"数据创建失败: {str(e)}", "ERROR")
        return False


def start_training():
    """开始训练"""
    print_status("开始训练...", "STEP")
    
    try:
        # 导入训练模块
        sys.path.insert(0, os.path.join(os.getcwd(), 'train'))
        from train import train_model, load_config
        
        # 加载配置
        config = load_config('config/model_config.yaml')
        
        # 修改配置以适应当前环境
        config['training']['max_epochs'] = 2  # 减少训练轮数用于测试
        config['training']['batch_size'] = 4   # 减少批次大小
        
        print_status("开始训练（测试模式：2个epoch）...", "INFO")
        
        # 开始训练
        model = train_model(config)
        
        print_status("训练完成", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"训练失败: {str(e)}", "ERROR")
        return False


def main():
    """主函数"""
    print_status("GDM-Net 自动化设置和测试", "STEP")
    print("=" * 50)
    
    # 检查系统要求
    if not check_conda():
        return False
    
    # 检查 CUDA
    has_cuda, cuda_version = check_cuda()
    
    # 创建环境配置
    create_environment_yml(has_cuda, cuda_version)
    
    # 设置环境
    if not setup_environment():
        return False
    
    # 安装包
    if not install_package():
        return False
    
    # 测试安装
    if not test_installation():
        return False
    
    # 创建数据
    if not create_data():
        return False
    
    # 开始训练
    if not start_training():
        print_status("训练失败，但环境设置成功", "WARNING")
        print_status("您可以手动运行训练命令", "INFO")
    
    # 显示后续步骤
    print("\n" + "=" * 50)
    print_status("设置完成！", "SUCCESS")
    print("\n后续操作:")
    print("1. 激活环境: conda activate gdmnet")
    print("2. 查看日志: tensorboard --logdir logs/")
    print("3. 运行示例: python examples/example_usage.py")
    print("4. 手动训练: python train/train.py --config config/model_config.yaml --mode train")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("\n用户中断操作", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"未预期的错误: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
