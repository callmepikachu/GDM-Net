"""
Google Colab GDM-Net 故障排除脚本
解决常见的导入和依赖问题
"""

import subprocess
import sys
import os
import importlib


def fix_pytorch_cuda_mismatch():
    """修复PyTorch CUDA版本不匹配"""
    print("🔧 修复PyTorch CUDA版本不匹配...")
    
    try:
        # 卸载现有版本
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                      capture_output=True, check=True)
        
        # 安装匹配版本
        subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchvision==0.16.0", 
                       "torchaudio==2.1.0", "--index-url", "https://download.pytorch.org/whl/cu118"], 
                      capture_output=True, check=True)
        
        print("✅ PyTorch重新安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch安装失败: {e}")
        return False


def fix_transformers_import():
    """修复transformers导入问题"""
    print("🔧 修复transformers导入问题...")
    
    try:
        # 重新安装transformers
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "transformers>=4.20.0"], 
                      capture_output=True, check=True)
        
        # 测试导入
        from transformers import BertModel, BertTokenizer
        print("✅ transformers重新安装成功")
        return True
    except Exception as e:
        print(f"❌ transformers安装失败: {e}")
        return False


def check_gdmnet_files():
    """检查GDM-Net文件完整性"""
    print("📁 检查GDM-Net文件...")
    
    required_files = {
        'gdmnet/__init__.py': 'GDM-Net包初始化文件',
        'gdmnet/model.py': 'GDM-Net主模型文件',
        'gdmnet/encoder.py': '文档编码器',
        'gdmnet/extractor.py': '结构提取器',
        'gdmnet/graph_memory.py': '图记忆模块',
        'gdmnet/reasoning.py': '推理模块',
        'train/train.py': '训练脚本',
        'train/dataset.py': '数据集处理'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"✅ {file_path} - {description}")
        else:
            print(f"❌ {file_path} - {description} (缺失)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files


def create_minimal_gdmnet_init():
    """创建最小的GDM-Net __init__.py文件"""
    print("📝 创建最小的GDM-Net初始化文件...")
    
    os.makedirs('gdmnet', exist_ok=True)
    
    init_content = '''"""
GDM-Net: Graph-Augmented Dual Memory Network
"""

try:
    from .model import GDMNet
    __all__ = ['GDMNet']
except ImportError as e:
    print(f"Warning: Could not import GDMNet: {e}")
    # 创建一个占位符类
    class GDMNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("GDM-Net model files are not properly installed")
    __all__ = ['GDMNet']
'''
    
    with open('gdmnet/__init__.py', 'w') as f:
        f.write(init_content)
    
    print("✅ GDM-Net初始化文件创建完成")


def test_imports():
    """测试所有必要的导入"""
    print("🧪 测试导入...")
    
    imports_to_test = [
        ('torch', 'PyTorch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'Transformers'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('yaml', 'PyYAML'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib')
    ]
    
    failed_imports = []
    
    for module_name, display_name in imports_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0, failed_imports


def test_bert_model():
    """测试BERT模型创建"""
    print("🧪 测试BERT模型...")
    
    try:
        from transformers import BertModel, BertTokenizer
        
        # 测试tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("✅ BERT Tokenizer创建成功")
        
        # 测试模型
        model = BertModel.from_pretrained('bert-base-uncased')
        print("✅ BERT Model创建成功")
        
        return True
    except Exception as e:
        print(f"❌ BERT测试失败: {e}")
        return False


def install_missing_packages(missing_packages):
    """安装缺失的包"""
    print(f"📦 安装缺失的包: {missing_packages}")
    
    package_mapping = {
        'torch': 'torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118',
        'torchvision': 'torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118',
        'transformers': 'transformers>=4.20.0',
        'pytorch_lightning': 'pytorch-lightning>=1.7.0',
        'torch_geometric': 'torch-geometric',
        'yaml': 'PyYAML>=6.0',
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.3.0',
        'sklearn': 'scikit-learn>=1.1.0',
        'matplotlib': 'matplotlib>=3.5.0'
    }
    
    for package in missing_packages:
        if package in package_mapping:
            install_cmd = package_mapping[package].split()
            try:
                subprocess.run([sys.executable, "-m", "pip", "install"] + install_cmd, 
                              capture_output=True, check=True)
                print(f"✅ {package} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ {package} 安装失败: {e}")


def comprehensive_fix():
    """综合修复方案"""
    print("🛠️  GDM-Net Colab 综合故障排除")
    print("=" * 50)
    
    # 1. 修复PyTorch CUDA问题
    print("\n1️⃣ 修复PyTorch CUDA版本...")
    pytorch_ok = fix_pytorch_cuda_mismatch()
    
    # 2. 修复transformers问题
    print("\n2️⃣ 修复transformers...")
    transformers_ok = fix_transformers_import()
    
    # 3. 测试导入
    print("\n3️⃣ 测试导入...")
    imports_ok, failed_imports = test_imports()
    
    if not imports_ok:
        print("\n🔧 安装缺失的包...")
        install_missing_packages(failed_imports)
        # 重新测试
        imports_ok, failed_imports = test_imports()
    
    # 4. 检查GDM-Net文件
    print("\n4️⃣ 检查GDM-Net文件...")
    files_ok, missing_files = check_gdmnet_files()
    
    if not files_ok:
        print(f"\n⚠️ 缺失文件: {missing_files}")
        print("请上传完整的GDM-Net项目文件")
        create_minimal_gdmnet_init()
    
    # 5. 测试BERT
    print("\n5️⃣ 测试BERT模型...")
    bert_ok = test_bert_model()
    
    # 6. 测试GDM-Net导入
    print("\n6️⃣ 测试GDM-Net导入...")
    try:
        sys.path.append('/content')
        from gdmnet import GDMNet
        print("✅ GDM-Net导入成功")
        gdmnet_ok = True
    except Exception as e:
        print(f"❌ GDM-Net导入失败: {e}")
        gdmnet_ok = False
    
    # 总结
    print("\n" + "=" * 50)
    print("🎯 故障排除总结:")
    print(f"  PyTorch: {'✅' if pytorch_ok else '❌'}")
    print(f"  Transformers: {'✅' if transformers_ok else '❌'}")
    print(f"  导入测试: {'✅' if imports_ok else '❌'}")
    print(f"  文件完整性: {'✅' if files_ok else '❌'}")
    print(f"  BERT测试: {'✅' if bert_ok else '❌'}")
    print(f"  GDM-Net导入: {'✅' if gdmnet_ok else '❌'}")
    
    all_ok = all([pytorch_ok, transformers_ok, imports_ok, files_ok, bert_ok, gdmnet_ok])
    
    if all_ok:
        print("\n🎉 所有问题已修复！可以开始训练了。")
    else:
        print("\n⚠️ 仍有问题需要解决:")
        if not files_ok:
            print("  - 请上传完整的GDM-Net项目文件")
        if not gdmnet_ok:
            print("  - 检查GDM-Net模型文件是否正确")
        if not (pytorch_ok and transformers_ok):
            print("  - 可能需要重启Colab运行时")
    
    return all_ok


if __name__ == "__main__":
    comprehensive_fix()
