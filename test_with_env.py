"""
在正确环境中测试 GDM-Net 安装的脚本
"""

import subprocess
import sys
import os


def run_in_conda_env(command, env_name="gdmnet"):
    """在指定的 conda 环境中运行命令"""
    if os.name == 'nt':  # Windows
        # 使用 PowerShell 在 Windows 上运行
        full_command = f'powershell -Command "conda activate {env_name}; {command}"'
    else:  # Linux/Mac
        full_command = f'bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && {command}"'
    
    try:
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result
    except subprocess.TimeoutExpired:
        print("命令执行超时")
        return None


def test_pytorch():
    """测试 PyTorch 安装"""
    print("🧪 测试 PyTorch...")
    
    command = "python -c \"import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())\""
    result = run_in_conda_env(command)
    
    if result and result.returncode == 0:
        print("✅ PyTorch 测试通过")
        print(result.stdout.strip())
        return True
    else:
        print("❌ PyTorch 测试失败")
        if result:
            print("错误:", result.stderr)
        return False


def test_gdmnet_import():
    """测试 GDM-Net 导入"""
    print("\n🧪 测试 GDM-Net 导入...")
    
    command = "python -c \"from gdmnet import GDMNet; print('GDM-Net 导入成功')\""
    result = run_in_conda_env(command)
    
    if result and result.returncode == 0:
        print("✅ GDM-Net 导入测试通过")
        return True
    else:
        print("❌ GDM-Net 导入测试失败")
        if result:
            print("错误:", result.stderr)
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n🧪 测试模型创建...")
    
    test_script = '''
import torch
from gdmnet import GDMNet

try:
    model = GDMNet(
        bert_model_name="bert-base-uncased",
        hidden_size=768,
        num_entities=5,
        num_relations=10,
        num_classes=3
    )
    print("模型创建成功")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"模型创建失败: {e}")
    import traceback
    traceback.print_exc()
'''
    
    # 将测试脚本写入临时文件
    with open('temp_test.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    command = "python temp_test.py"
    result = run_in_conda_env(command)
    
    # 清理临时文件
    try:
        os.remove('temp_test.py')
    except:
        pass
    
    if result and result.returncode == 0 and "模型创建成功" in result.stdout:
        print("✅ 模型创建测试通过")
        print(result.stdout.strip())
        return True
    else:
        print("❌ 模型创建测试失败")
        if result:
            print("输出:", result.stdout)
            print("错误:", result.stderr)
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n🧪 测试前向传播...")
    
    test_script = '''
import torch
from gdmnet import GDMNet

try:
    model = GDMNet(
        bert_model_name="bert-base-uncased",
        hidden_size=768,
        num_entities=5,
        num_relations=10,
        num_classes=3
    )
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    query = torch.randn(batch_size, 768)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query=query,
            return_intermediate=True
        )
    
    print("前向传播成功")
    print(f"输出 logits 形状: {outputs['logits'].shape}")
    print(f"实体 logits 形状: {outputs['entity_logits'].shape}")
    print(f"关系 logits 形状: {outputs['relation_logits'].shape}")
    
except Exception as e:
    print(f"前向传播失败: {e}")
    import traceback
    traceback.print_exc()
'''
    
    # 将测试脚本写入临时文件
    with open('temp_forward_test.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    command = "python temp_forward_test.py"
    result = run_in_conda_env(command)
    
    # 清理临时文件
    try:
        os.remove('temp_forward_test.py')
    except:
        pass
    
    if result and result.returncode == 0 and "前向传播成功" in result.stdout:
        print("✅ 前向传播测试通过")
        print(result.stdout.strip())
        return True
    else:
        print("❌ 前向传播测试失败")
        if result:
            print("输出:", result.stdout)
            print("错误:", result.stderr)
        return False


def install_missing_packages():
    """安装缺失的包"""
    print("\n🔧 检查并安装缺失的包...")
    
    # 安装 GDM-Net 包
    command = "pip install -e ."
    result = run_in_conda_env(command)
    
    if result and result.returncode == 0:
        print("✅ GDM-Net 包安装/更新成功")
        return True
    else:
        print("❌ GDM-Net 包安装失败")
        if result:
            print("错误:", result.stderr)
        return False


def main():
    """主测试函数"""
    print("🧠 GDM-Net 环境测试（在 gdmnet 环境中）")
    print("=" * 50)
    
    # 检查环境是否存在
    result = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    if "gdmnet" not in result.stdout:
        print("❌ gdmnet 环境不存在，请先创建环境")
        print("运行: conda env create -f environment.yml")
        return False
    
    tests = [
        ("PyTorch", test_pytorch),
        ("安装包", install_missing_packages),
        ("GDM-Net 导入", test_gdmnet_import),
        ("模型创建", test_model_creation),
        ("前向传播", test_forward_pass)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！GDM-Net 已准备就绪。")
        print("\n后续操作:")
        print("1. 激活环境: conda activate gdmnet")
        print("2. 创建数据: python train/dataset.py")
        print("3. 开始训练: python train/train.py --config config/model_config.yaml --mode train")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
