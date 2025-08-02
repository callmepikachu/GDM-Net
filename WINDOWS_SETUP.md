# Windows 系统 GDM-Net 设置指南

由于您遇到了批处理文件的编码问题，这里提供几种解决方案。

## 🚀 解决方案

### 方案一：使用 PowerShell（推荐）

```powershell
# 在 PowerShell 中运行（以管理员身份）
powershell -ExecutionPolicy Bypass -File setup_and_train.ps1
```

### 方案二：使用修复后的批处理文件

```cmd
# 运行新的批处理文件
setup_windows.bat
```

### 方案三：手动步骤（最可靠）

#### 1. 检查 conda 安装

```cmd
conda --version
```

如果提示 `conda` 不是内部或外部命令，请：
- 重新启动命令提示符
- 或者使用 Anaconda Prompt
- 或者添加 conda 到 PATH 环境变量

#### 2. 创建 conda 环境

```cmd
conda env create -f environment.yml
```

#### 3. 激活环境

```cmd
conda activate gdmnet
```

#### 4. 安装 GDM-Net 包

```cmd
pip install -e .
```

#### 5. 测试安装

```cmd
python test_installation.py
```

#### 6. 创建必要目录

```cmd
mkdir data
mkdir checkpoints
mkdir logs
```

#### 7. 生成训练数据

```cmd
python train/dataset.py
```

#### 8. 开始训练

```cmd
python train/train.py --config config/model_config.yaml --mode train
```

## 🔧 故障排除

### 问题1：conda 命令不识别

**解决方案：**
1. 使用 "Anaconda Prompt" 而不是普通的命令提示符
2. 或者重新安装 Anaconda/Miniconda 并确保添加到 PATH

### 问题2：编码问题（中文乱码）

**解决方案：**
1. 使用 PowerShell 而不是 cmd
2. 或者在 cmd 中先运行：`chcp 65001`
3. 使用英文版本的脚本

### 问题3：权限问题

**解决方案：**
1. 以管理员身份运行命令提示符或 PowerShell
2. 或者使用 `--user` 参数安装包：`pip install -e . --user`

### 问题4：网络问题

**解决方案：**
1. 使用国内镜像源：
   ```cmd
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

## 📋 完整的手动安装命令

如果所有自动化脚本都失败，请按顺序执行以下命令：

```cmd
REM 1. 检查 conda
conda --version

REM 2. 创建环境
conda env create -f environment.yml

REM 3. 激活环境
conda activate gdmnet

REM 4. 安装包
pip install -e .

REM 5. 测试
python test_installation.py

REM 6. 创建目录
mkdir data
mkdir checkpoints  
mkdir logs

REM 7. 生成数据
python train/dataset.py

REM 8. 训练
python train/train.py --config config/model_config.yaml --mode train
```

## 🎯 验证安装

运行以下命令验证安装是否成功：

```cmd
conda activate gdmnet
python -c "import gdmnet; print('GDM-Net imported successfully')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 监控训练

训练开始后，在新的命令提示符窗口中运行：

```cmd
conda activate gdmnet
tensorboard --logdir logs
```

然后在浏览器中打开：http://localhost:6006

## 💡 Windows 特定提示

1. **使用 Anaconda Prompt**：这是最可靠的方式，因为它已经配置好了所有环境变量

2. **避免中文路径**：确保项目路径中没有中文字符

3. **防火墙设置**：如果遇到网络问题，可能需要配置防火墙允许 Python 访问网络

4. **长路径支持**：如果遇到路径过长的问题，可以启用 Windows 长路径支持

## 🆘 如果仍有问题

1. 检查 Python 版本：`python --version`（应该是 3.8-3.10）
2. 检查 pip 版本：`pip --version`
3. 清理 pip 缓存：`pip cache purge`
4. 重新创建环境：
   ```cmd
   conda env remove -n gdmnet
   conda env create -f environment.yml
   ```

## 📞 获取帮助

如果问题仍然存在，请提供以下信息：
- Windows 版本
- Python 版本
- Conda 版本
- 完整的错误信息
- 使用的命令
