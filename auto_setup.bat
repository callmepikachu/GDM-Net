@echo off
REM 自动化 GDM-Net 设置脚本

echo GDM-Net 自动化设置
echo ==================

REM 检查 Python
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python first
    pause
    exit /b 1
)

echo 运行自动化设置脚本...
python auto_setup.py

if %errorlevel% equ 0 (
    echo.
    echo 设置完成！
    echo.
    echo 后续操作:
    echo 1. conda activate gdmnet
    echo 2. tensorboard --logdir logs/
    echo 3. python examples/example_usage.py
) else (
    echo.
    echo 设置过程中遇到问题，请查看上面的错误信息
)

echo.
pause
