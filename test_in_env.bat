@echo off
REM 在 gdmnet 环境中运行测试

echo 在 gdmnet 环境中测试 GDM-Net...
echo ================================

REM 激活环境并运行测试
call conda activate gdmnet
if %errorlevel% neq 0 (
    echo ERROR: 无法激活 gdmnet 环境
    echo 请确保环境已创建: conda env create -f environment.yml
    pause
    exit /b 1
)

echo 环境已激活，运行测试...
python simple_test.py

echo.
echo 测试完成！
pause
