@echo off
echo 运行 GDM-Net 示例...
echo ==================

powershell -Command "conda activate gdmnet; python run_example.py"

echo.
echo 示例运行完成！
pause
