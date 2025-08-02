@echo off
REM Simple Windows setup script for GDM-Net
REM Avoid special characters to prevent encoding issues

echo GDM-Net Windows Setup
echo ====================

REM Check conda
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: conda not found
    echo Please install Anaconda or Miniconda first
    echo Download: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Found conda installation

REM Check if environment exists
conda env list | findstr "gdmnet" >nul
if %errorlevel% equ 0 (
    echo Environment gdmnet already exists
    set /p choice="Remove and recreate? (y/N): "
    if /i "%choice%"=="y" (
        echo Removing existing environment...
        conda env remove -n gdmnet -y
    ) else (
        echo Using existing environment
        goto activate_env
    )
)

REM Create environment
echo Creating conda environment...
conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo ERROR: Failed to create environment
    pause
    exit /b 1
)

:activate_env
echo Activating environment...
call conda activate gdmnet

echo Installing GDM-Net package...
pip install -e .
if %errorlevel% neq 0 (
    echo ERROR: Package installation failed
    pause
    exit /b 1
)

echo Testing installation...
python test_installation.py
if %errorlevel% neq 0 (
    echo ERROR: Installation test failed
    pause
    exit /b 1
)

echo Creating directories...
if not exist "data" mkdir data
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs

echo Creating training data...
python train/dataset.py
if %errorlevel% neq 0 (
    echo ERROR: Data creation failed
    pause
    exit /b 1
)

echo Starting training...
python train/train.py --config config/model_config.yaml --mode train

echo.
echo Setup and training completed!
echo.
echo Next steps:
echo 1. conda activate gdmnet
echo 2. tensorboard --logdir logs/
echo 3. python examples/example_usage.py
echo.
pause
