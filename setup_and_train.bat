@echo off
chcp 65001 >nul
REM GDM-Net Windows Setup and Training Script

echo GDM-Net Environment Setup and Training (Windows)
echo ================================================

REM Check conda
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: conda not found. Please install Anaconda or Miniconda first.
    echo Download: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo SUCCESS: conda detected

REM Check if environment exists
conda env list | findstr "gdmnet" >nul
if %errorlevel% equ 0 (
    echo WARNING: Environment 'gdmnet' already exists
    set /p choice="Delete and recreate? (y/N): "
    if /i "%choice%"=="y" (
        echo Removing existing environment...
        conda env remove -n gdmnet -y
    ) else (
        echo Using existing environment...
        goto activate_env
    )
)

REM Create environment
echo Creating conda environment...
conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo ERROR: Environment creation failed
    pause
    exit /b 1
)

:activate_env
REM Activate environment and install package
echo Activating environment and installing package...
call conda activate gdmnet
pip install -e .

REM Test installation
echo Testing installation...
python test_installation.py
if %errorlevel% neq 0 (
    echo ERROR: Installation test failed
    pause
    exit /b 1
)

REM Create directories and data
echo Creating training data...
if not exist "data" mkdir data
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs

python train/dataset.py

REM Start training
echo Starting training...
python train/train.py --config config/model_config.yaml --mode train

echo.
echo SUCCESS: Setup and training completed!
echo.
echo Next steps:
echo 1. Activate environment: conda activate gdmnet
echo 2. View logs: tensorboard --logdir logs/
echo 3. Run examples: python examples/example_usage.py
echo.
pause
