# GDM-Net Windows PowerShell Setup Script
# Usage: powershell -ExecutionPolicy Bypass -File setup_and_train.ps1

Write-Host "GDM-Net Environment Setup and Training (PowerShell)" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# Function to print colored messages
function Write-Success { param($Message) Write-Host "SUCCESS: $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "ERROR: $Message" -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host "WARNING: $Message" -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host "INFO: $Message" -ForegroundColor Blue }

# Check if conda is installed
Write-Info "Checking for conda installation..."
try {
    $condaVersion = conda --version 2>$null
    if ($condaVersion) {
        Write-Success "Conda detected: $condaVersion"
    } else {
        throw "Conda not found"
    }
} catch {
    Write-Error "Conda not found. Please install Anaconda or Miniconda first."
    Write-Host "Download: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if environment already exists
Write-Info "Checking for existing gdmnet environment..."
$envExists = conda env list | Select-String "gdmnet"
if ($envExists) {
    Write-Warning "Environment 'gdmnet' already exists"
    $choice = Read-Host "Delete and recreate? (y/N)"
    if ($choice -eq "y" -or $choice -eq "Y") {
        Write-Info "Removing existing environment..."
        conda env remove -n gdmnet -y
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to remove existing environment"
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        Write-Info "Using existing environment..."
        goto ActivateEnv
    }
}

# Create conda environment
Write-Info "Creating conda environment from environment.yml..."
conda env create -f environment.yml
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment creation failed"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Success "Environment created successfully"

:ActivateEnv
# Activate environment and install package
Write-Info "Activating environment and installing GDM-Net package..."

# Initialize conda for PowerShell
$condaPath = (Get-Command conda).Source
$condaRoot = Split-Path (Split-Path $condaPath)
& "$condaRoot\Scripts\activate.bat" gdmnet

# Install package
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Package installation failed"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Success "Package installed successfully"

# Test installation
Write-Info "Testing installation..."
python test_installation.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Installation test failed"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Success "Installation test passed"

# Create directories
Write-Info "Creating necessary directories..."
if (!(Test-Path "data")) { New-Item -ItemType Directory -Path "data" }
if (!(Test-Path "checkpoints")) { New-Item -ItemType Directory -Path "checkpoints" }
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" }

# Create training data
Write-Info "Creating synthetic training data..."
python train/dataset.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Data creation failed"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Success "Training data created"

# Start training
Write-Info "Starting model training..."
python train/train.py --config config/model_config.yaml --mode train
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Success "Setup and training completed successfully!"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Activate environment: conda activate gdmnet" -ForegroundColor White
Write-Host "2. View training logs: tensorboard --logdir logs/" -ForegroundColor White
Write-Host "3. Run examples: python examples/example_usage.py" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
