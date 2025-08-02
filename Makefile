# Makefile for GDM-Net project

.PHONY: help env-create env-setup install install-dev test train eval clean data setup quick-start

# Default target
help:
	@echo "GDM-Net Project Commands:"
	@echo "  env-create   - Create conda environment"
	@echo "  env-setup    - Complete environment setup and training"
	@echo "  quick-start  - Quick setup and start training"
	@echo "  install      - Install the package and dependencies"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  test         - Run installation tests"
	@echo "  setup        - Setup project (install + create data)"
	@echo "  data         - Create synthetic datasets"
	@echo "  train        - Train the model with default config"
	@echo "  eval         - Evaluate a trained model"
	@echo "  clean        - Clean generated files"
	@echo "  help         - Show this help message"

# Create conda environment
env-create:
	@echo "Creating conda environment..."
	conda env create -f environment.yml
	@echo "Environment created! Activate with: conda activate gdmnet"

# Complete environment setup and training
env-setup:
	@echo "Running complete setup and training..."
	bash setup_and_train.sh

# Quick start (simplified setup)
quick-start:
	@echo "Running quick start..."
	bash quick_start.sh

# Install package and dependencies (assumes environment is activated)
install:
	pip install -r requirements.txt
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run installation tests
test:
	python test_installation.py

# Setup project (install + create data)
setup: install data
	@echo "Project setup complete!"

# Create synthetic datasets
data:
	python train/dataset.py
	@echo "Synthetic datasets created in data/ directory"

# Train model with default configuration
train:
	python train/train.py --config config/model_config.yaml --mode train --create_synthetic

# Evaluate model (requires MODEL_PATH environment variable)
eval:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Please set MODEL_PATH environment variable"; \
		echo "Example: make eval MODEL_PATH=checkpoints/gdmnet-epoch=05-val_loss=0.25.ckpt"; \
		exit 1; \
	fi
	python train/train.py --config config/model_config.yaml --mode eval --model_path $(MODEL_PATH)

# Run example usage
example:
	python examples/example_usage.py

# Clean generated files
clean:
	rm -rf __pycache__/
	rm -rf gdmnet/__pycache__/
	rm -rf train/__pycache__/
	rm -rf examples/__pycache__/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	rm -rf data/
	rm -rf checkpoints/
	rm -rf logs/
	rm -rf wandb/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Development commands
lint:
	flake8 gdmnet/ train/ examples/
	black --check gdmnet/ train/ examples/

format:
	black gdmnet/ train/ examples/

# Docker commands (if needed)
docker-build:
	docker build -t gdmnet:latest .

docker-run:
	docker run -it --rm -v $(PWD):/workspace gdmnet:latest

# Quick start command
quickstart: setup test example
	@echo ""
	@echo "🎉 GDM-Net is ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Train a model: make train"
	@echo "  2. Evaluate model: make eval MODEL_PATH=path/to/checkpoint"
	@echo "  3. Run examples: make example"
