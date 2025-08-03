"""
Training Script for GDM-Net

This script provides a complete training pipeline for GDM-Net including
data loading, model training, validation, and evaluation.
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl

# 禁用自动混合精度以避免T4 GPU兼容性问题
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# GPU内存优化
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # 设置内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    # 启用内存映射
    torch.cuda.set_per_process_memory_fraction(0.8)  # 限制使用80%的GPU内存
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb

# Import GDM-Net components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdmnet import GDMNet
from dataset import create_data_loaders, create_synthetic_dataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_callbacks(config: Dict[str, Any]) -> list:
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='gdmnet-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if config['training'].get('early_stopping', True):
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('patience', 5),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(config: Dict[str, Any]) -> Optional[pl.loggers.Logger]:
    """Setup experiment logger."""
    logger_config = config.get('logging', {})
    logger_type = logger_config.get('type', 'tensorboard')
    
    if logger_type == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=logger_config.get('save_dir', 'logs'),
            name=logger_config.get('name', 'gdmnet'),
            version=logger_config.get('version', None)
        )
    elif logger_type == 'wandb':
        logger = WandbLogger(
            project=logger_config.get('project', 'gdmnet'),
            name=logger_config.get('name', 'gdmnet-experiment'),
            save_dir=logger_config.get('save_dir', 'logs')
        )
    else:
        logger = None
    
    return logger


def train_model(config: Dict[str, Any]) -> GDMNet:
    """Train the GDM-Net model."""
    print("Starting GDM-Net training...")
    
    # Set random seeds for reproducibility
    pl.seed_everything(config.get('seed', 42))
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data'].get('test_path'),
        tokenizer_name=config['model']['bert_model_name'],
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length'],
        max_query_length=config['data']['max_query_length'],
        num_workers=config['training'].get('num_workers', 4)
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = GDMNet(**config['model'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    
    # Initialize trainer
    trainer_config = config['training'].copy()
    trainer_config.pop('checkpoint_dir', None)
    trainer_config.pop('patience', None)
    trainer_config.pop('early_stopping', None)
    trainer_config.pop('num_workers', None)
    trainer_config.pop('batch_size', None)
    
    # 强制禁用混合精度以避免T4 GPU兼容性问题
    trainer_config['precision'] = 32

    # 多GPU支持配置
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"🚀 检测到 {num_gpus} 个GPU，启用多GPU训练")
        trainer_config['devices'] = num_gpus
        trainer_config['strategy'] = 'ddp'  # 分布式数据并行
        # 调整批次大小以适应多GPU
        original_batch_size = config['training'].get('batch_size', 1)
        effective_batch_size = original_batch_size * num_gpus
        print(f"📊 多GPU批次大小: 每GPU {original_batch_size} → 总计 {effective_batch_size}")
    else:
        print(f"🔧 使用单GPU训练")
        trainer_config['devices'] = 1

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_config
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test model if test data is available
    if test_loader:
        print("Testing model...")
        trainer.test(model, test_loader)
    
    print("Training completed!")
    return model


def evaluate_model(model_path: str, config: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate a trained model."""
    print(f"Evaluating model from {model_path}...")
    
    # Load model
    model = GDMNet.load_from_checkpoint(model_path)
    model.eval()
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(
        train_path=config['data']['train_path'],  # Needed for type mappings
        val_path=config['data']['val_path'],
        test_path=config['data']['test_path'],
        tokenizer_name=config['model']['bert_model_name'],
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length'],
        max_query_length=config['data']['max_query_length'],
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Initialize trainer for testing
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False
    )
    
    # Run evaluation
    results = trainer.test(model, test_loader)
    
    return results[0] if results else {}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GDM-Net model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode: train or eval')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint for evaluation')
    parser.add_argument('--create_synthetic', action='store_true', help='Create synthetic dataset')
    
    args = parser.parse_args()

    print("Starting GDM-Net training...")

    # 检查GPU设置
    print("🔍 GPU环境检查:")
    print("=" * 40)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✅ 检测到 {num_gpus} 个GPU")

        total_memory = 0
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            total_memory += gpu_memory
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        print(f"📊 总GPU内存: {total_memory:.1f} GB")

        if num_gpus > 1:
            print(f"🚀 支持多GPU训练，当前配置将自动适配")
        else:
            print(f"🔧 单GPU训练模式")
    else:
        print("❌ CUDA不可用，将使用CPU训练")

    # Load configuration
    config = load_config(args.config)
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        print("Creating synthetic datasets...")
        os.makedirs('data', exist_ok=True)
        create_synthetic_dataset('data/train.json', num_samples=800)
        create_synthetic_dataset('data/val.json', num_samples=100)
        create_synthetic_dataset('data/test.json', num_samples=100)
        print("Synthetic datasets created!")
        return
    
    # Check if data files exist
    required_files = [config['data']['train_path'], config['data']['val_path']]
    if config['data'].get('test_path'):
        required_files.append(config['data']['test_path'])
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing data files: {missing_files}")
        print("Use --create_synthetic to create synthetic datasets for testing")
        return
    
    # Create output directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    if 'logging' in config and 'save_dir' in config['logging']:
        os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    if args.mode == 'train':
        # Train model
        model = train_model(config)
        print(f"Training completed. Best model saved in {config['training']['checkpoint_dir']}")
        
    elif args.mode == 'eval':
        # Evaluate model
        if not args.model_path:
            print("Model path is required for evaluation mode")
            return
        
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            return
        
        results = evaluate_model(args.model_path, config)
        print("Evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
