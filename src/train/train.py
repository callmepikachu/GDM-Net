#!/usr/bin/env python3

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from ..utils import load_config, setup_logger, validate_config
from .trainer import GDMNetTrainer


def setup_callbacks(config):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='gdmnet-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if config['training'].get('early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            mode='max',
            patience=config['training'].get('patience', 3),
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    return callbacks


def setup_logger_pl(config):
    """Setup PyTorch Lightning logger."""
    if config['logging']['type'] == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=config['logging']['save_dir'],
            name=config['logging']['name'],
            version=None
        )
    else:
        logger = None
    
    return logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GDM-Net model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test_run',
        action='store_true',
        help='Run a quick test with limited data'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Setup logging
    logger = setup_logger("GDMNet-Training")
    logger.info("Starting GDM-Net training")
    logger.info(f"Configuration: {args.config}")
    
    # Test run modifications
    if args.test_run:
        logger.info("Running in test mode with limited data")
        config['training']['max_epochs'] = 2
        config['training']['val_check_interval'] = 1.0
        config['training']['log_every_n_steps'] = 1
    
    # Set random seed
    pl.seed_everything(config.get('seed', 42))
    
    # Create directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Initialize trainer module
    model = GDMNetTrainer(config)
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    pl_logger = setup_logger_pl(config)
    
    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        val_check_interval=config['training']['val_check_interval'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        callbacks=callbacks,
        logger=pl_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True
    )
    
    # Log model summary
    logger.info("Model architecture:")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Start training
    try:
        trainer.fit(
            model,
            ckpt_path=args.resume_from_checkpoint
        )
        
        logger.info("Training completed successfully")
        
        # Save final model
        final_model_path = os.path.join(
            config['training']['checkpoint_dir'], 
            'final_model.pt'
        )
        torch.save(model.model.state_dict(), final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    # Log best model path
    if hasattr(trainer.checkpoint_callback, 'best_model_path'):
        logger.info(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    
    return trainer, model


if __name__ == "__main__":
    main()
