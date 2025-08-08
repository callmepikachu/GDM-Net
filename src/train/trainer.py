import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Any, Optional
import logging
import os
import sys

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import GDMNet
from src.dataloader import HotpotQADataset, GDMNetDataCollator
from src.utils import setup_logger


class GDMNetTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for GDM-Net."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset: Optional[HotpotQADataset] = None,
        val_dataset: Optional[HotpotQADataset] = None
    ):
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(config)
        
        # Initialize model
        self.model = GDMNet(**config['model'])
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model']['bert_model_name']
        )
        
        # Data collator
        self.data_collator = GDMNetDataCollator(
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Metrics tracking
        try:
            from torchmetrics import Accuracy
            self.train_accuracy = Accuracy(task='multiclass', num_classes=config['model']['num_classes'])
            self.val_accuracy = Accuracy(task='multiclass', num_classes=config['model']['num_classes'])
        except ImportError:
            # Fallback for older versions
            from pytorch_lightning.metrics import Accuracy
            self.train_accuracy = Accuracy()
            self.val_accuracy = Accuracy()
        
        # Logger
        self.logger_instance = setup_logger("GDMNetTrainer")
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(
            query_input_ids=batch['query_input_ids'],
            query_attention_mask=batch['query_attention_mask'],
            doc_input_ids=batch['doc_input_ids'],
            doc_attention_mask=batch['doc_attention_mask'],
            entity_spans=batch['entity_spans'],
            entity_labels=batch.get('entity_labels'),
            relation_labels=batch.get('relation_labels')
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""

        outputs = self.forward(batch)
        labels = batch['labels']

        # Compute loss with auxiliary tasks
        loss_dict = self.model.compute_loss(
            outputs=outputs,
            labels=labels,
            entity_labels=batch.get('entity_labels'),
            relation_labels=batch.get('relation_labels')
        )

        total_loss = loss_dict['total_loss']
        main_loss = loss_dict['main_loss']

        # Compute accuracy
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)

        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_main_loss', main_loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        # Log auxiliary losses if available
        if 'entity_loss' in loss_dict:
            self.log('train_entity_loss', loss_dict['entity_loss'], on_step=True, on_epoch=True)
        if 'relation_loss' in loss_dict:
            self.log('train_relation_loss', loss_dict['relation_loss'], on_step=True, on_epoch=True)

        return total_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""

        outputs = self.forward(batch)
        labels = batch['labels']

        # Compute loss with auxiliary tasks
        loss_dict = self.model.compute_loss(
            outputs=outputs,
            labels=labels,
            entity_labels=batch.get('entity_labels'),
            relation_labels=batch.get('relation_labels')
        )

        total_loss = loss_dict['total_loss']
        main_loss = loss_dict['main_loss']

        # Compute accuracy
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)

        # Log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_main_loss', main_loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # Log auxiliary losses if available
        if 'entity_loss' in loss_dict:
            self.log('val_entity_loss', loss_dict['entity_loss'], on_step=False, on_epoch=True)
        if 'relation_loss' in loss_dict:
            self.log('val_relation_loss', loss_dict['relation_loss'], on_step=False, on_epoch=True)

        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""

        # Separate parameters for different learning rates
        bert_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        # Get learning rate from config (check both model and training sections)
        learning_rate = self.config.get('model', {}).get('learning_rate',
                                       self.config.get('training', {}).get('learning_rate', 2e-5))
        learning_rate = float(learning_rate)

        # Optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': learning_rate},
            {'params': other_params, 'lr': learning_rate * 10}
        ], weight_decay=0.01)

        # Learning rate scheduler with error handling
        try:
            total_steps = self.trainer.estimated_stepping_batches
            # Ensure total_steps is an integer
            if isinstance(total_steps, str):
                total_steps = int(total_steps)
            elif total_steps is None:
                # Fallback calculation
                max_epochs = self.config['training'].get('max_epochs', 10)
                batch_size = self.config['training'].get('batch_size', 16)
                # Estimate based on dataset size (assuming 5000 samples)
                total_steps = int((5000 / batch_size) * max_epochs)

            total_steps = int(total_steps)
            warmup_steps = int(0.1 * total_steps)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }

        except Exception as e:
            print(f"Warning: Failed to create scheduler: {e}. Using optimizer only.")
            return optimizer
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            self.train_dataset = HotpotQADataset(
                data_path=self.config['data']['train_path'],
                tokenizer_name=self.config['model']['bert_model_name'],
                max_length=self.config['data']['max_length'],
                max_query_length=self.config['data']['max_query_length'],
                num_entities=self.config['model']['num_entities']
            )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            collate_fn=self.data_collator,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            self.val_dataset = HotpotQADataset(
                data_path=self.config['data']['val_path'],
                tokenizer_name=self.config['model']['bert_model_name'],
                max_length=self.config['data']['max_length'],
                max_query_length=self.config['data']['max_query_length'],
                num_entities=self.config['model']['num_entities']
            )
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            collate_fn=self.data_collator,
            pin_memory=True
        )
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        train_acc = self.train_accuracy.compute()
        self.logger_instance.info(f"Training accuracy: {train_acc:.4f}")
        self.train_accuracy.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        val_acc = self.val_accuracy.compute()
        self.logger_instance.info(f"Validation accuracy: {val_acc:.4f}")
        self.val_accuracy.reset()
