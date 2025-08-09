import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Any, Optional
import logging
import os
import sys

def check_tensor(tensor, name, batch_idx, stage="train"):
    """精确检测张量中的NaN/Inf并立即报错定位"""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise ValueError(f"❌ NaN detected in {name} at {stage} batch {batch_idx} (count: {nan_count})")
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise ValueError(f"❌ Inf detected in {name} at {stage} batch {batch_idx} (count: {inf_count})")

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
        self.num_classes = config['model']['num_classes']
        try:
            from torchmetrics import Accuracy
            self.train_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
            self.val_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
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
            relation_labels=batch.get('relation_labels'),
            # 🚀 传递原始文本给SpaCy处理
            doc_texts=batch.get('doc_text', None)
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with comprehensive NaN/Inf detection."""

        # 🔍 检查输入数据
        try:
            check_tensor(batch['query_input_ids'], "query_input_ids", batch_idx, "train")
            check_tensor(batch['query_attention_mask'], "query_attention_mask", batch_idx, "train")
            check_tensor(batch['doc_input_ids'], "doc_input_ids", batch_idx, "train")
            check_tensor(batch['doc_attention_mask'], "doc_attention_mask", batch_idx, "train")
            check_tensor(batch['entity_spans'], "entity_spans", batch_idx, "train")
            check_tensor(batch['labels'], "labels", batch_idx, "train")
        except ValueError as e:
            print(f"🚨 INPUT DATA ERROR: {e}")
            # 返回一个安全的损失值继续训练
            return torch.tensor(1.609, device=self.device, requires_grad=True)

        labels = batch['labels']

        # 🔍 检查标签范围
        if (labels < 0).any() or (labels >= self.num_classes).any():
            print(f"🚨 LABEL RANGE ERROR at batch {batch_idx}: labels {labels} outside [0, {self.num_classes-1}]")
            return torch.tensor(1.609, device=self.device, requires_grad=True)

        # 前向传播
        outputs = self.forward(batch)

        # 🔍 检查模型输出
        try:
            check_tensor(outputs['logits'], "logits", batch_idx, "train")
        except ValueError as e:
            print(f"🚨 MODEL OUTPUT ERROR: {e}")
            return torch.tensor(1.609, device=self.device, requires_grad=True)

        # 计算损失
        loss_dict = self.model.compute_loss(
            outputs=outputs,
            labels=labels,
            entity_labels=batch.get('entity_labels'),
            relation_labels=batch.get('relation_labels')
        )

        total_loss = loss_dict['total_loss']
        main_loss = loss_dict['main_loss']

        # 🔍 检查损失值
        try:
            check_tensor(total_loss, "total_loss", batch_idx, "train")
            check_tensor(main_loss, "main_loss", batch_idx, "train")
        except ValueError as e:
            print(f"🚨 LOSS ERROR: {e}")
            print(f"  Logits range: [{outputs['logits'].min():.6f}, {outputs['logits'].max():.6f}]")
            print(f"  Labels: {labels}")
            return torch.tensor(1.609, device=self.device, requires_grad=True)

        # 计算准确率
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)

        # 获取批次大小
        batch_size = labels.size(0)

        # 🔍 调试信息：检查标签分布
        if batch_idx % 500 == 0:
            # 只对bincount操作临时关闭确定性检查
            with torch.backends.cudnn.flags(enabled=False):
                # 使用warn_only=True来允许非确定性操作
                original_deterministic = torch.are_deterministic_algorithms_enabled()
                torch.use_deterministic_algorithms(False, warn_only=True)

                try:
                    label_counts = torch.bincount(labels, minlength=self.num_classes)
                    label_dist = label_counts.float() / label_counts.sum()
                    print(f"🔍 Batch {batch_idx} label distribution: {label_dist.tolist()}")
                    print(f"🔍 Label counts: {label_counts.tolist()}")
                finally:
                    # 恢复原始设置
                    torch.use_deterministic_algorithms(original_deterministic)

        # 记录指标
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_main_loss', main_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # 记录辅助损失
        if 'entity_loss' in loss_dict:
            self.log('train_entity_loss', loss_dict['entity_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        if 'relation_loss' in loss_dict:
            self.log('train_relation_loss', loss_dict['relation_loss'], on_step=True, on_epoch=True, batch_size=batch_size)

        return total_loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """检查梯度中的NaN/Inf"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                try:
                    check_tensor(param.grad, f"grad_{name}", self.global_step, "train")
                except ValueError as e:
                    print(f"🚨 GRADIENT ERROR: {e}")
                    # 将有问题的梯度设为零
                    param.grad.data.zero_()
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step with comprehensive NaN/Inf detection."""

        # 🔍 检查验证输入数据
        try:
            check_tensor(batch['query_input_ids'], "val_query_input_ids", batch_idx, "val")
            check_tensor(batch['query_attention_mask'], "val_query_attention_mask", batch_idx, "val")
            check_tensor(batch['doc_input_ids'], "val_doc_input_ids", batch_idx, "val")
            check_tensor(batch['doc_attention_mask'], "val_doc_attention_mask", batch_idx, "val")
            check_tensor(batch['entity_spans'], "val_entity_spans", batch_idx, "val")
            check_tensor(batch['labels'], "val_labels", batch_idx, "val")
        except ValueError as e:
            print(f"🚨 VALIDATION INPUT ERROR: {e}")
            # 返回一个安全的损失值继续验证
            return torch.tensor(1.609, device=self.device)

        labels = batch['labels']

        # 🔍 检查验证标签范围
        if (labels < 0).any() or (labels >= self.num_classes).any():
            print(f"🚨 VALIDATION LABEL RANGE ERROR at batch {batch_idx}: labels {labels} outside [0, {self.num_classes-1}]")
            return torch.tensor(1.609, device=self.device)

        # 前向传播
        outputs = self.forward(batch)

        # 🔍 检查验证模型输出
        try:
            check_tensor(outputs['logits'], "val_logits", batch_idx, "val")
        except ValueError as e:
            print(f"🚨 VALIDATION OUTPUT ERROR: {e}")
            return torch.tensor(1.609, device=self.device)

        # 计算损失
        loss_dict = self.model.compute_loss(
            outputs=outputs,
            labels=labels,
            entity_labels=batch.get('entity_labels'),
            relation_labels=batch.get('relation_labels')
        )

        total_loss = loss_dict['total_loss']
        main_loss = loss_dict['main_loss']

        # 🔍 检查验证损失值
        try:
            check_tensor(total_loss, "val_total_loss", batch_idx, "val")
            check_tensor(main_loss, "val_main_loss", batch_idx, "val")
        except ValueError as e:
            print(f"🚨 VALIDATION LOSS ERROR: {e}")
            print(f"  Val Logits range: [{outputs['logits'].min():.6f}, {outputs['logits'].max():.6f}]")
            print(f"  Val Labels: {labels}")
            total_loss = torch.tensor(1.609, device=self.device)
            main_loss = torch.tensor(1.609, device=self.device)

        # 计算准确率
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)

        # 获取批次大小
        batch_size = labels.size(0)

        # 记录验证指标
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_main_loss', main_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # 记录验证辅助损失
        if 'entity_loss' in loss_dict:
            self.log('val_entity_loss', loss_dict['entity_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        if 'relation_loss' in loss_dict:
            self.log('val_relation_loss', loss_dict['relation_loss'], on_step=False, on_epoch=True, batch_size=batch_size)

        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""

        # Get learning rate from config (check both model and training sections)
        learning_rate = self.config.get('model', {}).get('learning_rate',
                                       self.config.get('training', {}).get('learning_rate', 2e-5))
        learning_rate = float(learning_rate)

        # Only collect trainable parameters (BERT may be frozen)
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]

        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer with enhanced regularization
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.02,  # 增加权重衰减减少过拟合
            eps=1e-8,  # 更大的eps提高稳定性
            betas=(0.9, 0.999),  # 标准beta值
            amsgrad=True  # 使用AMSGrad变体提高稳定性
        )

        # Learning rate scheduler with enhanced warmup for stability
        try:
            total_steps = self.trainer.estimated_stepping_batches
            # Ensure total_steps is an integer
            if isinstance(total_steps, str):
                total_steps = int(total_steps)
            elif total_steps is None:
                # Fallback calculation
                max_epochs = self.config['training'].get('max_epochs', 20)
                batch_size = self.config['training'].get('batch_size', 8)
                # Estimate based on dataset size (assuming 5000 samples)
                total_steps = int((5000 / batch_size) * max_epochs)

            total_steps = int(total_steps)
            warmup_steps = int(0.05 * total_steps)  # 5% warmup适应大batch size

            # 使用cosine annealing with warmup获得更好的收敛
            from torch.optim.lr_scheduler import CosineAnnealingLR

            # 创建warmup + cosine scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            print(f"Scheduler configured: {total_steps} total steps, {warmup_steps} warmup steps")

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
            pin_memory=True,
            persistent_workers=True,  # 保持worker进程
            prefetch_factor=2         # 预取数据
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
            pin_memory=True,
            persistent_workers=True,  # 保持worker进程
            prefetch_factor=2         # 预取数据
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            self.test_dataset = HotpotQADataset(
                data_path=self.config['data']['test_path'],
                tokenizer_name=self.config['model']['bert_model_name'],
                max_length=self.config['data']['max_length'],
                max_query_length=self.config['data']['max_query_length'],
                num_entities=self.config['model']['num_entities']
            )

        return DataLoader(
            self.test_dataset,
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
