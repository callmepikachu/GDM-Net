"""
Document Encoder Module

This module implements the DocumentEncoder class that uses BERT to encode input documents.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Optional, Tuple


class DocumentEncoder(nn.Module):
    """
    Document encoder using BERT for encoding input documents.
    
    Args:
        model_name (str): Name of the pre-trained BERT model
        hidden_size (int): Hidden size of the model
        dropout_rate (float): Dropout rate for regularization
        freeze_bert (bool): Whether to freeze BERT parameters
    """
    
    def __init__(
        self, 
        model_name: str = 'bert-base-uncased',
        hidden_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        # Load pre-trained BERT model (支持国内镜像)
        import os

        # 设置国内镜像
        mirror_urls = [
            "https://hf-mirror.com",  # HuggingFace国内镜像
            "https://huggingface.co"  # 原始地址作为备选
        ]

        self.bert = None
        for mirror_url in mirror_urls:
            try:
                # 设置镜像环境变量
                os.environ['HF_ENDPOINT'] = mirror_url
                print(f"🔄 尝试从镜像加载BERT: {mirror_url}")

                self.bert = BertModel.from_pretrained(model_name)
                print(f"✅ 成功从镜像加载BERT模型: {model_name} (镜像: {mirror_url})")
                break

            except Exception as e:
                print(f"⚠️ 镜像 {mirror_url} 连接失败: {str(e)[:100]}...")
                continue

        # 如果所有镜像都失败，尝试离线模式
        if self.bert is None:
            try:
                print("🔄 尝试离线模式...")
                self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
                print(f"✅ 离线模式加载BERT模型成功: {model_name}")
            except Exception as e:
                print(f"❌ 离线模式也失败，使用随机初始化: {str(e)[:100]}...")
                # 使用随机初始化的BERT配置
                from transformers import BertConfig
                config = BertConfig(
                    vocab_size=30522,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    max_position_embeddings=512,
                    type_vocab_size=2
                )
                self.bert = BertModel(config)
                print("⚠️ 使用随机初始化的BERT模型，性能可能受影响")
        
        # Get actual hidden size from BERT config
        self.hidden_size = self.bert.config.hidden_size
        if hidden_size and hidden_size != self.hidden_size:
            # Add projection layer if different hidden size is required
            self.projection = nn.Linear(self.hidden_size, hidden_size)
            self.hidden_size = hidden_size
        else:
            self.projection = None
            
        self.dropout = nn.Dropout(dropout_rate)
        
        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the document encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            
        Returns:
            sequence_output: Encoded sequence [batch_size, seq_len, hidden_size]
            pooled_output: Pooled representation [batch_size, hidden_size]
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state  # [B, seq_len, H]
        pooled_output = outputs.pooler_output  # [B, H]
        
        # Apply projection if needed
        if self.projection is not None:
            sequence_output = self.projection(sequence_output)
            pooled_output = self.projection(pooled_output)
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        return sequence_output, pooled_output
    
    def get_hidden_size(self) -> int:
        """Get the hidden size of the encoder."""
        return self.hidden_size
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings for new vocabulary size."""
        self.bert.resize_token_embeddings(new_num_tokens)
