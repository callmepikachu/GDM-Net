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
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
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
