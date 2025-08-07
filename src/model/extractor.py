"""
Structure Extractor Module

This module implements the StructureExtractor class for extracting entities, 
relations, and other structural information from encoded documents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math


class StructureExtractor(nn.Module):
    """
    Structure extractor for extracting entities and relations from document representations.
    
    Args:
        hidden_size (int): Hidden size of input representations
        num_entities (int): Number of entity types
        num_relations (int): Number of relation types
        dropout_rate (float): Dropout rate for regularization
        use_crf (bool): Whether to use CRF for entity extraction
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_entities: int,
        num_relations: int,
        dropout_rate: float = 0.1,
        use_crf: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.use_crf = use_crf
        
        # Entity extraction layers
        self.entity_classifier = nn.Linear(hidden_size, num_entities)
        
        # Relation extraction layers
        self.relation_head_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.relation_tail_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.relation_classifier = nn.Linear(hidden_size, num_relations)
        
        # Attention mechanism for relation extraction
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        sequence_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the structure extractor.
        
        Args:
            sequence_output: Encoded sequence [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            entity_logits: Entity classification logits [batch_size, seq_len, num_entities]
            relation_logits: Relation classification logits [batch_size, seq_len, seq_len, num_relations]
            extra_outputs: Dictionary containing additional outputs
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # Apply layer normalization and dropout
        sequence_output = self.layer_norm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        # Entity extraction
        entity_logits = self.entity_classifier(sequence_output)  # [B, seq_len, num_entities]
        
        # Relation extraction using bilinear attention
        relation_logits = self._extract_relations(sequence_output, attention_mask)
        
        # Additional outputs
        extra_outputs = {
            'entity_representations': sequence_output,
            'attention_weights': None  # Can be filled by attention mechanism
        }
        
        return entity_logits, relation_logits, extra_outputs
    
    def _extract_relations(
        self, 
        sequence_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract relations between token pairs.
        
        Args:
            sequence_output: Encoded sequence [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            relation_logits: Relation logits [batch_size, seq_len, seq_len, num_relations]
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # Project to head and tail representations
        head_repr = self.relation_head_proj(sequence_output)  # [B, seq_len, H/2]
        tail_repr = self.relation_tail_proj(sequence_output)  # [B, seq_len, H/2]
        
        # Create pairwise representations
        head_expanded = head_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, seq_len, seq_len, H/2]
        tail_expanded = tail_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, seq_len, seq_len, H/2]
        
        # Concatenate head and tail representations
        pairwise_repr = torch.cat([head_expanded, tail_expanded], dim=-1)  # [B, seq_len, seq_len, H]
        
        # Classify relations
        relation_logits = self.relation_classifier(pairwise_repr)  # [B, seq_len, seq_len, num_relations]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Create pairwise mask
            mask_expanded = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)  # [B, seq_len, seq_len]
            mask_expanded = mask_expanded.unsqueeze(-1).expand(-1, -1, -1, self.num_relations)
            relation_logits = relation_logits.masked_fill(~mask_expanded.bool(), float('-inf'))
        
        return relation_logits
    
    def extract_entities(
        self, 
        entity_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> List[List[Dict]]:
        """
        Extract entity spans from entity logits.
        
        Args:
            entity_logits: Entity logits [batch_size, seq_len, num_entities]
            attention_mask: Attention mask [batch_size, seq_len]
            threshold: Confidence threshold for entity extraction
            
        Returns:
            List of entity lists for each batch item
        """
        batch_size, seq_len, num_entities = entity_logits.shape
        
        # Apply softmax to get probabilities
        entity_probs = F.softmax(entity_logits, dim=-1)
        
        entities_batch = []
        
        for b in range(batch_size):
            entities = []
            mask = attention_mask[b] if attention_mask is not None else torch.ones(seq_len)
            
            for i in range(seq_len):
                if mask[i] == 0:  # Skip padded tokens
                    continue
                    
                # Get the most likely entity type (excluding background class 0)
                max_prob, max_type = torch.max(entity_probs[b, i, 1:], dim=0)
                max_type += 1  # Adjust for background class
                
                if max_prob > threshold:
                    entities.append({
                        'start': i,
                        'end': i + 1,
                        'type': max_type.item(),
                        'confidence': max_prob.item()
                    })
            
            entities_batch.append(entities)
        
        return entities_batch
    
    def extract_relations(
        self,
        relation_logits: torch.Tensor,
        entities: List[List[Dict]],
        threshold: float = 0.5
    ) -> List[List[Dict]]:
        """
        Extract relations from relation logits and entity spans.
        
        Args:
            relation_logits: Relation logits [batch_size, seq_len, seq_len, num_relations]
            entities: List of entity lists for each batch item
            threshold: Confidence threshold for relation extraction
            
        Returns:
            List of relation lists for each batch item
        """
        batch_size = relation_logits.shape[0]
        
        # Apply softmax to get probabilities
        relation_probs = F.softmax(relation_logits, dim=-1)
        
        relations_batch = []
        
        for b in range(batch_size):
            relations = []
            batch_entities = entities[b]
            
            # Extract relations between entity pairs
            for i, head_entity in enumerate(batch_entities):
                for j, tail_entity in enumerate(batch_entities):
                    if i == j:  # Skip self-relations
                        continue
                    
                    head_pos = head_entity['start']
                    tail_pos = tail_entity['start']
                    
                    # Get relation probabilities (excluding background class 0)
                    rel_probs = relation_probs[b, head_pos, tail_pos, 1:]
                    max_prob, max_rel = torch.max(rel_probs, dim=0)
                    max_rel += 1  # Adjust for background class
                    
                    if max_prob > threshold:
                        relations.append({
                            'head': i,
                            'tail': j,
                            'type': max_rel.item(),
                            'confidence': max_prob.item()
                        })
            
            relations_batch.append(relations)
        
        return relations_batch
