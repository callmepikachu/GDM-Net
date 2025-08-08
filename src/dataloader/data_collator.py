import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from typing import List, Dict, Any


class GDMNetDataCollator:
    """Data collator for GDM-Net model."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of samples."""
        
        # Extract components
        query_input_ids = [item['query_input_ids'] for item in batch]
        query_attention_mask = [item['query_attention_mask'] for item in batch]
        doc_input_ids = [item['doc_input_ids'] for item in batch]
        doc_attention_mask = [item['doc_attention_mask'] for item in batch]
        entity_spans = [item['entity_spans'] for item in batch]
        labels = [item['label'] for item in batch]
        metadata = [item['metadata'] for item in batch]

        # Stack tensors
        batch_dict = {
            'query_input_ids': torch.stack(query_input_ids),
            'query_attention_mask': torch.stack(query_attention_mask),
            'doc_input_ids': torch.stack(doc_input_ids),
            'doc_attention_mask': torch.stack(doc_attention_mask),
            'entity_spans': torch.stack(entity_spans),
            'labels': torch.stack(labels),
            'metadata': metadata
        }

        # Add optional auxiliary labels if available
        if 'entity_labels' in batch[0]:
            entity_labels = [item['entity_labels'] for item in batch]
            batch_dict['entity_labels'] = torch.stack(entity_labels)

        if 'relation_labels' in batch[0]:
            relation_labels = [item['relation_labels'] for item in batch]
            # Handle variable length relation labels
            max_relations = max(len(rel_labels) for rel_labels in relation_labels) if relation_labels else 1
            if max_relations > 0:
                padded_relation_labels = []
                for rel_labels in relation_labels:
                    if len(rel_labels) < max_relations:
                        padding = torch.zeros(max_relations - len(rel_labels), dtype=rel_labels.dtype, device=rel_labels.device)
                        padded = torch.cat([rel_labels, padding])
                    else:
                        padded = rel_labels[:max_relations]
                    padded_relation_labels.append(padded)
                batch_dict['relation_labels'] = torch.stack(padded_relation_labels)

        return batch_dict
    
    def pad_sequences(
        self, 
        sequences: List[torch.Tensor], 
        padding_value: int = 0
    ) -> torch.Tensor:
        """Pad sequences to same length."""
        return pad_sequence(
            sequences, 
            batch_first=True, 
            padding_value=padding_value
        )
