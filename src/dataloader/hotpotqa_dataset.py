import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import logging
from ..utils.graph_utils import build_graph


class HotpotQADataset(Dataset):
    """HotpotQA dataset for multi-document reasoning."""

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        max_query_length: int = 64,
        num_entities: int = 9,
        num_relations: int = 10
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.num_entities = num_entities
        self.num_relations = num_relations

        # Entity type mapping
        self.entity_type_map = {
            'TITLE': 1,
            'PERSON': 2,
            'LOCATION': 3,
            'ORGANIZATION': 4,
            'DATE': 5,
            'NUMBER': 6,
            'MISC': 7,
            'O': 0  # Outside entity
        }

        # Relation type mapping
        self.relation_type_map = {
            'NO_RELATION': 0,
            'RELATED': 1,
            'TEMPORAL': 2,
            'CAUSAL': 3,
            'SPATIAL': 4
        }

        # Initialize tokenizer with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {tokenizer_name}. Using fallback.")
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)

        # Load data
        self.data = self._load_data()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Extract components
        document = sample['document']
        query = sample['query']
        entities = sample.get('entities', [])
        relations = sample.get('relations', [])
        label = sample.get('label', 0)
        
        # Tokenize query and document
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        doc_encoding = self.tokenizer(
            document,
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        # Note: Graph construction is now handled by GraphWriter in the model
        
        # Create entity spans tensor
        entity_spans = torch.zeros(self.num_entities, 2, dtype=torch.long)
        for i, entity in enumerate(entities[:self.num_entities]):
            span = entity.get('span', [0, 0])
            # Ensure span values are integers and within bounds
            start = min(max(int(span[0]), 0), self.max_length - 1)
            end = min(max(int(span[1]), start + 1), self.max_length)
            entity_spans[i] = torch.tensor([start, end], dtype=torch.long)
        
        # Create entity and relation labels for auxiliary tasks (optional)
        entity_labels = torch.zeros(doc_encoding['input_ids'].size(1), dtype=torch.long)
        relation_labels = torch.zeros(len(relations), dtype=torch.long)

        # Simple entity labeling based on spans
        for i, entity in enumerate(entities[:self.num_entities]):
            span = entity.get('span', [0, 0])
            start, end = int(span[0]), int(span[1])
            start = min(max(start, 0), entity_labels.size(0) - 1)
            end = min(max(end, start + 1), entity_labels.size(0))
            if start < end:
                # Map string type to integer
                entity_type_str = entity.get('type', 'O')
                entity_type_id = self.entity_type_map.get(entity_type_str, 0)
                entity_labels[start:end] = entity_type_id

        # Simple relation labeling
        for i, relation in enumerate(relations):
            if i < relation_labels.size(0):
                # Map string type to integer
                relation_type_str = relation.get('type', 'NO_RELATION')
                relation_type_id = self.relation_type_map.get(relation_type_str, 0)
                relation_labels[i] = relation_type_id

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'doc_input_ids': doc_encoding['input_ids'].squeeze(0),
            'doc_attention_mask': doc_encoding['attention_mask'].squeeze(0),
            'entity_spans': entity_spans,
            'entity_labels': entity_labels,
            'relation_labels': relation_labels,
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': sample.get('metadata', {})
        }
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in dataset."""
        label_counts = {}
        for sample in self.data:
            label = sample.get('label', 0)
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
