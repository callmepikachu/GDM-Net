import json
import torch
import os
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

        # Initialize tokenizer with error handling - try local first
        local_model_path = "models"
        local_tokenizer_files = [
            os.path.join(local_model_path, "tokenizer.json"),
            os.path.join(local_model_path, "vocab.txt"),
            os.path.join(local_model_path, "tokenizer_config.json")
        ]

        try:
            # Check if we have the necessary tokenizer files locally
            if os.path.exists(local_model_path) and any(os.path.exists(f) for f in local_tokenizer_files):
                print(f"Loading tokenizer from local path: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
            else:
                print(f"Loading tokenizer from HuggingFace: {tokenizer_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        except Exception as e:
            print(f"Warning: Failed to load tokenizer. Creating basic tokenizer.")
            print(f"Error: {str(e)}")
            # Create a basic tokenizer with required methods
            class BasicTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                    self.vocab_size = 30522

                def __call__(self, text, max_length=512, padding='max_length', truncation=True, return_tensors='pt'):
                    # Simple tokenization - convert to token IDs
                    if isinstance(text, str):
                        # Simple word-based tokenization
                        tokens = text.lower().split()[:max_length-2]  # Leave space for [CLS] and [SEP]
                        token_ids = [101] + [hash(token) % (self.vocab_size-1000) + 1000 for token in tokens] + [102]
                    else:
                        token_ids = [101, 102]  # Just [CLS] and [SEP]

                    # Pad or truncate
                    if len(token_ids) < max_length:
                        token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
                    else:
                        token_ids = token_ids[:max_length]

                    attention_mask = [1 if tid != self.pad_token_id else 0 for tid in token_ids]

                    return {
                        'input_ids': torch.tensor([token_ids], dtype=torch.long),
                        'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
                    }

            self.tokenizer = BasicTokenizer()

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

    def _extract_label(self, sample):
        """Extract label from a sample."""
        # HotpotQA typically has 'answer' field, not 'label'
        # We need to convert answer to classification label
        if 'answer' in sample:
            answer = sample['answer'].lower().strip()
            # Map common answers to class indices
            if answer in ['yes', 'true', '1']:
                return 1
            elif answer in ['no', 'false', '0']:
                return 0
            else:
                # For other answers, use hash to get consistent label
                return hash(answer) % 5  # Assuming 5 classes
        elif 'label' in sample:
            return int(sample['label'])
        else:
            return 0

    def get_label_distribution(self):
        """Get distribution of labels in the dataset."""
        label_counts = {}
        for sample in self.data:
            label = self._extract_label(sample)
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Extract components
        document = sample['document']
        query = sample['query']
        entities = sample.get('entities', [])
        relations = sample.get('relations', [])
        label = self._extract_label(sample)
        
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

        # Create the return dictionary
        result = {
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

        # Check for NaN/Inf in the data before returning
        for key, tensor in result.items():
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any():
                    print(f"WARNING: NaN detected in dataset output {key} at sample {idx}")
                    print(f"  NaN count: {torch.isnan(tensor).sum()}")
                    # Replace NaN with safe values
                    if 'input_ids' in key:
                        result[key] = torch.where(torch.isnan(tensor), torch.ones_like(tensor), tensor)
                    elif 'attention_mask' in key:
                        result[key] = torch.where(torch.isnan(tensor), torch.ones_like(tensor), tensor)
                    else:
                        result[key] = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

                if torch.isinf(tensor).any():
                    print(f"WARNING: Inf detected in dataset output {key} at sample {idx}")
                    print(f"  Inf count: {torch.isinf(tensor).sum()}")
                    # Replace Inf with safe values
                    if 'input_ids' in key:
                        result[key] = torch.where(torch.isinf(tensor), torch.ones_like(tensor), tensor)
                    elif 'attention_mask' in key:
                        result[key] = torch.where(torch.isinf(tensor), torch.ones_like(tensor), tensor)
                    else:
                        result[key] = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)

        return result
    

