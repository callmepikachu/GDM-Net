import json
import torch
import os
import pickle
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
        num_relations: int = 10,
        pretokenized_dir: str = "/root/autodl-tmp/hotpotqa-pretokenized"
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.pretokenized_dir = pretokenized_dir

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

        # 🚀 尝试加载预处理的tokenization数据
        pretokenized_file = self._get_pretokenized_file_path()

        if os.path.exists(pretokenized_file):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(f"Loading pre-tokenized data from {pretokenized_file}")
            self._load_pretokenized_data(pretokenized_file)
            self.logger.info(f"Loaded {len(self.tokenized_data)} pre-tokenized samples")
        else:
            # 回退到原始方法
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(f"Pre-tokenized file not found: {pretokenized_file}")
            self.logger.info("Loading original data and computing tokenization...")

            # Load data
            self.data = self._load_data()
            self.logger.info(f"Loaded {len(self.data)} samples from {data_path}")

            # 预计算tokenization
            self.logger.info(f"Pre-computing tokenization for {len(self.data)} samples...")
            self._precompute_tokenization()
            self.logger.info("Tokenization pre-computation completed!")
    
    def _get_pretokenized_file_path(self) -> str:
        """获取预处理文件路径"""
        filename = os.path.basename(self.data_path).replace('.json', '.pkl')
        return os.path.join(self.pretokenized_dir, f"tokenized_{filename}")

    def _load_pretokenized_data(self, pretokenized_file: str):
        """加载预处理的tokenization数据"""
        with open(pretokenized_file, 'rb') as f:
            self.tokenized_data = pickle.load(f)

        # 从tokenized_data中提取原始数据，确保data属性存在
        self.data = [item['original_sample'] for item in self.tokenized_data]

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _precompute_tokenization(self):
        """预计算所有样本的tokenization，消除训练时的CPU瓶颈"""
        self.tokenized_data = []

        for i, sample in enumerate(self.data):
            if i % 1000 == 0:
                print(f"Tokenizing sample {i}/{len(self.data)}")

            # 提取查询和文档
            query = sample.get('question', '')
            document = ' '.join(sample.get('context', []))

            # 预计算tokenization
            query_tokens = self.tokenizer(
                query,
                max_length=self.max_query_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            doc_tokens = self.tokenizer(
                document,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 存储预计算的结果
            tokenized_sample = {
                'query_input_ids': query_tokens['input_ids'].squeeze(0),
                'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
                'doc_input_ids': doc_tokens['input_ids'].squeeze(0),
                'doc_attention_mask': doc_tokens['attention_mask'].squeeze(0),
                'original_sample': sample  # 保留原始数据用于其他处理
            }

            self.tokenized_data.append(tokenized_sample)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if hasattr(self, 'tokenized_data'):
            return len(self.tokenized_data)
        elif hasattr(self, 'data'):
            return len(self.data)
        else:
            return 0

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
        """Get a single sample using pre-computed tokenization."""
        # 🚀 使用预计算的tokenization，消除CPU瓶颈
        tokenized_sample = self.tokenized_data[idx]
        sample = tokenized_sample['original_sample']

        # Extract components
        entities = sample.get('entities', [])
        relations = sample.get('relations', [])
        label = self._extract_label(sample)

        # 直接使用预计算的tokenization结果
        query_input_ids = tokenized_sample['query_input_ids']
        query_attention_mask = tokenized_sample['query_attention_mask']
        doc_input_ids = tokenized_sample['doc_input_ids']
        doc_attention_mask = tokenized_sample['doc_attention_mask']
        
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
        entity_labels = torch.zeros(doc_input_ids.size(0), dtype=torch.long)  # 使用预处理的doc_input_ids
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

        # 🚀 使用预计算的tokenization结果创建返回字典
        result = {
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'doc_input_ids': doc_input_ids,
            'doc_attention_mask': doc_attention_mask,
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
    

