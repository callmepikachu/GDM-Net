"""
Dataset Module for GDM-Net

This module provides dataset classes and utilities for loading and preprocessing
data for GDM-Net training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import os
from pathlib import Path


class GDMNetDataset(Dataset):
    """
    Dataset class for GDM-Net training and evaluation.
    
    Expected data format:
    {
        "documents": ["doc1 text", "doc2 text"],
        "entities": [{"span": [0, 3], "type": "PERSON"}, ...],
        "relations": [{"head": 0, "tail": 1, "type": "WORKS_FOR"}, ...],
        "query": "Who is the CEO of Apple?",
        "answer": "Tim Cook",
        "label": 0
    }
    
    Args:
        data_path (str): Path to the dataset file (JSON or JSONL)
        tokenizer_name (str): Name of the tokenizer to use
        max_length (int): Maximum sequence length
        max_query_length (int): Maximum query length
        entity_types (Dict): Mapping of entity type names to IDs
        relation_types (Dict): Mapping of relation type names to IDs
        label_mapping (Dict): Mapping of label names to IDs
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 512,
        max_query_length: int = 64,
        entity_types: Optional[Dict[str, int]] = None,
        relation_types: Optional[Dict[str, int]] = None,
        label_mapping: Optional[Dict[str, int]] = None
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.max_query_length = max_query_length
        
        # Initialize tokenizer (支持国内镜像)
        import os

        # 设置国内镜像
        mirror_urls = [
            "https://hf-mirror.com",  # HuggingFace国内镜像
            "https://huggingface.co"  # 原始地址作为备选
        ]

        self.tokenizer = None
        for mirror_url in mirror_urls:
            try:
                # 设置镜像环境变量
                os.environ['HF_ENDPOINT'] = mirror_url
                print(f"🔄 尝试从镜像加载tokenizer: {mirror_url}")

                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"✅ 成功从镜像加载tokenizer: {tokenizer_name} (镜像: {mirror_url})")
                break

            except Exception as e:
                print(f"⚠️ 镜像 {mirror_url} 连接失败: {str(e)[:100]}...")
                continue

        # 如果所有镜像都失败，尝试离线模式
        if self.tokenizer is None:
            try:
                print("🔄 尝试离线模式...")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
                print(f"✅ 离线模式加载tokenizer成功: {tokenizer_name}")
            except Exception as e:
                print(f"❌ 离线模式也失败，使用备选tokenizer: {str(e)[:100]}...")
                self.tokenizer = self._create_fallback_tokenizer()
        
        # Initialize type mappings
        self.entity_types = entity_types or self._create_default_entity_types()
        self.relation_types = relation_types or self._create_default_relation_types()
        self.label_mapping = label_mapping or {}
        
        # Load data
        self.data = self._load_data()

    def _check_local_cache(self):
        """检查本地是否有BERT模型缓存"""
        import os
        from pathlib import Path

        # 检查常见的缓存位置
        cache_dirs = [
            Path.home() / '.cache' / 'huggingface' / 'transformers',
            Path('/root/.cache/huggingface/transformers'),
            Path('/tmp/huggingface/transformers')
        ]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                # 查找bert-base-uncased相关文件
                for item in cache_dir.iterdir():
                    if 'bert-base-uncased' in str(item):
                        print(f"✅ 找到本地缓存: {item}")
                        return True
        return False

    def _create_fallback_tokenizer(self):
        """创建备选tokenizer"""
        print("🔧 创建基础tokenizer作为备选...")

        # 创建一个基础的词汇表
        vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
        }

        # 添加常用词汇
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'where', 'when', 'who', 'why', 'how', 'which', 'that', 'this'
        ]

        for i, word in enumerate(common_words, start=5):
            vocab[word] = i

        # 创建简单的tokenizer类
        class FallbackTokenizer:
            def __init__(self, vocab):
                self.vocab = vocab
                self.inv_vocab = {v: k for k, v in vocab.items()}
                self.pad_token_id = vocab['[PAD]']
                self.unk_token_id = vocab['[UNK]']
                self.cls_token_id = vocab['[CLS]']
                self.sep_token_id = vocab['[SEP]']
                self.mask_token_id = vocab['[MASK]']

            def encode(self, text, max_length=512, padding=True, truncation=True, return_tensors=None):
                # 简单的词汇分割
                words = text.lower().split()
                token_ids = [self.cls_token_id]

                for word in words:
                    token_id = self.vocab.get(word, self.unk_token_id)
                    token_ids.append(token_id)
                    if len(token_ids) >= max_length - 1:
                        break

                token_ids.append(self.sep_token_id)

                # 填充或截断
                if padding and len(token_ids) < max_length:
                    token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
                elif truncation and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]

                if return_tensors == 'pt':
                    import torch
                    return {'input_ids': torch.tensor([token_ids])}
                return {'input_ids': token_ids}

            def __call__(self, text, **kwargs):
                return self.encode(text, **kwargs)

        return FallbackTokenizer(vocab)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Entity types: {len(self.entity_types)}")
        print(f"Relation types: {len(self.relation_types)}")
        print(f"Label classes: {len(self.label_mapping)}")
    
    def _create_default_entity_types(self) -> Dict[str, int]:
        """Create default entity type mapping."""
        return {
            'O': 0,  # Background
            'PERSON': 1,
            'ORGANIZATION': 2,
            'LOCATION': 3,
            'DATE': 4,
            'MONEY': 5,
            'PRODUCT': 6,
            'EVENT': 7,
            'MISC': 8
        }
    
    def _create_default_relation_types(self) -> Dict[str, int]:
        """Create default relation type mapping."""
        return {
            'NO_RELATION': 0,  # Background
            'WORKS_FOR': 1,
            'FOUNDED_BY': 2,
            'LOCATED_IN': 3,
            'PART_OF': 4,
            'CEO_OF': 5,
            'ACQUIRED_BY': 6,
            'COMPETITOR_OF': 7,
            'SUBSIDIARY_OF': 8,
            'PARTNER_OF': 9
        }
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file."""
        data = []
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            if self.data_path.endswith('.jsonl'):
                # JSONL format
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:
                # JSON format
                data = json.load(f)
        
        # Process and validate data
        processed_data = []
        for item in data:
            processed_item = self._process_item(item)
            if processed_item:
                processed_data.append(processed_item)
        
        return processed_data
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate a single data item."""
        try:
            # Combine documents if multiple
            if isinstance(item.get('documents'), list):
                document = ' '.join(item['documents'])
            else:
                document = item.get('document', item.get('text', ''))
            
            if not document:
                return None
            
            # Process entities
            entities = []
            for entity in item.get('entities', []):
                entity_type = entity.get('type', 0)  # 默认为0

                # 如果是整数，直接使用（确保在范围内）
                if isinstance(entity_type, int):
                    entity_type_id = max(0, min(entity_type, 7))  # 强制在0-7范围内
                else:
                    # 如果是字符串，查找映射
                    entity_type_id = self.entity_types.get(entity_type, 0)  # 默认为0而不是8
                
                entities.append({
                    'span': entity['span'],
                    'type': entity_type_id,
                    'text': entity.get('text', '')
                })
            
            # Process relations
            relations = []
            for relation in item.get('relations', []):
                relation_type = relation.get('type', 0)  # 默认为0

                # 如果是整数，直接使用（确保在范围内）
                if isinstance(relation_type, int):
                    relation_type_id = max(0, min(relation_type, 3))  # 强制在0-3范围内
                else:
                    # 如果是字符串，查找映射
                    relation_type_id = self.relation_types.get(relation_type, 0)  # 默认为0
                
                relations.append({
                    'head': relation['head'],
                    'tail': relation['tail'],
                    'type': relation_type_id
                })
            
            # Process label
            label = item.get('label', item.get('answer', 0))
            if isinstance(label, str) and self.label_mapping:
                label = self.label_mapping.get(label, 0)
            elif not isinstance(label, int):
                label = 0
            
            return {
                'document': document,
                'query': item.get('query', ''),
                'entities': entities,
                'relations': relations,
                'label': label,
                'metadata': item.get('metadata', {})
            }
            
        except Exception as e:
            print(f"Error processing item: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.data[idx]
        
        # Tokenize document
        doc_encoding = self.tokenizer(
            item['document'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize query
        query_encoding = self.tokenizer(
            item['query'],
            max_length=self.max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create entity labels
        entity_labels = self._create_entity_labels(item['entities'])
        
        # Create relation labels
        relation_labels = self._create_relation_labels(item['relations'])
        
        return {
            'input_ids': doc_encoding['input_ids'].squeeze(0),
            'attention_mask': doc_encoding['attention_mask'].squeeze(0),
            'token_type_ids': doc_encoding.get('token_type_ids', torch.zeros_like(doc_encoding['input_ids'])).squeeze(0),
            'query': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'entity_labels': entity_labels,
            'relation_labels': relation_labels,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }
    
    def _create_entity_labels(self, entities: List[Dict]) -> torch.Tensor:
        """Create entity labels tensor."""
        entity_labels = torch.full((self.max_length,), -1, dtype=torch.long)
        
        for entity in entities:
            start, end = entity['span']
            entity_type = entity['type']
            
            # Map character spans to token spans (simplified)
            if start < self.max_length:
                entity_labels[start:min(end, self.max_length)] = entity_type
        
        return entity_labels
    
    def _create_relation_labels(self, relations: List[Dict]) -> torch.Tensor:
        """Create relation labels tensor."""
        relation_labels = torch.full((self.max_length, self.max_length), -1, dtype=torch.long)
        
        for relation in relations:
            head_idx = relation['head']
            tail_idx = relation['tail']
            relation_type = relation['type']
            
            if head_idx < self.max_length and tail_idx < self.max_length:
                relation_labels[head_idx, tail_idx] = relation_type
        
        return relation_labels
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'num_samples': len(self.data),
            'avg_doc_length': 0,
            'avg_query_length': 0,
            'avg_entities_per_sample': 0,
            'avg_relations_per_sample': 0,
            'entity_type_distribution': {},
            'relation_type_distribution': {},
            'label_distribution': {}
        }
        
        doc_lengths = []
        query_lengths = []
        entity_counts = []
        relation_counts = []
        
        for item in self.data:
            doc_lengths.append(len(item['document'].split()))
            query_lengths.append(len(item['query'].split()))
            entity_counts.append(len(item['entities']))
            relation_counts.append(len(item['relations']))
            
            # Count entity types
            for entity in item['entities']:
                entity_type = entity['type']
                stats['entity_type_distribution'][entity_type] = \
                    stats['entity_type_distribution'].get(entity_type, 0) + 1
            
            # Count relation types
            for relation in item['relations']:
                relation_type = relation['type']
                stats['relation_type_distribution'][relation_type] = \
                    stats['relation_type_distribution'].get(relation_type, 0) + 1
            
            # Count labels
            label = item['label']
            stats['label_distribution'][label] = \
                stats['label_distribution'].get(label, 0) + 1
        
        stats['avg_doc_length'] = np.mean(doc_lengths)
        stats['avg_query_length'] = np.mean(query_lengths)
        stats['avg_entities_per_sample'] = np.mean(entity_counts)
        stats['avg_relations_per_sample'] = np.mean(relation_counts)
        
        return stats


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    tokenizer_name: str = 'bert-base-uncased',
    batch_size: int = 8,
    max_length: int = 512,
    max_query_length: int = 64,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data (optional)
        tokenizer_name: Name of the tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        max_query_length: Maximum query length
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = GDMNetDataset(
        train_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_query_length=max_query_length,
        **kwargs
    )
    
    val_dataset = GDMNetDataset(
        val_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_query_length=max_query_length,
        entity_types=train_dataset.entity_types,
        relation_types=train_dataset.relation_types,
        label_mapping=train_dataset.label_mapping
    )
    
    test_dataset = None
    if test_path:
        test_dataset = GDMNetDataset(
            test_path,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            max_query_length=max_query_length,
            entity_types=train_dataset.entity_types,
            relation_types=train_dataset.relation_types,
            label_mapping=train_dataset.label_mapping
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def create_synthetic_dataset(
    output_path: str,
    num_samples: int = 1000,
    format: str = 'json'
) -> None:
    """
    Create a synthetic dataset for testing and demonstration.
    
    Args:
        output_path: Path to save the synthetic dataset
        num_samples: Number of samples to generate
        format: Output format ('json' or 'jsonl')
    """
    # Sample templates
    templates = [
        {
            'document': "{company} is a technology company founded by {founder}. {ceo} is the current CEO.",
            'query': "Who is the CEO of {company}?",
            'entities': [
                {'span': [0, 5], 'type': 'ORGANIZATION'},
                {'span': [50, 60], 'type': 'PERSON'},
                {'span': [70, 80], 'type': 'PERSON'}
            ],
            'relations': [
                {'head': 0, 'tail': 2, 'type': 'CEO_OF'},
                {'head': 0, 'tail': 1, 'type': 'FOUNDED_BY'}
            ],
            'label': 0
        }
    ]
    
    # Sample data
    companies = ['Apple', 'Microsoft', 'Google', 'Tesla', 'Amazon']
    founders = ['Steve Jobs', 'Bill Gates', 'Larry Page', 'Elon Musk', 'Jeff Bezos']
    ceos = ['Tim Cook', 'Satya Nadella', 'Sundar Pichai', 'Elon Musk', 'Andy Jassy']
    
    # Generate samples
    samples = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        company = companies[i % len(companies)]
        founder = founders[i % len(founders)]
        ceo = ceos[i % len(ceos)]
        
        sample = {
            'document': template['document'].format(company=company, founder=founder, ceo=ceo),
            'query': template['query'].format(company=company),
            'entities': template['entities'],
            'relations': template['relations'],
            'label': template['label']
        }
        
        samples.append(sample)
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'jsonl':
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        else:
            json.dump(samples, f, indent=2)
    
    print(f"Created synthetic dataset with {num_samples} samples at {output_path}")


if __name__ == "__main__":
    # Create synthetic datasets for demonstration
    os.makedirs('data', exist_ok=True)
    
    create_synthetic_dataset('data/train.json', num_samples=800)
    create_synthetic_dataset('data/val.json', num_samples=100)
    create_synthetic_dataset('data/test.json', num_samples=100)
    
    print("Synthetic datasets created successfully!")
