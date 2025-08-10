import json
import torch
import os
import pickle
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import logging

# 🚀 全局缓存，避免多worker重复加载
_GLOBAL_TOKENIZED_CACHE = {}

def clear_global_cache():
    """清理全局缓存（可选，用于释放内存）"""
    global _GLOBAL_TOKENIZED_CACHE
    _GLOBAL_TOKENIZED_CACHE.clear()

def get_cache_info():
    """获取缓存信息"""
    global _GLOBAL_TOKENIZED_CACHE
    return {
        'cached_files': list(_GLOBAL_TOKENIZED_CACHE.keys()),
        'cache_count': len(_GLOBAL_TOKENIZED_CACHE)
    }
from ..utils.graph_utils import build_graph


class HotpotQADataset(Dataset):
    """HotpotQA dataset for multi-document reasoning."""

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 2048,  # 🚀 扩大到2048
        max_query_length: int = 256,  # 🚀 相应扩大query长度
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

            # 根据加载模式显示不同信息
            if hasattr(self, 'num_samples'):
                # 分片模式
                self.logger.info(f"Loaded {self.num_samples} samples in {self.num_shards} shards")
            elif hasattr(self, 'tokenized_data'):
                # 单文件模式
                self.logger.info(f"Loaded {len(self.tokenized_data)} pre-tokenized samples")
            else:
                self.logger.warning("Unknown loading mode")
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
        """获取预处理文件路径（优先查找分片元数据）"""
        base_name = os.path.basename(self.data_path).replace('.json', '')

        # 优先查找分片元数据文件
        sharded_metadata_file = os.path.join(self.pretokenized_dir, f"sharded_metadata_{base_name}.json")

        if os.path.exists(sharded_metadata_file):
            return sharded_metadata_file

        # 回退到单文件模式
        filename = os.path.basename(self.data_path).replace('.json', '.pkl')
        return os.path.join(self.pretokenized_dir, f"tokenized_{filename}")

    def _load_pretokenized_data(self, pretokenized_file: str):
        """加载预处理的tokenization数据（支持分片和缓存）"""
        if pretokenized_file.endswith('.json'):
            # 分片模式：加载元数据
            self._load_sharded_data(pretokenized_file)
        else:
            # 单文件模式：直接加载
            self._load_single_file_data(pretokenized_file)

    def _load_sharded_data(self, metadata_file: str):
        """加载分片数据的元数据"""
        with open(metadata_file, 'r') as f:
            self.shard_metadata = json.load(f)

        self.num_samples = self.shard_metadata['num_samples']
        self.num_shards = self.shard_metadata['num_shards']
        self.shard_size = self.shard_metadata['shard_size']
        self.shard_files = [
            os.path.join(self.pretokenized_dir, fname)
            for fname in self.shard_metadata['shard_files']
        ]

        # 初始化分片缓存
        self.shard_cache = {}

        # 创建虚拟data属性用于兼容性
        self.data = [None] * self.num_samples  # 占位符

        self.logger.info(f"Loaded sharded metadata: {self.num_shards} shards, {self.num_samples} samples")

    def _load_single_file_data(self, pretokenized_file: str):
        """加载单文件数据（保持向后兼容）"""
        global _GLOBAL_TOKENIZED_CACHE

        # 检查全局缓存
        if pretokenized_file in _GLOBAL_TOKENIZED_CACHE:
            self.logger.info(f"Using cached tokenized data for {pretokenized_file}")
            self.tokenized_data = _GLOBAL_TOKENIZED_CACHE[pretokenized_file]
        else:
            self.logger.info(f"Loading tokenized data from disk: {pretokenized_file}")
            with open(pretokenized_file, 'rb') as f:
                tokenized_data = pickle.load(f)

            # 缓存到全局变量
            _GLOBAL_TOKENIZED_CACHE[pretokenized_file] = tokenized_data
            self.tokenized_data = tokenized_data
            self.logger.info(f"Cached tokenized data for future workers")

        # 从tokenized_data中提取原始数据，确保data属性存在
        self.data = [item['original_sample'] for item in self.tokenized_data]

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file (使用全局缓存避免重复加载)."""
        global _GLOBAL_TOKENIZED_CACHE

        cache_key = f"raw_data_{self.data_path}"

        if cache_key in _GLOBAL_TOKENIZED_CACHE:
            self.logger.info(f"Using cached raw data for {self.data_path}")
            return _GLOBAL_TOKENIZED_CACHE[cache_key]
        else:
            self.logger.info(f"Loading raw data from disk: {self.data_path}")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 缓存原始数据
            _GLOBAL_TOKENIZED_CACHE[cache_key] = data
            self.logger.info(f"Cached raw data for future workers")
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
    
    def _get_sample_from_shard(self, idx: int) -> Dict[str, Any]:
        """从分片中获取样本（按需加载）"""
        # 计算样本属于哪个分片
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size

        # 检查分片是否已缓存
        if shard_idx not in self.shard_cache:
            # 按需加载分片
            shard_file = self.shard_files[shard_idx]

            global _GLOBAL_TOKENIZED_CACHE
            if shard_file in _GLOBAL_TOKENIZED_CACHE:
                shard_data = _GLOBAL_TOKENIZED_CACHE[shard_file]
            else:
                with open(shard_file, 'rb') as f:
                    shard_data = pickle.load(f)
                _GLOBAL_TOKENIZED_CACHE[shard_file] = shard_data

            self.shard_cache[shard_idx] = shard_data

        return self.shard_cache[shard_idx][local_idx]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if hasattr(self, 'num_samples'):
            return self.num_samples
        elif hasattr(self, 'tokenized_data'):
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
        """Get a single sample using pre-computed tokenization (支持分片访问)."""
        if hasattr(self, 'shard_metadata'):
            # 分片模式：按需加载分片
            tokenized_sample = self._get_sample_from_shard(idx)
        else:
            # 单文件模式：直接访问
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

        # 🚀 提取原始文本用于SpaCy处理
        query_text = sample.get('question', '')

        # 重构文档文本（与预处理时相同的逻辑）
        context = sample.get('context', [])
        if isinstance(context, list) and len(context) > 0:
            document_parts = []
            for ctx_item in context:
                if isinstance(ctx_item, list) and len(ctx_item) >= 2:
                    title = ctx_item[0]
                    sentences = ctx_item[1]
                    if isinstance(sentences, list):
                        doc_text = f"{title}. " + " ".join(sentences)
                    else:
                        doc_text = f"{title}. {sentences}"
                    document_parts.append(doc_text)
            doc_text = " ".join(document_parts)
        else:
            doc_text = sample.get('document', '')

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
            'metadata': sample.get('metadata', {}),
            # 🚀 添加原始文本用于SpaCy处理
            'query_text': query_text,
            'doc_text': doc_text
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
    

