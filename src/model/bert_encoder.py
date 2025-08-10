import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Tuple, Optional
import os


class DocumentEncoder(nn.Module):
    """BERT-based document encoder with support for Chinese mirrors."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        freeze_bert: bool = True,  # 默认冻结BERT
        use_chinese_mirror: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size

        # Try to load from local models directory first
        local_model_path = "models"
        local_config_path = os.path.join(local_model_path, "config.json")
        local_weights_path = os.path.join(local_model_path, "pytorch_model.bin")

        try:
            if os.path.exists(local_config_path):
                print(f"Loading BERT config from local path: {local_model_path}")
                self.config = AutoConfig.from_pretrained(local_model_path)

                # Check if weights file exists
                if os.path.exists(local_weights_path):
                    print(f"Loading BERT weights from local path: {local_model_path}")
                    self.bert = AutoModel.from_pretrained(local_model_path, config=self.config, local_files_only=True)
                else:
                    print(f"Local config found but no weights file. Creating model with local config.")
                    from transformers import BertModel
                    self.bert = BertModel(self.config)
            else:
                # Set up Chinese mirror for Hugging Face if needed
                if use_chinese_mirror:
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                    os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'

                print(f"Loading BERT model from HuggingFace: {model_name}")
                self.config = AutoConfig.from_pretrained(model_name)
                self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        except Exception as e:
            print(f"Warning: Failed to load {model_name}. Using fallback model.")
            print(f"Error: {str(e)}")
            # Fallback: create a simple BERT-like model with correct hidden size
            from transformers import BertConfig, BertModel
            self.config = BertConfig(
                vocab_size=30522,
                hidden_size=hidden_size,  # Use the target hidden size directly
                num_hidden_layers=6,  # Smaller model for fallback
                num_attention_heads=12,
                intermediate_size=hidden_size * 4,
                max_position_embeddings=512
            )
            self.bert = BertModel(self.config)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            print(f"Freezing BERT parameters for {model_name}")
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            print(f"BERT parameters will be fine-tuned for {model_name}")
        
        # Projection layers - ensure correct dimensions
        bert_hidden_size = getattr(self.config, 'hidden_size', hidden_size)
        # Only add projection if dimensions don't match
        if bert_hidden_size != hidden_size:
            self.doc_projection = nn.Linear(bert_hidden_size, hidden_size)
            self.query_projection = nn.Linear(bert_hidden_size, hidden_size)
            self.need_projection = True
        else:
            self.doc_projection = nn.Identity()
            self.query_projection = nn.Identity()
            self.need_projection = False
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through document encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]

        Returns:
            pooled_output: Document-level representation [batch_size, hidden_size]
            sequence_output: Token-level representations [batch_size, seq_len, hidden_size]
        """

        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # Extract outputs
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, bert_hidden]
        pooled_output = outputs.pooler_output  # [batch_size, bert_hidden]

        # Project to target hidden size if needed
        if self.need_projection:
            pooled_output = self.layer_norm(self.dropout(
                self.doc_projection(pooled_output)
            ))

            sequence_output = self.layer_norm(self.dropout(
                self.doc_projection(sequence_output)
            ))
        else:
            pooled_output = self.layer_norm(self.dropout(pooled_output))
            sequence_output = self.layer_norm(self.dropout(sequence_output))

        return pooled_output, sequence_output
    
    def encode_document(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode document."""
        return self.forward(doc_input_ids, doc_attention_mask)

    def encode_query(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode query."""
        return self.forward(query_input_ids, query_attention_mask)


class StructureExtractor(nn.Module):
    """Extract entities and relations using SpaCy + lightweight adapters."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_entity_types: int = 9,
        num_relation_types: int = 10,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_entity_types = num_entity_types
        self.num_relation_types = num_relation_types

        # 🚀 加载预训练SpaCy模型 (完全冻结)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Loaded SpaCy en_core_web_sm model")
        except OSError:
            print("⚠️ SpaCy en_core_web_sm not found, trying en_core_web_trf...")
            try:
                self.nlp = spacy.load("en_core_web_trf")
                print("✅ Loaded SpaCy en_core_web_trf model")
            except OSError:
                print("❌ No SpaCy model found. Please install: python -m spacy download en_core_web_sm")
                raise RuntimeError("SpaCy model not available")

        # SpaCy实体类型到自定义类型的映射
        self.spacy_to_custom = {
            'PERSON': 2,      # 人名
            'ORG': 4,         # 组织
            'GPE': 3,         # 地理政治实体 (地点)
            'LOC': 3,         # 地点
            'DATE': 5,        # 日期
            'TIME': 5,        # 时间
            'CARDINAL': 6,    # 基数
            'ORDINAL': 6,     # 序数
            'QUANTITY': 6,    # 数量
            'MONEY': 6,       # 金钱
            'PERCENT': 6,     # 百分比
            'MISC': 7,        # 其他
            'EVENT': 7,       # 事件
            'FAC': 7,         # 设施
            'LANGUAGE': 7,    # 语言
            'LAW': 7,         # 法律
            'NORP': 7,        # 国籍/宗教/政治团体
            'PRODUCT': 7,     # 产品
            'WORK_OF_ART': 7, # 艺术作品
        }

        # 🔥 轻量级适配器 (只有这些参数需要训练)
        # 实体适配器：将SpaCy特征映射到我们的隐藏空间
        self.entity_adapter = nn.Sequential(
            nn.Linear(96, hidden_size // 2),  # SpaCy token.vector维度通常是96
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # 关系适配器：处理实体对关系
        self.relation_adapter = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_relation_types)
        )

        # 实体类型分类器
        self.entity_type_classifier = nn.Linear(hidden_size, num_entity_types)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_spans: Optional[torch.Tensor] = None,
        input_texts: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """
        Extract entities and relations using SpaCy + adapters.

        Args:
            sequence_output: [batch_size, seq_len, hidden_size] - BERT输出
            attention_mask: [batch_size, seq_len]
            entity_spans: [batch_size, num_entities, 2] (optional)
            input_texts: List of original text strings for SpaCy processing

        Returns:
            entity_logits: [batch_size, seq_len, num_entity_types]
            relation_logits: [batch_size, num_pairs, num_relation_types]
            entities_batch: List of entity information per batch
            relations_batch: List of relation information per batch
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        device = sequence_output.device

        # 如果没有提供原始文本，回退到基于span的方法
        if input_texts is None:
            print("⚠️ No input_texts provided, using fallback extraction")
            return self._fallback_extraction(sequence_output, attention_mask, entity_spans)

        # 🚀 使用SpaCy进行实体关系提取
        entities_batch = []
        relations_batch = []

        for b, text in enumerate(input_texts):
            # 🔒 使用冻结的SpaCy模型进行NER (不参与梯度计算)
            try:
                doc = self.nlp(text)
            except Exception as e:
                print(f"❌ SpaCy processing failed for batch {b}: {e}")
                doc = None

            batch_entities = []

            # 🔧 获取BERT实际处理的序列长度
            bert_seq_len = sequence_output.size(1)  # 实际的BERT序列长度
            max_valid_pos = bert_seq_len - 1  # 最大有效位置

            if doc and doc.ents:
                valid_entities = 0
                skipped_entities = 0

                for i, ent in enumerate(doc.ents):
                    # 🔧 首先检查实体位置是否在BERT序列范围内
                    if ent.start >= bert_seq_len:
                        skipped_entities += 1
                        continue  # 完全超出范围，跳过

                    # 获取SpaCy的实体向量 (冻结特征)
                    if hasattr(ent, 'vector') and ent.vector.shape[0] > 0:
                        spacy_vector = torch.tensor(ent.vector, dtype=torch.float32, device=device)
                    else:
                        # 如果没有向量，使用零向量
                        spacy_vector = torch.zeros(96, device=device)

                    try:
                        # 🔥 通过可训练的适配器映射到我们的隐藏空间
                        entity_repr = self.entity_adapter(spacy_vector)

                        # 映射SpaCy实体类型到我们的类型
                        spacy_label = ent.label_
                        custom_type = self.spacy_to_custom.get(spacy_label, 7)  # 默认为MISC

                        # 🔧 修正span边界以适应BERT序列长度
                        start_pos = min(ent.start, max_valid_pos)
                        end_pos = min(ent.end, bert_seq_len)

                        # 确保start < end且都在有效范围内
                        if start_pos >= end_pos:
                            end_pos = min(start_pos + 1, bert_seq_len)

                        # 如果修正后的位置仍然有效，添加实体
                        if start_pos < bert_seq_len and end_pos <= bert_seq_len:
                            batch_entities.append({
                                'span': (start_pos, end_pos),
                                'type': custom_type,
                                'representation': entity_repr,
                                'text': ent.text,
                                'spacy_label': spacy_label
                            })
                            valid_entities += 1
                        else:
                            skipped_entities += 1

                    except Exception as e:
                        print(f"❌ Failed to process entity '{ent.text}': {e}")
                        skipped_entities += 1
                        continue

                # 🔍 调试信息：显示实体过滤结果
                if skipped_entities > 0:
                    print(f"🔧 Batch {b}: kept {valid_entities} entities, skipped {skipped_entities} out-of-range entities")

            entities_batch.append(batch_entities)

            # 关系提取：对实体对进行分类
            batch_relations = []
            entities = batch_entities

            if len(entities) > 1:
                for i in range(len(entities)):
                    for j in range(i+1, len(entities)):
                        head_repr = entities[i]['representation']
                        tail_repr = entities[j]['representation']

                        try:
                            # 🔥 通过可训练的关系适配器
                            pair_repr = torch.cat([head_repr, tail_repr], dim=0)
                            rel_logits = self.relation_adapter(pair_repr.unsqueeze(0))
                            rel_type = rel_logits.argmax(dim=-1).item()
                            confidence = torch.softmax(rel_logits, dim=-1).max().item()

                            if rel_type > 0:  # 非零关系
                                batch_relations.append({
                                    'head': i,
                                    'tail': j,
                                    'type': int(rel_type),
                                    'confidence': confidence
                                })

                        except Exception as e:
                            continue

            relations_batch.append(batch_relations)

        # 🔍 简洁的总体调试信息
        total_entities = sum(len(batch) for batch in entities_batch)
        total_relations = sum(len(batch) for batch in relations_batch)
        if total_entities > 0 or total_relations > 0:
            print(f"✅ StructureExtractor: {total_entities} entities, {total_relations} relations (BERT seq_len: {sequence_output.size(1)})")

        # 生成entity_logits用于损失计算
        entity_logits = torch.zeros(batch_size, seq_len, self.num_entity_types, device=device)

        # 根据SpaCy结果填充entity_logits
        for b, entities in enumerate(entities_batch):
            for entity in entities:
                start, end = entity['span']
                entity_type = entity['type']
                if start < seq_len and end <= seq_len:
                    entity_logits[b, start:end, entity_type] = 1.0

        # 生成relation_logits用于损失计算
        max_pairs = max(len(relations_batch[b]) for b in range(batch_size)) if any(relations_batch) else 1
        relation_logits = torch.zeros(batch_size, max_pairs, self.num_relation_types, device=device)

        for b, relations in enumerate(relations_batch):
            for r, relation in enumerate(relations[:max_pairs]):
                rel_type = relation['type']
                relation_logits[b, r, rel_type] = 1.0

        return entity_logits, relation_logits, entities_batch, relations_batch

    def _fallback_extraction(self, sequence_output, attention_mask, entity_spans):
        """回退到基于BERT的实体关系提取（保持向后兼容）"""
        batch_size, seq_len, hidden_size = sequence_output.shape
        device = sequence_output.device

        # 简化的实体分类
        entity_logits = torch.zeros(batch_size, seq_len, self.num_entity_types, device=device)
        entities_batch = []
        relations_batch = []

        for b in range(batch_size):
            batch_entities = []

            if entity_spans is not None:
                # 使用提供的entity spans
                for i, (start, end) in enumerate(entity_spans[b]):
                    start_idx = int(start.item()) if hasattr(start, 'item') else int(start)
                    end_idx = int(end.item()) if hasattr(end, 'item') else int(end)

                    if start_idx < seq_len and end_idx <= seq_len and start_idx < end_idx:
                        # 使用BERT表示通过适配器
                        bert_repr = sequence_output[b, start_idx:end_idx].mean(dim=0)

                        # 创建虚拟SpaCy向量
                        dummy_spacy_vector = torch.zeros(96, device=device)
                        entity_repr = self.entity_adapter(dummy_spacy_vector)

                        batch_entities.append({
                            'span': (start_idx, end_idx),
                            'type': 1,  # 默认类型
                            'representation': entity_repr
                        })

            entities_batch.append(batch_entities)
            relations_batch.append([])  # 空关系列表

        # 虚拟relation_logits
        relation_logits = torch.zeros(batch_size, 1, self.num_relation_types, device=device)

        return entity_logits, relation_logits, entities_batch, relations_batch
    
    def get_entity_representations(
        self,
        sequence_output: torch.Tensor,
        entity_spans: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract entity representations from sequence output.
        
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            entity_spans: [batch_size, num_entities, 2]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            entity_representations: [batch_size, num_entities, hidden_size]
        """
        batch_size, num_entities, _ = entity_spans.shape
        seq_len, hidden_size = sequence_output.shape[1], sequence_output.shape[2]
        
        entity_representations = []
        
        for b in range(batch_size):
            batch_entities = []
            for e in range(num_entities):
                start, end = entity_spans[b, e]
                start, end = max(0, start), min(seq_len, end)
                
                if start < end and start < seq_len:
                    # Average pooling over entity span
                    entity_repr = sequence_output[b, start:end].mean(dim=0)
                else:
                    # Use [CLS] token if span is invalid
                    entity_repr = sequence_output[b, 0]
                
                batch_entities.append(entity_repr)
            
            entity_representations.append(torch.stack(batch_entities))
        
        return torch.stack(entity_representations)
