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
        freeze_bert: bool = False,
        use_chinese_mirror: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size

        # Try to load from local models directory first
        local_model_path = "models"

        try:
            if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
                print(f"Loading BERT model from local path: {local_model_path}")
                self.config = AutoConfig.from_pretrained(local_model_path)
                self.bert = AutoModel.from_pretrained(local_model_path, config=self.config, local_files_only=True)
            else:
                # Set up Chinese mirror for Hugging Face if needed
                if use_chinese_mirror:
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                    os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'

                print(f"Loading BERT model from HuggingFace: {model_name}")
                self.config = AutoConfig.from_pretrained(model_name)
                self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        except Exception as e:
            print(f"Warning: Failed to load {model_name} from both local and HuggingFace. Using fallback model.")
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
            for param in self.bert.parameters():
                param.requires_grad = False
        
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
    """Extract entities and relations from sequence output."""

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

        # Entity type mapping (same as in dataset)
        self.entity_type_map = {
            'TITLE': 1,
            'PERSON': 2,
            'LOCATION': 3,
            'ORGANIZATION': 4,
            'DATE': 5,
            'NUMBER': 6,
            'MISC': 7,
            'O': 0
        }

        # Entity extraction layers
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_entity_types)
        )

        # Relation extraction layers
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_relation_types)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_spans: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """
        Extract entities and relations from sequence output.

        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            entity_spans: [batch_size, num_entities, 2] (optional)

        Returns:
            entity_logits: [batch_size, seq_len, num_entity_types]
            relation_logits: [batch_size, num_pairs, num_relation_types]
            entities_batch: List of entity information per batch
            relations_batch: List of relation information per batch
        """
        batch_size, seq_len, hidden_size = sequence_output.shape

        # Entity classification for each token
        entity_logits = self.entity_classifier(sequence_output)

        # Extract entities based on spans or predictions
        entities_batch = []
        relations_batch = []

        for b in range(batch_size):
            batch_entities = []

            if entity_spans is not None:
                # Use provided entity spans
                for i, (start, end) in enumerate(entity_spans[b]):
                    start_idx = int(start.item()) if hasattr(start, 'item') else int(start)
                    end_idx = int(end.item()) if hasattr(end, 'item') else int(end)

                    if start_idx < seq_len and end_idx <= seq_len and start_idx < end_idx:
                        entity_repr = sequence_output[b, start_idx:end_idx].mean(dim=0)
                        entity_type = entity_logits[b, start_idx:end_idx].mean(dim=0).argmax().item()

                        # Ensure entity_repr has correct dimension
                        if entity_repr.size(0) != self.hidden_size:
                            if entity_repr.size(0) < self.hidden_size:
                                padding = torch.zeros(self.hidden_size - entity_repr.size(0), device=entity_repr.device)
                                entity_repr = torch.cat([entity_repr, padding], dim=0)
                            else:
                                entity_repr = entity_repr[:self.hidden_size]

                        batch_entities.append({
                            'span': (start_idx, end_idx),
                            'type': int(entity_type),
                            'representation': entity_repr
                        })
            else:
                # Extract entities from predictions (simplified)
                entity_preds = entity_logits[b].argmax(dim=-1)
                for i in range(seq_len):
                    if entity_preds[i] > 0 and attention_mask[b, i] == 1:
                        entity_repr = sequence_output[b, i]

                        # Ensure entity_repr has correct dimension
                        if entity_repr.size(0) != self.hidden_size:
                            if entity_repr.size(0) < self.hidden_size:
                                padding = torch.zeros(self.hidden_size - entity_repr.size(0), device=entity_repr.device)
                                entity_repr = torch.cat([entity_repr, padding], dim=0)
                            else:
                                entity_repr = entity_repr[:self.hidden_size]

                        batch_entities.append({
                            'span': (i, i+1),
                            'type': int(entity_preds[i].item()),
                            'representation': entity_repr
                        })

            entities_batch.append(batch_entities)

        # Relation extraction between entity pairs
        relation_logits_list = []
        for b in range(batch_size):
            batch_relations = []
            entities = entities_batch[b]

            if len(entities) > 1:
                for i in range(len(entities)):
                    for j in range(i+1, len(entities)):
                        head_repr = entities[i]['representation']
                        tail_repr = entities[j]['representation']
                        pair_repr = torch.cat([head_repr, tail_repr], dim=0)
                        rel_logits = self.relation_classifier(pair_repr.unsqueeze(0))
                        rel_type = rel_logits.argmax(dim=-1).item()

                        if rel_type > 0:  # Non-zero relation
                            batch_relations.append({
                                'head': i,
                                'tail': j,
                                'type': int(rel_type)
                            })

            relations_batch.append(batch_relations)

        # Dummy relation logits for loss computation
        max_pairs = max(len(relations_batch[b]) for b in range(batch_size)) if any(relations_batch) else 1
        relation_logits = torch.zeros(batch_size, max_pairs, self.num_relation_types, device=sequence_output.device)

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
