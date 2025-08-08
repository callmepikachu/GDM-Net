import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from .bert_encoder import DocumentEncoder, StructureExtractor
from .graph_memory import GraphWriter, GraphMemory
from .dual_memory import DualMemorySystem
from .reasoning_module import ReasoningModule


class GDMNet(nn.Module):
    """Graph-Augmented Dual Memory Network with dual-path processing."""

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_entities: int = 9,
        num_relations: int = 10,
        num_classes: int = 5,
        gnn_type: str = "rgcn",
        num_gnn_layers: int = 3,
        num_reasoning_hops: int = 4,
        fusion_method: str = "gate",
        dropout_rate: float = 0.1,
        entity_loss_weight: float = 0.1,
        relation_loss_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.entity_loss_weight = entity_loss_weight
        self.relation_loss_weight = relation_loss_weight

        # Document encoder (BERT-based)
        self.document_encoder = DocumentEncoder(
            model_name=bert_model_name,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )

        # Structure extractor for entities and relations
        self.structure_extractor = StructureExtractor(
            hidden_size=hidden_size,
            num_entity_types=num_entities,
            num_relation_types=num_relations,
            dropout_rate=dropout_rate
        )

        # Graph writer to convert structures to graph format
        self.graph_writer = GraphWriter(
            hidden_size=hidden_size,
            num_entity_types=num_entities,
            num_relation_types=num_relations
        )

        # Graph memory for processing structured knowledge
        self.graph_memory = GraphMemory(
            hidden_size=hidden_size,
            num_relation_types=num_relations,
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            dropout_rate=dropout_rate
        )

        # Dual memory system (optional, can be integrated later)
        self.dual_memory = DualMemorySystem(
            hidden_size=hidden_size,
            memory_size=hidden_size,
            dropout_rate=dropout_rate
        )

        # Reasoning module with PathFinder and GraphReader
        self.reasoning_module = ReasoningModule(
            hidden_size=hidden_size,
            max_hops=num_reasoning_hops,
            fusion_method=fusion_method,
            dropout_rate=dropout_rate
        )
        
        # Classification head for main task
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Loss functions
        self.main_loss_fn = nn.CrossEntropyLoss()
        self.entity_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.relation_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        entity_spans: torch.Tensor,
        entity_labels: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GDM-Net with dual-path processing.

        Args:
            query_input_ids: [batch_size, query_len]
            query_attention_mask: [batch_size, query_len]
            doc_input_ids: [batch_size, doc_len]
            doc_attention_mask: [batch_size, doc_len]
            entity_spans: [batch_size, num_entities, 2]
            entity_labels: [batch_size, seq_len] (optional, for training)
            relation_labels: [batch_size, num_pairs] (optional, for training)

        Returns:
            Dictionary containing logits and intermediate representations
        """

        # Step 1: Document encoding (implicit path)
        doc_pooled, doc_sequence = self.document_encoder.encode_document(
            doc_input_ids, doc_attention_mask
        )

        # Step 2: Query encoding
        query_pooled, query_sequence = self.document_encoder.encode_query(
            query_input_ids, query_attention_mask
        ) if query_input_ids is not None else (doc_pooled, doc_sequence)

        # Step 3: Structure extraction (explicit path)
        entity_logits, relation_logits, entities_batch, relations_batch = self.structure_extractor(
            doc_sequence, doc_attention_mask, entity_spans
        )

        # Step 4: Graph construction
        node_features, edge_index, edge_type, batch_indices = self.graph_writer(
            entities_batch, relations_batch, doc_sequence
        )

        # Step 5: Graph memory processing
        updated_node_features = self.graph_memory(
            node_features, edge_index, edge_type, batch_indices
        )

        # Step 6: Dual memory processing (optional enhancement)
        memory_output, episodic_output, semantic_output = self.dual_memory(
            query_pooled
        )

        # Step 7: Multi-hop reasoning and fusion
        fused_representation, path_representation, graph_representation = self.reasoning_module(
            query_pooled, doc_pooled, updated_node_features, edge_index, edge_type, batch_indices
        )

        # Step 8: Final classification
        logits = self.classifier(fused_representation)

        # Prepare outputs
        outputs = {
            'logits': logits,
            'entity_logits': entity_logits,
            'relation_logits': relation_logits,
            'doc_representation': doc_pooled,
            'query_representation': query_pooled,
            'graph_representation': graph_representation,
            'path_representation': path_representation,
            'memory_output': memory_output,
            'fused_representation': fused_representation,
            'entities_batch': entities_batch,
            'relations_batch': relations_batch
        }

        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        entity_labels: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with main task and auxiliary tasks.

        Args:
            outputs: Model outputs dictionary
            labels: Main task labels [batch_size]
            entity_labels: Entity labels [batch_size, seq_len] (optional)
            relation_labels: Relation labels [batch_size, num_pairs] (optional)

        Returns:
            Dictionary containing different loss components
        """

        # Main classification loss
        main_loss = self.main_loss_fn(outputs['logits'], labels)

        total_loss = main_loss
        loss_dict = {'main_loss': main_loss}

        # Entity extraction auxiliary loss
        if entity_labels is not None and 'entity_logits' in outputs:
            entity_logits = outputs['entity_logits']
            batch_size, seq_len, num_classes = entity_logits.shape

            # Flatten for loss computation
            entity_logits_flat = entity_logits.view(-1, num_classes)
            entity_labels_flat = entity_labels.view(-1)

            entity_loss = self.entity_loss_fn(entity_logits_flat, entity_labels_flat)
            total_loss += self.entity_loss_weight * entity_loss
            loss_dict['entity_loss'] = entity_loss

        # Relation extraction auxiliary loss
        if relation_labels is not None and 'relation_logits' in outputs:
            relation_logits = outputs['relation_logits']
            if relation_logits.numel() > 0 and relation_labels.numel() > 0:
                # Ensure shapes match
                min_size = min(relation_logits.size(1), relation_labels.size(1))
                if min_size > 0:
                    relation_logits_subset = relation_logits[:, :min_size]
                    relation_labels_subset = relation_labels[:, :min_size]

                    relation_logits_flat = relation_logits_subset.view(-1, relation_logits.size(-1))
                    relation_labels_flat = relation_labels_subset.view(-1)

                    relation_loss = self.relation_loss_fn(relation_logits_flat, relation_labels_flat)
                    total_loss += self.relation_loss_weight * relation_loss
                    loss_dict['relation_loss'] = relation_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict
    
    def predict(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        entity_spans: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Make predictions."""

        with torch.no_grad():
            outputs = self.forward(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                doc_input_ids=doc_input_ids,
                doc_attention_mask=doc_attention_mask,
                entity_spans=entity_spans,
                **kwargs
            )

            predictions = torch.argmax(outputs['logits'], dim=1)

        return predictions
