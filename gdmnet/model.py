"""
GDM-Net Main Model

This module implements the main GDMNet model that combines all components
for graph-augmented dual memory network processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List, Dict, Union
import pytorch_lightning as pl

# 尝试导入 torchmetrics，如果失败则使用 pytorch_lightning.metrics
try:
    from torchmetrics import Accuracy
except ImportError:
    try:
        from pytorch_lightning.metrics import Accuracy
    except ImportError:
        # 如果都没有，创建一个简单的准确率计算类
        class Accuracy:
            def __init__(self, task='multiclass', num_classes=None):
                self.task = task
                self.num_classes = num_classes
                self.correct = 0
                self.total = 0

            def __call__(self, preds, target):
                if preds.dim() > 1:
                    preds = torch.argmax(preds, dim=-1)
                correct = (preds == target).sum().item()
                total = target.size(0)
                self.correct += correct
                self.total += total
                return correct / total if total > 0 else 0.0

            def compute(self):
                return self.correct / self.total if self.total > 0 else 0.0

            def reset(self):
                self.correct = 0
                self.total = 0

from .encoder import DocumentEncoder
from .extractor import StructureExtractor
from .graph_memory import GraphMemory, GraphWriter
from .reasoning import PathFinder, GraphReader, ReasoningFusion


class GDMNet(pl.LightningModule):
    """
    Graph-Augmented Dual Memory Network (GDM-Net) main model.
    
    A neural architecture that combines explicit graph memory with dual-path processing
    for structured information extraction and multi-hop reasoning over documents.
    
    Args:
        bert_model_name (str): Name of the pre-trained BERT model
        hidden_size (int): Hidden size of the model
        num_entities (int): Number of entity types
        num_relations (int): Number of relation types
        num_classes (int): Number of output classes
        gnn_type (str): Type of GNN ('rgcn' or 'gat')
        num_gnn_layers (int): Number of GNN layers
        num_reasoning_hops (int): Number of reasoning hops
        fusion_method (str): Fusion method for combining representations
        learning_rate (float): Learning rate for training
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        hidden_size: int = 768,
        num_entities: int = 10,
        num_relations: int = 20,
        num_classes: int = 5,
        gnn_type: str = 'rgcn',
        num_gnn_layers: int = 2,
        num_reasoning_hops: int = 3,
        fusion_method: str = 'gate',
        learning_rate: float = 2e-5,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model components
        self.encoder = DocumentEncoder(
            model_name=bert_model_name,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        self.structure_extractor = StructureExtractor(
            hidden_size=hidden_size,
            num_entities=num_entities,
            num_relations=num_relations,
            dropout_rate=dropout_rate
        )
        
        self.graph_writer = GraphWriter(
            hidden_size=hidden_size,
            node_dim=hidden_size,
            num_relations=num_relations
        )
        
        self.graph_memory = GraphMemory(
            node_dim=hidden_size,
            num_relations=num_relations,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout_rate=dropout_rate
        )
        
        self.path_finder = PathFinder(
            node_dim=hidden_size,
            query_dim=hidden_size,
            num_hops=num_reasoning_hops
        )
        
        self.graph_reader = GraphReader(
            node_dim=hidden_size,
            query_dim=hidden_size
        )
        
        self.reasoning_fusion = ReasoningFusion(
            hidden_size=hidden_size,
            num_classes=num_classes,
            fusion_method=fusion_method,
            dropout_rate=dropout_rate
        )
        
        # Loss functions
        self.entity_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.relation_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.main_loss_fn = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        entity_labels: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of GDM-Net.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            query: Query vector [batch_size, hidden_size] or [hidden_size]
            token_type_ids: Token type IDs [batch_size, seq_len]
            entity_labels: Entity labels for training [batch_size, seq_len]
            relation_labels: Relation labels for training [batch_size, seq_len, seq_len]
            return_intermediate: Whether to return intermediate results
            
        Returns:
            logits: Output logits [batch_size, num_classes] or dict with intermediate results
        """
        batch_size = input_ids.size(0)
        
        # 1. Document Encoding
        sequence_output, pooled_output = self.encoder(
            input_ids, attention_mask, token_type_ids
        )
        
        # 2. Structure Extraction
        entity_logits, relation_logits, extraction_info = self.structure_extractor(
            sequence_output, attention_mask
        )
        
        # Extract entities and relations from logits
        entities_batch = self.structure_extractor.extract_entities(
            entity_logits, attention_mask, threshold=0.5
        )
        relations_batch = self.structure_extractor.extract_relations(
            relation_logits, entities_batch, threshold=0.5
        )
        
        # 3. Graph Construction and Memory Update
        node_features, edge_index, edge_type, batch_indices = self.graph_writer(
            entities_batch, relations_batch, sequence_output, batch_size
        )
        
        # Update graph memory if nodes exist
        if node_features.size(0) > 0:
            updated_node_features = self.graph_memory(
                node_features, edge_index, edge_type, batch_indices
            )
        else:
            updated_node_features = node_features
        
        # 4. Query Processing
        if query is None:
            # Use pooled document representation as query
            query = pooled_output
        elif query.dtype == torch.long:
            # Query is token IDs, need to encode it
            query_mask = (query != 0).float()  # Simple mask for non-zero tokens
            query_output, query_pooled = self.encoder(query, query_mask)
            query = query_pooled
        elif query.dim() == 1:
            query = query.unsqueeze(0).expand(batch_size, -1)

        # Ensure query has correct batch size
        if query.size(0) != batch_size:
            if query.size(0) == 1:
                query = query.expand(batch_size, -1)
            else:
                # Use pooled output as fallback
                query = pooled_output
        
        # 5. Path Finding and Graph Reading
        if updated_node_features.size(0) > 0:
            # Path finding
            path_repr, path_info = self.path_finder(
                updated_node_features, edge_index, edge_type, query, batch_indices
            )
            
            # Graph reading
            graph_repr, attention_weights = self.graph_reader(
                updated_node_features, query, batch_indices
            )
            
            # Handle batch dimension for path representation
            if path_repr.dim() == 1:
                path_repr = path_repr.unsqueeze(0).expand(batch_size, -1)
            if graph_repr.dim() == 1:
                graph_repr = graph_repr.unsqueeze(0).expand(batch_size, -1)
        else:
            # No graph nodes, use zero representations
            path_repr = torch.zeros(batch_size, self.hparams.hidden_size, device=input_ids.device)
            graph_repr = torch.zeros(batch_size, self.hparams.hidden_size, device=input_ids.device)
            attention_weights = None
            path_info = {'num_paths': 0}
        
        # 6. Reasoning Fusion
        doc_repr = pooled_output
        logits = self.reasoning_fusion(doc_repr, graph_repr, path_repr)
        
        if return_intermediate:
            return {
                'logits': logits,
                'entity_logits': entity_logits,
                'relation_logits': relation_logits,
                'sequence_output': sequence_output,
                'pooled_output': pooled_output,
                'node_features': updated_node_features,
                'edge_index': edge_index,
                'edge_type': edge_type,
                'path_representation': path_repr,
                'graph_representation': graph_repr,
                'attention_weights': attention_weights,
                'path_info': path_info,
                'entities': entities_batch,
                'relations': relations_batch
            }
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Extract batch data
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Optional components
        query = batch.get('query', None)
        entity_labels = batch.get('entity_labels', None)
        relation_labels = batch.get('relation_labels', None)
        
        # Forward pass
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query=query,
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            return_intermediate=True
        )
        
        # Main classification loss
        main_loss = self.main_loss_fn(outputs['logits'], labels)
        total_loss = main_loss
        
        # Auxiliary losses
        if entity_labels is not None:
            entity_loss = self.entity_loss_fn(
                outputs['entity_logits'].view(-1, self.hparams.num_entities),
                entity_labels.view(-1)
            )
            total_loss += 0.1 * entity_loss
            self.log('train_entity_loss', entity_loss, prog_bar=True)
        
        if relation_labels is not None:
            relation_loss = self.relation_loss_fn(
                outputs['relation_logits'].view(-1, self.hparams.num_relations),
                relation_labels.view(-1)
            )
            total_loss += 0.1 * relation_loss
            self.log('train_relation_loss', relation_loss, prog_bar=True)
        
        # Compute accuracy
        preds = torch.argmax(outputs['logits'], dim=-1)
        self.train_accuracy(preds, labels)
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_main_loss', main_loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Extract batch data
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        query = batch.get('query', None)
        
        # Forward pass
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query=query
        )
        
        # Compute loss and accuracy
        loss = self.main_loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        self.val_accuracy(preds, labels)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Ensure learning rate is a float
        lr = float(self.hparams.learning_rate) if isinstance(self.hparams.learning_rate, str) else self.hparams.learning_rate

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        query = batch.get('query', None)
        
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query=query
        )
        
        predictions = torch.argmax(logits, dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'logits': logits
        }
