import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv, GATConv
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, Any, List, Tuple


class GraphWriter(nn.Module):
    """Convert extracted entities and relations to graph format."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_entity_types: int = 9,
        num_relation_types: int = 10,
        max_entities: int = 64
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_entity_types = num_entity_types
        self.num_relation_types = num_relation_types
        self.max_entities = max_entities

        # Entity type embeddings
        self.entity_type_embedding = nn.Embedding(num_entity_types, hidden_size)

        # Position encoding
        self.position_embedding = nn.Embedding(512, hidden_size)  # Max sequence length

        # Node feature projection
        self.node_projection = nn.Linear(hidden_size * 3, hidden_size)  # text + type + position

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        entities_batch: List[List[Dict]],
        relations_batch: List[List[Dict]],
        sequence_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert entities and relations to graph format.

        Args:
            entities_batch: List of entity lists for each batch
            relations_batch: List of relation lists for each batch
            sequence_output: [batch_size, seq_len, hidden_size]

        Returns:
            node_features: [total_nodes, hidden_size]
            edge_index: [2, total_edges]
            edge_type: [total_edges]
            batch_indices: [total_nodes]
        """
        batch_size = len(entities_batch)
        device = sequence_output.device

        all_node_features = []
        all_edge_indices = []
        all_edge_types = []
        all_batch_indices = []

        node_offset = 0

        for b in range(batch_size):
            entities = entities_batch[b]
            relations = relations_batch[b]

            # Pad entities to max_entities if needed
            num_entities = min(len(entities), self.max_entities)
            if num_entities == 0:
                # Create dummy entity if no entities found
                num_entities = 1
                entities = [{'span': (0, 1), 'type': 0, 'representation': sequence_output[b, 0]}]

            # Create node features
            batch_node_features = []
            for i in range(num_entities):
                entity = entities[i] if i < len(entities) else entities[0]

                # Text representation
                text_repr = entity['representation']

                # Entity type embedding
                entity_type = torch.tensor(entity['type'], device=device)
                type_repr = self.entity_type_embedding(entity_type)

                # Position embedding
                position = torch.tensor(entity['span'][0], device=device)
                pos_repr = self.position_embedding(position)

                # Combine features
                combined_repr = torch.cat([text_repr, type_repr, pos_repr], dim=0)
                node_feature = self.node_projection(combined_repr)
                batch_node_features.append(node_feature)

            batch_node_features = torch.stack(batch_node_features)
            batch_node_features = self.layer_norm(batch_node_features)
            all_node_features.append(batch_node_features)

            # Create edges
            batch_edge_indices = []
            batch_edge_types = []

            for relation in relations:
                head = relation['head']
                tail = relation['tail']
                rel_type = relation['type']

                if head < num_entities and tail < num_entities:
                    # Add forward edge
                    batch_edge_indices.append([head + node_offset, tail + node_offset])
                    batch_edge_types.append(rel_type)

                    # Add backward edge (bidirectional)
                    batch_edge_indices.append([tail + node_offset, head + node_offset])
                    batch_edge_types.append(rel_type)

            # Add self-loops
            for i in range(num_entities):
                batch_edge_indices.append([i + node_offset, i + node_offset])
                batch_edge_types.append(0)  # Self-loop type

            if batch_edge_indices:
                all_edge_indices.extend(batch_edge_indices)
                all_edge_types.extend(batch_edge_types)

            # Batch indices
            batch_indices = [b] * num_entities
            all_batch_indices.extend(batch_indices)

            node_offset += num_entities

        # Convert to tensors
        if all_node_features:
            node_features = torch.cat(all_node_features, dim=0)
        else:
            node_features = torch.zeros(1, self.hidden_size, device=device)

        if all_edge_indices:
            edge_index = torch.tensor(all_edge_indices, device=device).t().contiguous()
            edge_type = torch.tensor(all_edge_types, device=device)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_type = torch.zeros(0, dtype=torch.long, device=device)

        batch_indices = torch.tensor(all_batch_indices, device=device)

        return node_features, edge_index, edge_type, batch_indices


class GraphMemory(nn.Module):
    """Graph neural network for processing structured knowledge."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_relation_types: int = 10,
        gnn_type: str = "rgcn",
        num_gnn_layers: int = 3,
        dropout_rate: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_relation_types = num_relation_types
        self.gnn_type = gnn_type.lower()
        self.num_gnn_layers = num_gnn_layers
        self.use_residual = use_residual
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()

        for i in range(num_gnn_layers):
            if self.gnn_type == "rgcn":
                layer = RGCNConv(
                    hidden_size,
                    hidden_size,
                    num_relations=num_relation_types,
                    aggr='mean'
                )
            elif self.gnn_type == "gcn":
                layer = GCNConv(hidden_size, hidden_size, aggr='mean')
            elif self.gnn_type == "gat":
                layer = GATConv(
                    hidden_size,
                    hidden_size // 8,
                    heads=8,
                    dropout=dropout_rate,
                    concat=True
                )
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

            self.gnn_layers.append(layer)

        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_gnn_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through graph memory.

        Args:
            node_features: [total_nodes, hidden_size]
            edge_index: [2, total_edges]
            edge_type: [total_edges]
            batch_indices: [total_nodes]

        Returns:
            updated_node_features: [total_nodes, hidden_size]
        """

        x = node_features

        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.use_residual:
                residual = x

            if self.gnn_type == "rgcn":
                x = gnn_layer(x, edge_index, edge_type)
            else:
                x = gnn_layer(x, edge_index)

            if self.use_residual:
                x = self.layer_norms[i](x + residual)
            else:
                x = self.layer_norms[i](x)

            x = F.relu(x)
            x = self.dropout(x)

        return x
    
    def compute_graph_attention(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention weights between connected nodes."""
        
        if edge_index.size(1) == 0:
            return torch.zeros(0, device=node_features.device)
        
        # Get source and target node features
        source_features = node_features[edge_index[0]]  # [num_edges, hidden_size]
        target_features = node_features[edge_index[1]]  # [num_edges, hidden_size]
        
        # Compute attention scores
        attention_scores = torch.sum(source_features * target_features, dim=1)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        return attention_weights
