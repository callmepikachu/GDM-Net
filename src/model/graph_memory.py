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
                dummy_repr = sequence_output[b, 0]
                # Ensure dummy representation has correct dimension
                if dummy_repr.size(0) != self.hidden_size:
                    if dummy_repr.size(0) < self.hidden_size:
                        padding = torch.zeros(self.hidden_size - dummy_repr.size(0), device=device)
                        dummy_repr = torch.cat([dummy_repr, padding], dim=0)
                    else:
                        dummy_repr = dummy_repr[:self.hidden_size]
                entities = [{'span': (0, 1), 'type': 0, 'representation': dummy_repr}]

            # Create node features
            batch_node_features = []
            valid_entities = 0  # 🔧 跟踪实际创建的节点数量

            for i in range(num_entities):
                entity = entities[i] if i < len(entities) else entities[0]

                try:
                    # Text representation
                    text_repr = entity['representation']

                    # Ensure text_repr is the correct size
                    if text_repr.dim() == 0:
                        text_repr = text_repr.unsqueeze(0)
                    if text_repr.size(0) != self.hidden_size:
                        # Pad or truncate to hidden_size
                        if text_repr.size(0) < self.hidden_size:
                            padding = torch.zeros(self.hidden_size - text_repr.size(0), device=device)
                            text_repr = torch.cat([text_repr, padding], dim=0)
                        else:
                            text_repr = text_repr[:self.hidden_size]

                    # Entity type embedding
                    entity_type = int(entity['type']) if isinstance(entity['type'], (int, float)) else 0
                    entity_type = torch.tensor(entity_type, device=device, dtype=torch.long)
                    type_repr = self.entity_type_embedding(entity_type)

                    # Position embedding with additional safety checks
                    position = int(entity['span'][0]) if len(entity['span']) > 0 else 0
                    # 🔧 确保position在有效范围内 (双重检查)
                    if position >= 512:
                        print(f"⚠️ Entity position {position} still out of range after StructureExtractor filtering")
                        position = 511  # 强制设为最大有效位置
                    position = max(0, min(position, 511))  # Clamp to [0, 511]
                    position = torch.tensor(position, device=device, dtype=torch.long)
                    pos_repr = self.position_embedding(position)

                    # Combine features - ensure all have same dimension
                    combined_repr = torch.cat([text_repr, type_repr, pos_repr], dim=0)
                    node_feature = self.node_projection(combined_repr)
                    batch_node_features.append(node_feature)
                    valid_entities += 1  # 🔧 成功创建节点，计数+1

                except Exception as e:
                    print(f"⚠️ Failed to create node for entity {i}: {e}")
                    # 跳过这个实体，不创建节点
                    continue

            # 🔧 处理节点特征和计算实际节点数
            if batch_node_features:
                batch_node_features = torch.stack(batch_node_features)
                batch_node_features = self.layer_norm(batch_node_features)
                all_node_features.append(batch_node_features)
                # 🔧 使用实际创建的节点数量
                actual_nodes_created = batch_node_features.size(0)
            else:
                # 如果没有有效节点，创建一个dummy节点
                dummy_feature = torch.zeros(1, self.hidden_size, device=device)
                all_node_features.append(dummy_feature)
                actual_nodes_created = 1

            # Create edges - 使用actual_nodes_created
            batch_edge_indices = []
            batch_edge_types = []

            for relation in relations:
                head = int(relation['head'])
                tail = int(relation['tail'])
                rel_type = int(relation['type']) if isinstance(relation['type'], (int, float)) else 1

                # 🔧 使用actual_nodes_created确保边索引有效
                if head < actual_nodes_created and tail < actual_nodes_created:
                    # Add forward edge
                    batch_edge_indices.append([head + node_offset, tail + node_offset])
                    batch_edge_types.append(rel_type)

                    # Add backward edge (bidirectional)
                    batch_edge_indices.append([tail + node_offset, head + node_offset])
                    batch_edge_types.append(rel_type)

            # Add self-loops - 使用actual_nodes_created
            for i in range(actual_nodes_created):
                batch_edge_indices.append([i + node_offset, i + node_offset])
                batch_edge_types.append(0)  # Self-loop type

            if batch_edge_indices:
                all_edge_indices.extend(batch_edge_indices)
                all_edge_types.extend(batch_edge_types)

            # 🔧 Batch indices - 使用实际创建的节点数量
            batch_indices = [b] * actual_nodes_created
            all_batch_indices.extend(batch_indices)

            node_offset += actual_nodes_created

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

        # 🔍 调试信息：验证张量形状匹配
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1

        if self._debug_count <= 3:
            print(f"🔧 GraphWriter Debug {self._debug_count}:")
            print(f"  - node_features.shape: {node_features.shape}")
            print(f"  - batch_indices.shape: {batch_indices.shape}")
            print(f"  - Shapes match: {node_features.size(0) == batch_indices.size(0)}")
            if node_features.size(0) != batch_indices.size(0):
                print(f"  ❌ MISMATCH: {node_features.size(0)} nodes vs {batch_indices.size(0)} indices")

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
