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

            # 🔧 创建节点特征 - 只处理有效实体
            batch_node_features = []
            valid_entity_mapping = {}  # 原始索引 -> 新索引的映射
            seq_len = sequence_output.size(1)

            # 处理空实体列表的情况
            if not entities:
                # 创建一个dummy实体
                dummy_repr = sequence_output[b, 0]  # 使用[CLS] token
                if dummy_repr.size(0) != self.hidden_size:
                    if dummy_repr.size(0) < self.hidden_size:
                        padding = torch.zeros(self.hidden_size - dummy_repr.size(0), device=device)
                        dummy_repr = torch.cat([dummy_repr, padding], dim=0)
                    else:
                        dummy_repr = dummy_repr[:self.hidden_size]
                entities = [{'span': (0, 1), 'type': 0, 'representation': dummy_repr}]

            # 🔧 逐个处理实体，严格检查边界
            for i, entity in enumerate(entities):
                try:
                    # 获取实体位置信息
                    start_pos = entity['span'][0] if 'span' in entity and len(entity['span']) > 0 else 0
                    end_pos = entity['span'][1] if 'span' in entity and len(entity['span']) > 1 else start_pos + 1

                    # 🔧 严格的边界检查
                    if start_pos >= seq_len:
                        continue  # 跳过越界实体

                    if end_pos > seq_len:
                        end_pos = seq_len

                    if start_pos >= end_pos:
                        continue  # 跳过无效span

                    # Text representation
                    text_repr = entity['representation']

                    # Ensure text_repr is the correct size
                    if text_repr.dim() == 0:
                        text_repr = text_repr.unsqueeze(0)
                    if text_repr.size(0) != self.hidden_size:
                        if text_repr.size(0) < self.hidden_size:
                            padding = torch.zeros(self.hidden_size - text_repr.size(0), device=device)
                            text_repr = torch.cat([text_repr, padding], dim=0)
                        else:
                            text_repr = text_repr[:self.hidden_size]

                    # Entity type embedding
                    entity_type = int(entity['type']) if isinstance(entity['type'], (int, float)) else 0
                    entity_type = torch.tensor(entity_type, device=device, dtype=torch.long)
                    type_repr = self.entity_type_embedding(entity_type)

                    # Position embedding - 使用检查后的start_pos
                    position = max(0, min(start_pos, 511))
                    position = torch.tensor(position, device=device, dtype=torch.long)
                    pos_repr = self.position_embedding(position)

                    # Combine features
                    combined_repr = torch.cat([text_repr, type_repr, pos_repr], dim=0)
                    node_feature = self.node_projection(combined_repr)

                    # 🔧 成功创建节点，记录映射
                    valid_entity_mapping[i] = len(batch_node_features)
                    batch_node_features.append(node_feature)

                except Exception as e:
                    continue

            # 🔧 处理节点特征 - 确保至少有一个节点
            if not batch_node_features:
                # 如果没有有效节点，创建一个dummy节点
                dummy_feature = torch.zeros(self.hidden_size, device=device)
                batch_node_features.append(dummy_feature)
                valid_entity_mapping[0] = 0  # dummy实体的映射
                print(f"  Created dummy node for batch {b}")

            # Stack节点特征
            batch_node_features = torch.stack(batch_node_features)
            batch_node_features = self.layer_norm(batch_node_features)
            actual_nodes_created = batch_node_features.size(0)

            # 添加到总列表
            all_node_features.append(batch_node_features)

            # 🔧 创建边 - 使用valid_entity_mapping确保索引正确
            batch_edge_indices = []
            batch_edge_types = []
            valid_relations = 0

            for relation in relations:
                head = int(relation['head'])
                tail = int(relation['tail'])
                rel_type = int(relation['type']) if isinstance(relation['type'], (int, float)) else 1

                # 🔧 检查head和tail是否在valid_entity_mapping中
                if head in valid_entity_mapping and tail in valid_entity_mapping:
                    # 使用映射后的索引
                    mapped_head = valid_entity_mapping[head]
                    mapped_tail = valid_entity_mapping[tail]

                    # 添加边（使用全局偏移）
                    batch_edge_indices.append([mapped_head + node_offset, mapped_tail + node_offset])
                    batch_edge_types.append(rel_type)

                    # 添加反向边
                    batch_edge_indices.append([mapped_tail + node_offset, mapped_head + node_offset])
                    batch_edge_types.append(rel_type)

                    valid_relations += 1

            # 添加自环
            for i in range(actual_nodes_created):
                batch_edge_indices.append([i + node_offset, i + node_offset])
                batch_edge_types.append(0)  # Self-loop type

            if batch_edge_indices:
                all_edge_indices.extend(batch_edge_indices)
                all_edge_types.extend(batch_edge_types)

            # 🔧 创建batch_indices - 长度必须等于actual_nodes_created
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

        # 🔍 最终验证和调试信息
        total_nodes = node_features.size(0)
        total_indices = batch_indices.size(0)

        print(f"🔧 GraphWriter Final Summary:")
        print(f"  - all_node_features list length: {len(all_node_features)}")
        print(f"  - all_batch_indices list length: {len(all_batch_indices)}")
        print(f"  - Total nodes created: {total_nodes}")
        print(f"  - Total batch indices: {total_indices}")
        print(f"  - Shapes match: {total_nodes == total_indices}")

        # 详细分析不匹配的原因
        if total_nodes != total_indices:
            print(f"  ❌ CRITICAL MISMATCH: {total_nodes} nodes vs {total_indices} indices")
            print(f"  - Node features per batch: {[nf.size(0) for nf in all_node_features]}")
            print(f"  - Expected total from node features: {sum(nf.size(0) for nf in all_node_features)}")
            print(f"  - Actual batch_indices length: {len(all_batch_indices)}")

            # 强制修复不匹配
            if total_indices > total_nodes:
                batch_indices = batch_indices[:total_nodes]
                print(f"  🔧 Truncated batch_indices to {total_nodes}")
            elif total_nodes > total_indices:
                # 用最后一个batch值填充
                last_batch = batch_indices[-1] if len(batch_indices) > 0 else 0
                padding = torch.full((total_nodes - total_indices,), last_batch, device=device, dtype=torch.long)
                batch_indices = torch.cat([batch_indices, padding])
                print(f"  🔧 Padded batch_indices to {total_nodes}")
        else:
            print(f"  ✅ Perfect match: {total_nodes} nodes = {total_indices} indices")

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
