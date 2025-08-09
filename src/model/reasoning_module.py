import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from collections import deque
from .graph_sampler import AdaptiveGraphSampler


class PathFinder(nn.Module):
    """Multi-hop path discovery in graph based on query."""

    def __init__(
        self,
        hidden_size: int = 768,
        max_hops: int = 4,
        max_paths: int = 8,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_hops = min(max_hops, 2)  # 还原到2跳
        self.max_paths = min(max_paths, 4)  # 还原到4条路径

        # 🚀 图采样器用于大图优化
        self.graph_sampler = AdaptiveGraphSampler(
            max_nodes=200,
            max_edges=400
        )

        # Query-node similarity computation
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.node_projection = nn.Linear(hidden_size, hidden_size)

        # Path aggregation - 修复维度问题
        self.path_aggregator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 直接使用hidden_size而不是乘以max_hops
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Find reasoning paths from query-relevant nodes.

        Args:
            query: [batch_size, hidden_size]
            node_features: [total_nodes, hidden_size]
            edge_index: [2, total_edges]
            batch_indices: [total_nodes]

        Returns:
            path_representation: [batch_size, hidden_size]
        """
        batch_size = query.size(0)
        device = query.device

        # 🚀 图采样优化（对于大图）
        if node_features.size(0) > 200:
            # 使用自适应采样减少计算复杂度
            sampled_node_features, sampled_edge_index, _, sampled_node_mapping = \
                self.graph_sampler.adaptive_sampling(
                    node_features, edge_index, torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
                )

            # 更新batch_indices以匹配采样后的节点
            sampled_batch_indices = torch.zeros(sampled_node_features.size(0), dtype=torch.long, device=device)
            for new_idx, old_idx in enumerate(sampled_node_mapping):
                if old_idx < batch_indices.size(0):
                    sampled_batch_indices[new_idx] = batch_indices[old_idx]

            # 使用采样后的图进行推理
            working_node_features = sampled_node_features
            working_edge_index = sampled_edge_index
            working_batch_indices = sampled_batch_indices
        else:
            # 图足够小，直接使用原图
            working_node_features = node_features
            working_edge_index = edge_index
            working_batch_indices = batch_indices

        # Project query and nodes for similarity computation
        query_proj = self.query_projection(query)  # [batch_size, hidden_size]
        node_proj = self.node_projection(working_node_features)  # [sampled_nodes, hidden_size]

        path_representations = []

        for b in range(batch_size):
            # Get nodes for this batch (使用采样后的batch_indices)
            batch_mask = working_batch_indices == b
            batch_nodes = node_proj[batch_mask]  # [num_nodes_b, hidden_size]
            batch_node_indices = torch.where(batch_mask)[0]

            if batch_nodes.size(0) == 0:
                # No nodes in this batch, use zero representation
                path_representations.append(torch.zeros(self.hidden_size, device=device))
                continue

            # Compute similarity between query and nodes
            similarities = F.cosine_similarity(
                query_proj[b].unsqueeze(0),  # [1, hidden_size]
                batch_nodes,  # [num_nodes_b, hidden_size]
                dim=1
            )

            # Find starting nodes (top-k most similar)
            start_nodes = similarities.topk(min(3, batch_nodes.size(0))).indices

            # Perform BFS from each starting node
            all_paths = []
            for start_idx in start_nodes:
                start_node = batch_node_indices[start_idx]
                paths = self._bfs_paths(start_node, edge_index, batch_indices, b, node_features)
                all_paths.extend(paths[:self.max_paths // len(start_nodes)])

            # Aggregate paths
            if all_paths:
                path_repr = self._aggregate_paths(all_paths, node_features)
            else:
                path_repr = batch_nodes.mean(dim=0)  # Fallback to mean node representation

            path_representations.append(path_repr)

        path_representations = torch.stack(path_representations)
        path_representations = self.layer_norm(self.dropout(path_representations))

        return path_representations

    def _bfs_paths(
        self,
        start_node: int,
        edge_index: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_id: int,
        node_features: torch.Tensor
    ) -> List[List[int]]:
        """Perform BFS to find paths from start_node."""

        # Build adjacency list for this batch
        batch_mask = batch_indices == batch_id
        batch_node_indices = torch.where(batch_mask)[0]

        # Create mapping from global to local indices
        global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(batch_node_indices)}

        if start_node.item() not in global_to_local:
            return []

        # Build adjacency list
        adj_list = {i: [] for i in range(len(batch_node_indices))}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in global_to_local and dst in global_to_local:
                src_local = global_to_local[src]
                dst_local = global_to_local[dst]
                adj_list[src_local].append(dst_local)

        # BFS
        start_local = global_to_local[start_node.item()]
        queue = deque([(start_local, [start_local])])
        paths = []
        visited = set()

        while queue and len(paths) < self.max_paths:
            node, path = queue.popleft()

            if len(path) > self.max_hops:
                continue

            if len(path) > 1:  # Don't include single-node paths
                paths.append([batch_node_indices[n].item() for n in path])

            for neighbor in adj_list[node]:
                if neighbor not in visited or len(path) < 2:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)

        return paths

    def _aggregate_paths(self, paths: List[List[int]], node_features: torch.Tensor) -> torch.Tensor:
        """Aggregate multiple paths into a single representation."""

        if not paths:
            return torch.zeros(self.hidden_size, device=node_features.device)

        path_reprs = []
        for path in paths:
            if len(path) == 0:  # 跳过空路径
                continue

            # Get node features for this path
            path_features = node_features[path]  # [path_length, hidden_size]

            # Pad or truncate to max_hops
            if len(path) < self.max_hops:
                padding = torch.zeros(
                    self.max_hops - len(path),
                    self.hidden_size,
                    device=node_features.device
                )
                path_features = torch.cat([path_features, padding], dim=0)
            else:
                path_features = path_features[:self.max_hops]

            # 使用平均池化而不是flatten，保持hidden_size维度
            path_repr = torch.mean(path_features, dim=0)  # [hidden_size]
            path_reprs.append(path_repr)

        if not path_reprs:  # 如果没有有效路径
            return torch.zeros(self.hidden_size, device=node_features.device)

        # Average all path representations
        path_reprs = torch.stack(path_reprs)  # [num_paths, hidden_size]
        aggregated = path_reprs.mean(dim=0)   # [hidden_size]

        # Project to hidden size
        return self.path_aggregator(aggregated)


class GraphReader(nn.Module):
    """Read graph information based on query using attention."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Multi-head attention for graph reading
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Query projection
        self.query_projection = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        node_features: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Read graph information based on query.

        Args:
            query: [batch_size, hidden_size]
            node_features: [total_nodes, hidden_size]
            batch_indices: [total_nodes]

        Returns:
            graph_representation: [batch_size, hidden_size]
        """
        batch_size = query.size(0)
        device = query.device

        # Project query
        query_proj = self.query_projection(query)  # [batch_size, hidden_size]

        graph_representations = []

        for b in range(batch_size):
            # Get nodes for this batch
            batch_mask = batch_indices == b
            batch_nodes = node_features[batch_mask]  # [num_nodes_b, hidden_size]

            if batch_nodes.size(0) == 0:
                # No nodes in this batch
                graph_representations.append(torch.zeros(self.hidden_size, device=device))
                continue

            # Apply attention
            query_input = query_proj[b].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            node_input = batch_nodes.unsqueeze(0)  # [1, num_nodes_b, hidden_size]

            attended_output, attention_weights = self.attention(
                query_input,  # query
                node_input,   # key
                node_input    # value
            )

            graph_repr = attended_output.squeeze(0).squeeze(0)  # [hidden_size]
            graph_representations.append(graph_repr)

        graph_representations = torch.stack(graph_representations)
        graph_representations = self.layer_norm(self.dropout(graph_representations))

        return graph_representations


class ReasoningFusion(nn.Module):
    """Fuse different types of representations."""

    def __init__(
        self,
        hidden_size: int = 768,
        fusion_method: str = "gate",
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.fusion_method = fusion_method

        if fusion_method == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.Sigmoid()
            )
            self.fusion_projection = nn.Linear(hidden_size * 3, hidden_size)
        elif fusion_method == "attention":
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads=8, dropout=dropout_rate, batch_first=True
            )
        elif fusion_method == "concat":
            self.fusion_projection = nn.Linear(hidden_size * 3, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        doc_repr: torch.Tensor,
        graph_repr: torch.Tensor,
        path_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse document, graph, and path representations.

        Args:
            doc_repr: [batch_size, hidden_size]
            graph_repr: [batch_size, hidden_size]
            path_repr: [batch_size, hidden_size]

        Returns:
            fused_representation: [batch_size, hidden_size]
        """

        # Ensure all inputs have the same dimension
        if doc_repr.size(-1) != self.hidden_size:
            print(f"WARNING: doc_repr dimension mismatch! Expected {self.hidden_size}, got {doc_repr.size(-1)}")
        if graph_repr.size(-1) != self.hidden_size:
            print(f"WARNING: graph_repr dimension mismatch! Expected {self.hidden_size}, got {graph_repr.size(-1)}")
        if path_repr.size(-1) != self.hidden_size:
            print(f"WARNING: path_repr dimension mismatch! Expected {self.hidden_size}, got {path_repr.size(-1)}")

        if self.fusion_method == "gate":
            # Gated fusion
            combined = torch.cat([doc_repr, graph_repr, path_repr], dim=1)
            gate = self.gate(combined)

            # Weighted combination
            fused = gate * doc_repr + (1 - gate) * 0.5 * (graph_repr + path_repr)

        elif self.fusion_method == "attention":
            # Attention-based fusion
            representations = torch.stack([doc_repr, graph_repr, path_repr], dim=1)
            fused, _ = self.attention(
                doc_repr.unsqueeze(1),  # query
                representations,        # key
                representations         # value
            )
            fused = fused.squeeze(1)

        elif self.fusion_method == "concat":
            # Concatenation and projection
            combined = torch.cat([doc_repr, graph_repr, path_repr], dim=1)
            fused = self.fusion_projection(combined)

        else:  # "sum"
            # Simple summation
            fused = doc_repr + graph_repr + path_repr

        # Ensure output has correct dimension
        if fused.size(-1) != self.hidden_size:
            print(f"ERROR: fused dimension mismatch! Expected {self.hidden_size}, got {fused.size(-1)}")
            # Emergency fix
            if not hasattr(self, 'emergency_proj'):
                self.emergency_proj = nn.Linear(fused.size(-1), self.hidden_size).to(fused.device)
            fused = self.emergency_proj(fused)

        fused = self.layer_norm(self.dropout(fused))

        return fused


class ReasoningModule(nn.Module):
    """Complete reasoning module combining PathFinder, GraphReader, and ReasoningFusion."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        max_hops: int = 4,
        max_paths: int = 8,
        fusion_method: str = "gate",
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Initialize sub-modules
        self.path_finder = PathFinder(
            hidden_size=hidden_size,
            max_hops=max_hops,
            max_paths=max_paths,
            dropout_rate=dropout_rate
        )

        self.graph_reader = GraphReader(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )

        self.reasoning_fusion = ReasoningFusion(
            hidden_size=hidden_size,
            fusion_method=fusion_method,
            dropout_rate=dropout_rate
        )
    
    def forward(
        self,
        query: torch.Tensor,
        doc_representation: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform reasoning using PathFinder and GraphReader.

        Args:
            query: [batch_size, hidden_size]
            doc_representation: [batch_size, hidden_size]
            node_features: [total_nodes, hidden_size]
            edge_index: [2, total_edges]
            edge_type: [total_edges]
            batch_indices: [total_nodes]

        Returns:
            fused_representation: [batch_size, hidden_size]
            path_representation: [batch_size, hidden_size]
            graph_representation: [batch_size, hidden_size]
        """

        # Path-based reasoning
        path_representation = self.path_finder(
            query, node_features, edge_index, batch_indices
        )

        # Graph-based reading
        graph_representation = self.graph_reader(
            query, node_features, batch_indices
        )

        # Fuse all representations
        fused_representation = self.reasoning_fusion(
            doc_representation, graph_representation, path_representation
        )

        return fused_representation, path_representation, graph_representation
    
    def compute_reasoning_attention(
        self,
        query: torch.Tensor,
        entities: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention between query and entities."""
        
        # Compute attention scores
        scores = torch.matmul(query.unsqueeze(1), entities.transpose(1, 2))
        scores = scores.squeeze(1)  # [batch_size, num_entities]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute attended representation
        attended = torch.matmul(attention_weights.unsqueeze(1), entities)
        attended = attended.squeeze(1)  # [batch_size, hidden_size]
        
        return attended, attention_weights
    
    def get_reasoning_path(
        self,
        hop_representations: List[torch.Tensor],
        entity_representations: torch.Tensor
    ) -> List[int]:
        """Extract reasoning path from hop representations."""
        
        reasoning_path = []
        
        for hop_repr in hop_representations:
            # Find most similar entity for each hop
            similarities = F.cosine_similarity(
                hop_repr.unsqueeze(1),  # [batch_size, 1, hidden_size]
                entity_representations,  # [batch_size, num_entities, hidden_size]
                dim=2
            )
            
            # Get most similar entity index
            most_similar = similarities.argmax(dim=1)
            reasoning_path.append(most_similar.item())
        
        return reasoning_path
