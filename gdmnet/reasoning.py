"""
Reasoning Module

This module implements PathFinder, GraphReader, and ReasoningFusion classes
for multi-hop reasoning and information fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Optional, Tuple, List, Dict, Union
import math


class PathFinder(nn.Module):
    """
    Path finder module for discovering reasoning paths in the graph memory.
    
    Args:
        node_dim (int): Dimension of node features
        query_dim (int): Dimension of query vectors
        num_hops (int): Maximum number of reasoning hops
        num_paths (int): Number of paths to consider
        attention_heads (int): Number of attention heads
    """
    
    def __init__(
        self,
        node_dim: int = 768,
        query_dim: int = 768,
        num_hops: int = 3,
        num_paths: int = 5,
        attention_heads: int = 8
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.query_dim = query_dim
        self.num_hops = num_hops
        self.num_paths = num_paths
        self.attention_heads = attention_heads
        
        # Query projection
        self.query_projector = nn.Linear(query_dim, node_dim)
        
        # Multi-head attention for path finding
        self.path_attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Path encoding layers
        self.path_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=node_dim,
                nhead=attention_heads,
                dim_feedforward=node_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Path scoring
        self.path_scorer = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(node_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        query: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Find reasoning paths in the graph.
        
        Args:
            node_features: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
            query: Query vector [query_dim] or [batch_size, query_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            path_representation: Aggregated path representation
            path_info: Dictionary containing path information
        """
        # Project query to node dimension
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, query_dim]
        query_proj = self.query_projector(query)  # [batch_size, node_dim]
        
        # Find starting nodes based on query similarity
        start_nodes = self._find_start_nodes(node_features, query_proj, batch)
        
        # Perform multi-hop reasoning
        paths = self._multi_hop_reasoning(
            node_features, edge_index, edge_type, start_nodes, batch
        )
        
        # Encode and score paths
        path_representations = []
        path_scores = []
        
        for path in paths:
            if len(path) > 0:
                path_repr = self._encode_path(path, node_features)
                path_score = self.path_scorer(path_repr.mean(dim=0))
                path_representations.append(path_repr.mean(dim=0))
                path_scores.append(path_score)
        
        if path_representations:
            # Aggregate paths using attention
            path_stack = torch.stack(path_representations)  # [num_paths, node_dim]
            path_scores = torch.stack(path_scores).squeeze(-1)  # [num_paths]
            
            # Apply softmax to get path weights
            path_weights = F.softmax(path_scores, dim=0)
            
            # Weighted aggregation
            aggregated_path = torch.sum(
                path_weights.unsqueeze(-1) * path_stack, dim=0
            )  # [node_dim]
        else:
            # Return zero vector if no paths found
            aggregated_path = torch.zeros(self.node_dim, device=node_features.device)
            path_weights = torch.empty(0, device=node_features.device)
        
        path_info = {
            'path_weights': path_weights,
            'num_paths': len(paths),
            'start_nodes': start_nodes
        }
        
        return aggregated_path, path_info
    
    def _find_start_nodes(
        self,
        node_features: torch.Tensor,
        query: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> List[int]:
        """Find starting nodes for path exploration based on query similarity."""
        # Compute similarity between query and all nodes
        similarities = torch.matmul(node_features, query.t())  # [num_nodes, batch_size]
        
        if batch is not None:
            # Handle batched graphs
            start_nodes = []
            for b in range(query.size(0)):
                batch_mask = (batch == b)
                if batch_mask.any():
                    batch_similarities = similarities[batch_mask, b]
                    _, top_indices = torch.topk(batch_similarities, min(self.num_paths, batch_mask.sum().item()))
                    batch_node_indices = torch.where(batch_mask)[0]
                    # 确保索引在同一设备上，然后移到CPU进行tolist()
                    top_indices = top_indices.to(batch_node_indices.device)
                    start_nodes.extend(batch_node_indices[top_indices].cpu().tolist())
        else:
            # Single graph
            _, top_indices = torch.topk(similarities.squeeze(-1), min(self.num_paths, node_features.size(0)))
            # 确保在CPU上进行tolist()操作
            start_nodes = top_indices.cpu().tolist()
        
        return start_nodes
    
    def _multi_hop_reasoning(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        start_nodes: List[int],
        batch: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """Perform multi-hop reasoning to find paths."""
        paths = []
        
        # Create adjacency information
        adjacency = self._create_adjacency_dict(edge_index, edge_type)
        
        for start_node in start_nodes:
            # Perform breadth-first search with limited depth
            current_paths = [[start_node]]
            
            for hop in range(self.num_hops):
                new_paths = []
                
                for path in current_paths:
                    current_node = path[-1]
                    
                    # Get neighbors
                    if current_node in adjacency:
                        neighbors = adjacency[current_node]
                        
                        for neighbor, edge_type_val in neighbors:
                            # Avoid cycles (simple check)
                            if neighbor not in path:
                                new_path = path + [neighbor]
                                new_paths.append(new_path)
                
                current_paths = new_paths[:self.num_paths]  # Limit number of paths
                
                if not current_paths:
                    break
            
            paths.extend(current_paths)
        
        return paths[:self.num_paths]  # Return top paths
    
    def _create_adjacency_dict(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Create adjacency dictionary from edge information."""
        adjacency = {}
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_t = edge_type[i].item()
            
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append((dst, edge_t))
        
        return adjacency
    
    def _encode_path(
        self,
        path: List[int],
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode a path using transformer encoder."""
        if not path:
            return torch.zeros(1, self.node_dim, device=node_features.device)
        
        # Get node features for the path
        path_features = node_features[path]  # [path_length, node_dim]
        
        # Add positional encoding
        path_length = len(path)
        pos_encoding = self._get_positional_encoding(path_length, self.node_dim)
        pos_encoding = pos_encoding.to(node_features.device)
        
        path_features = path_features + pos_encoding
        
        # Apply transformer encoder
        path_features = path_features.unsqueeze(0)  # [1, path_length, node_dim]
        encoded_path = self.path_encoder(path_features)  # [1, path_length, node_dim]
        
        return encoded_path.squeeze(0)  # [path_length, node_dim]
    
    def _get_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Generate positional encoding for path sequences."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe


class GraphReader(nn.Module):
    """
    Graph reader module for reading relevant information from graph memory.
    
    Args:
        node_dim (int): Dimension of node features
        query_dim (int): Dimension of query vectors
        attention_heads (int): Number of attention heads
        pooling_method (str): Pooling method ('mean', 'max', 'add', 'attention')
    """
    
    def __init__(
        self,
        node_dim: int = 768,
        query_dim: int = 768,
        attention_heads: int = 8,
        pooling_method: str = 'attention'
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.query_dim = query_dim
        self.attention_heads = attention_heads
        self.pooling_method = pooling_method
        
        # Query projection
        self.query_projector = nn.Linear(query_dim, node_dim)
        
        # Attention mechanism for reading
        self.read_attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_projector = nn.Linear(node_dim, node_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(node_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        query: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read information from graph memory based on query.
        
        Args:
            node_features: Node features [num_nodes, node_dim]
            query: Query vector [query_dim] or [batch_size, query_dim]
            batch: Batch indices [num_nodes]
            attention_mask: Attention mask [num_nodes]
            
        Returns:
            read_vector: Read information vector
            attention_weights: Attention weights over nodes
        """
        # Project query to node dimension
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, query_dim]
        query_proj = self.query_projector(query)  # [batch_size, node_dim]
        
        # Prepare node features for attention
        if batch is not None:
            # Handle batched graphs
            batch_size = query.size(0)
            read_vectors = []
            attention_weights_list = []
            
            for b in range(batch_size):
                batch_mask = (batch == b)
                if batch_mask.any():
                    batch_nodes = node_features[batch_mask]  # [num_batch_nodes, node_dim]
                    batch_query = query_proj[b:b+1]  # [1, node_dim]
                    
                    # Apply attention
                    read_vec, attn_weights = self.read_attention(
                        batch_query.unsqueeze(1),  # query [1, 1, node_dim]
                        batch_nodes.unsqueeze(0),  # key [1, num_batch_nodes, node_dim]
                        batch_nodes.unsqueeze(0)   # value [1, num_batch_nodes, node_dim]
                    )
                    
                    read_vectors.append(read_vec.squeeze(0).squeeze(0))  # [node_dim]
                    attention_weights_list.append(attn_weights.squeeze(0).squeeze(0))  # [num_batch_nodes]
                else:
                    # No nodes in this batch
                    read_vectors.append(torch.zeros(self.node_dim, device=node_features.device))
                    attention_weights_list.append(torch.empty(0, device=node_features.device))
            
            read_vector = torch.stack(read_vectors)  # [batch_size, node_dim]
            attention_weights = attention_weights_list  # List of tensors
        else:
            # Single graph
            if node_features.size(0) == 0:
                # No nodes case
                read_vector = torch.zeros(query_proj.size(0), self.node_dim, device=query_proj.device)
                attention_weights = torch.empty(query_proj.size(0), 0, device=query_proj.device)
            else:
                # Expand node features to match batch size
                batch_size = query_proj.size(0)
                node_features_expanded = node_features.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]

                # Apply attention
                read_vector, attention_weights = self.read_attention(
                    query_proj.unsqueeze(1),  # [batch_size, 1, node_dim]
                    node_features_expanded,   # [batch_size, num_nodes, node_dim]
                    node_features_expanded    # [batch_size, num_nodes, node_dim]
                )

                read_vector = read_vector.squeeze(1)  # [batch_size, node_dim]
                attention_weights = attention_weights.squeeze(1)  # [batch_size, num_nodes]
        
        # Apply output projection and layer normalization
        read_vector = self.output_projector(read_vector)
        read_vector = self.layer_norm(read_vector)
        
        return read_vector, attention_weights


class ReasoningFusion(nn.Module):
    """
    Reasoning fusion module for combining document and graph representations.
    
    Args:
        hidden_size (int): Hidden size of input representations
        num_classes (int): Number of output classes
        fusion_method (str): Fusion method ('concat', 'add', 'gate', 'attention')
        dropout_rate (float): Dropout rate
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        fusion_method: str = 'gate',
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)
        elif fusion_method == 'gate':
            self.gate = nn.Linear(hidden_size * 2, hidden_size)
            self.fusion_layer = nn.Linear(hidden_size, hidden_size)
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.fusion_layer = nn.Linear(hidden_size, hidden_size)
        else:  # 'add'
            self.fusion_layer = nn.Linear(hidden_size, hidden_size)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        doc_repr: torch.Tensor,
        graph_repr: torch.Tensor,
        path_repr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse document and graph representations.
        
        Args:
            doc_repr: Document representation [batch_size, hidden_size]
            graph_repr: Graph representation [batch_size, hidden_size]
            path_repr: Path representation [batch_size, hidden_size] (optional)
            
        Returns:
            logits: Output logits [batch_size, num_classes]
        """
        # Combine graph and path representations if available
        if path_repr is not None:
            graph_repr = graph_repr + path_repr
        
        # Ensure same batch size
        if doc_repr.size(0) != graph_repr.size(0):
            # Handle size mismatch
            min_batch_size = min(doc_repr.size(0), graph_repr.size(0))
            doc_repr = doc_repr[:min_batch_size]
            graph_repr = graph_repr[:min_batch_size]
        
        # Apply fusion method
        if self.fusion_method == 'concat':
            fused = torch.cat([doc_repr, graph_repr], dim=-1)
            fused = self.fusion_layer(fused)
        elif self.fusion_method == 'add':
            fused = doc_repr + graph_repr
            fused = self.fusion_layer(fused)
        elif self.fusion_method == 'gate':
            combined = torch.cat([doc_repr, graph_repr], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            fused = gate * doc_repr + (1 - gate) * graph_repr
            fused = self.fusion_layer(fused)
        elif self.fusion_method == 'attention':
            # Use attention to fuse representations
            representations = torch.stack([doc_repr, graph_repr], dim=1)  # [batch_size, 2, hidden_size]
            fused, _ = self.attention(representations, representations, representations)
            fused = fused.mean(dim=1)  # [batch_size, hidden_size]
            fused = self.fusion_layer(fused)
        
        # Apply layer normalization
        fused = self.layer_norm(fused)
        
        # Generate output logits
        logits = self.output_layers(fused)
        
        return logits
