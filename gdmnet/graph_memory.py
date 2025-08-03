"""
Graph Memory Module

This module implements GraphMemory and GraphWriter classes for maintaining 
and updating graph-structured memory using Graph Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List, Dict, Union
import math


class GraphMemory(nn.Module):
    """
    Graph memory module using Graph Neural Networks (RGCN/GAT) to maintain graph structure.
    
    Args:
        node_dim (int): Dimension of node features
        edge_dim (int): Dimension of edge features
        num_relations (int): Number of relation types
        num_layers (int): Number of GNN layers
        gnn_type (str): Type of GNN ('rgcn' or 'gat')
        dropout_rate (float): Dropout rate
        use_residual (bool): Whether to use residual connections
    """
    
    def __init__(
        self,
        node_dim: int = 768,
        edge_dim: Optional[int] = None,
        num_relations: int = 10,
        num_layers: int = 2,
        gnn_type: str = 'rgcn',
        dropout_rate: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.use_residual = use_residual
        
        # Build GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if self.gnn_type == 'rgcn':
                layer = RGCNConv(
                    in_channels=node_dim,
                    out_channels=node_dim,
                    num_relations=num_relations,
                    aggr='mean'
                )
            elif self.gnn_type == 'gat':
                layer = GATConv(
                    in_channels=node_dim,
                    out_channels=node_dim // 8,  # 8 attention heads
                    heads=8,
                    dropout=dropout_rate,
                    concat=True
                )
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
            self.gnn_layers.append(layer)
        
        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        
        # Node update gate for memory management
        self.update_gate = nn.Linear(node_dim * 2, node_dim)
        self.reset_gate = nn.Linear(node_dim * 2, node_dim)
        self.new_gate = nn.Linear(node_dim * 2, node_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the graph memory.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges] (for RGCN)
            batch: Batch indices [num_nodes] (for batched graphs)
            
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        h = x
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_prev = h
            
            # Apply GNN layer
            if self.gnn_type == 'rgcn':
                h = gnn_layer(h, edge_index, edge_type)
            else:  # GAT
                h = gnn_layer(h, edge_index)
            
            # Apply layer normalization
            h = self.layer_norms[i](h)
            
            # Apply residual connection
            if self.use_residual and i > 0:
                h = h + h_prev
            
            # Apply dropout
            h = self.dropout(h)
            
            # Apply activation (except for the last layer)
            if i < len(self.gnn_layers) - 1:
                h = F.relu(h)
        
        return h
    
    def update_memory(
        self, 
        current_memory: torch.Tensor, 
        new_information: torch.Tensor
    ) -> torch.Tensor:
        """
        Update graph memory with new information using gating mechanism.
        
        Args:
            current_memory: Current memory state [num_nodes, node_dim]
            new_information: New information to integrate [num_nodes, node_dim]
            
        Returns:
            Updated memory state [num_nodes, node_dim]
        """
        # Concatenate current memory and new information
        combined = torch.cat([current_memory, new_information], dim=-1)
        
        # Compute gates
        update_gate = torch.sigmoid(self.update_gate(combined))
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        
        # Compute new candidate values
        reset_memory = reset_gate * current_memory
        new_candidate = torch.tanh(self.new_gate(
            torch.cat([reset_memory, new_information], dim=-1)
        ))
        
        # Update memory
        updated_memory = (1 - update_gate) * current_memory + update_gate * new_candidate
        
        return updated_memory


class GraphWriter(nn.Module):
    """
    Graph writer module for converting extracted structures into graph representations.
    
    Args:
        hidden_size (int): Hidden size of input representations
        node_dim (int): Dimension of output node features
        edge_dim (int): Dimension of output edge features
        num_relations (int): Number of relation types
        max_nodes (int): Maximum number of nodes per graph
    """
    
    def __init__(
        self,
        hidden_size: int,
        node_dim: int = 768,
        edge_dim: Optional[int] = None,
        num_relations: int = 10,
        max_nodes: int = 512
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_relations = num_relations
        self.max_nodes = max_nodes
        
        # Node feature projection
        self.node_projector = nn.Linear(hidden_size, node_dim)
        
        # Edge feature projection (if edge features are used)
        if edge_dim:
            self.edge_projector = nn.Linear(hidden_size, edge_dim)
        
        # Entity type embedding
        self.entity_type_embedding = nn.Embedding(100, node_dim // 4)  # Support up to 100 entity types
        
        # Relation type embedding
        self.relation_type_embedding = nn.Embedding(num_relations, node_dim // 4)
        
        # Position encoding for nodes
        self.position_encoding = nn.Parameter(torch.randn(max_nodes, node_dim // 4))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(node_dim)
        
    def forward(
        self,
        entities: List[List[Dict]],
        relations: List[List[Dict]],
        entity_representations: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert extracted entities and relations into graph representation.
        
        Args:
            entities: List of entity lists for each batch item
            relations: List of relation lists for each batch item
            entity_representations: Entity representations [batch_size, seq_len, hidden_size]
            batch_size: Batch size
            
        Returns:
            node_features: Node features [total_nodes, node_dim]
            edge_index: Edge indices [2, total_edges]
            edge_type: Edge types [total_edges]
            batch_indices: Batch indices for each node [total_nodes]
        """
        all_node_features = []
        all_edge_indices = []
        all_edge_types = []
        all_batch_indices = []
        
        node_offset = 0
        
        for b in range(batch_size):
            batch_entities = entities[b]
            batch_relations = relations[b]
            
            if not batch_entities:
                # Create a dummy node if no entities
                dummy_node = torch.zeros(1, self.node_dim)
                all_node_features.append(dummy_node)
                device = next(self.parameters()).device
                all_batch_indices.append(torch.tensor([b], device=device))
                node_offset += 1
                continue
            
            # Create node features
            node_features = self._create_node_features(
                batch_entities, entity_representations[b], b
            )
            all_node_features.append(node_features)
            
            # Create edges
            if batch_relations:
                edge_index, edge_type = self._create_edges(batch_relations, node_offset)
                all_edge_indices.append(edge_index)
                all_edge_types.append(edge_type)
            
            # Create batch indices
            num_nodes = len(batch_entities)
            batch_indices = torch.full((num_nodes,), b, dtype=torch.long)
            all_batch_indices.append(batch_indices)
            
            node_offset += num_nodes
        
        # Concatenate all features
        node_features = torch.cat(all_node_features, dim=0) if all_node_features else torch.empty(0, self.node_dim)
        edge_index = torch.cat(all_edge_indices, dim=1) if all_edge_indices else torch.empty(2, 0, dtype=torch.long)
        edge_type = torch.cat(all_edge_types, dim=0) if all_edge_types else torch.empty(0, dtype=torch.long)
        batch_indices = torch.cat(all_batch_indices, dim=0) if all_batch_indices else torch.empty(0, dtype=torch.long)
        
        return node_features, edge_index, edge_type, batch_indices
    
    def _create_node_features(
        self,
        entities: List[Dict],
        entity_repr: torch.Tensor,
        batch_idx: int
    ) -> torch.Tensor:
        """Create node features from entities."""
        node_features = []
        
        for i, entity in enumerate(entities):
            # Get entity representation
            start_pos = entity['start']
            entity_vec = entity_repr[start_pos]  # [hidden_size]
            
            # Project to node dimension
            node_feat = self.node_projector(entity_vec)  # [node_dim * 3/4]
            
            # Add entity type embedding
            entity_type = entity.get('type', 0)
            # 确保张量在正确的设备上
            device = next(self.parameters()).device
            type_emb = self.entity_type_embedding(torch.tensor(entity_type, device=device))
            
            # Add position encoding
            pos_emb = self.position_encoding[i % self.max_nodes]
            
            # Combine features
            combined_feat = torch.cat([node_feat[:self.node_dim//2], type_emb, pos_emb], dim=0)
            
            # Ensure correct dimension
            if combined_feat.size(0) != self.node_dim:
                combined_feat = F.pad(combined_feat, (0, self.node_dim - combined_feat.size(0)))
            
            node_features.append(combined_feat)
        
        node_features = torch.stack(node_features)  # [num_entities, node_dim]
        node_features = self.layer_norm(node_features)
        
        return node_features
    
    def _create_edges(
        self,
        relations: List[Dict],
        node_offset: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edge indices and types from relations."""
        edge_indices = []
        edge_types = []
        
        for relation in relations:
            head_idx = relation['head'] + node_offset
            tail_idx = relation['tail'] + node_offset
            rel_type = relation['type']
            
            # Add forward edge
            edge_indices.append([head_idx, tail_idx])
            edge_types.append(rel_type)
            
            # Add backward edge (optional, for undirected relations)
            edge_indices.append([tail_idx, head_idx])
            edge_types.append(rel_type)
        
        # 确保所有张量在正确的设备上
        device = next(self.parameters()).device

        if edge_indices:
            edge_index = torch.tensor(edge_indices, device=device).t().contiguous()  # [2, num_edges]
            edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)  # [num_edges]
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_type = torch.empty(0, dtype=torch.long, device=device)
        
        return edge_index, edge_type
