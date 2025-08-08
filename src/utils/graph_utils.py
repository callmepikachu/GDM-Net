import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def build_graph(entities: List[Dict], relations: List[Dict], num_entities: int) -> Data:
    """Build PyTorch Geometric graph from entities and relations."""
    
    # Create edge index and edge attributes
    edge_index = []
    edge_attr = []
    
    for relation in relations:
        head = relation['head']
        tail = relation['tail']
        rel_type = relation['type']
        
        if head < num_entities and tail < num_entities:
            edge_index.append([head, tail])
            edge_attr.append(rel_type)
    
    if not edge_index:
        # Create empty graph with self-loops
        edge_index = [[i, i] for i in range(num_entities)]
        edge_attr = [0] * num_entities
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    # Create node features (entity types)
    node_features = torch.zeros(num_entities, dtype=torch.long)
    for i, entity in enumerate(entities[:num_entities]):
        node_features[i] = entity.get('type', 0)
    
    # Make graph undirected for better connectivity
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_entities
    )


def create_adjacency_matrix(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Create adjacency matrix from edge index."""
    adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    
    if edge_index.size(1) > 0:
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
    
    return adj_matrix


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Add self-loops to graph."""
    self_loops = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    return edge_index


def normalize_adjacency(adj_matrix: torch.Tensor) -> torch.Tensor:
    """Normalize adjacency matrix."""
    # Add self-loops
    adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
    
    # Compute degree matrix
    degree = adj_matrix.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    degree_matrix = torch.diag(degree_inv_sqrt)
    normalized_adj = torch.mm(torch.mm(degree_matrix, adj_matrix), degree_matrix)
    
    return normalized_adj


def compute_graph_statistics(data: Data) -> Dict[str, Any]:
    """Compute basic graph statistics."""
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    
    # Convert to NetworkX for advanced statistics
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    if num_edges > 0:
        edges = data.edge_index.t().numpy()
        G.add_edges_from(edges)
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': nx.density(G) if num_nodes > 1 else 0.0,
        'num_connected_components': nx.number_connected_components(G),
        'average_clustering': nx.average_clustering(G) if num_nodes > 2 else 0.0
    }
    
    return stats
