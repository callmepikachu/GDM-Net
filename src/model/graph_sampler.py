"""
图采样器 - 用于大图的高效采样和推理优化
"""

import torch
import numpy as np
from typing import List, Tuple, Set, Dict, Any
import random


class GraphSampler:
    """
    图采样器，提供多种采样策略来处理大规模图
    """
    
    def __init__(self, max_nodes: int = 200, max_edges: int = 500):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
    
    def random_walk_sampling(self, 
                           node_features: torch.Tensor,
                           edge_index: torch.Tensor,
                           edge_type: torch.Tensor,
                           start_nodes: List[int],
                           walk_length: int = 3,
                           num_walks: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        🚀 随机游走采样
        Args:
            node_features: 节点特征 [num_nodes, hidden_size]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            start_nodes: 起始节点列表
            walk_length: 游走长度
            num_walks: 每个起始节点的游走次数
        Returns:
            采样后的子图组件
        """
        if node_features.size(0) <= self.max_nodes:
            # 图已经足够小，不需要采样
            return node_features, edge_index, edge_type, list(range(node_features.size(0)))
        
        # 构建邻接表
        adj_list = self._build_adjacency_list(edge_index, node_features.size(0))
        
        # 执行随机游走
        sampled_nodes = set()
        for start_node in start_nodes:
            if start_node < len(adj_list):
                for _ in range(num_walks):
                    walk_nodes = self._random_walk(adj_list, start_node, walk_length)
                    sampled_nodes.update(walk_nodes)
        
        # 如果采样节点太少，添加一些随机节点
        if len(sampled_nodes) < self.max_nodes // 2:
            additional_nodes = random.sample(
                range(node_features.size(0)), 
                min(self.max_nodes - len(sampled_nodes), node_features.size(0) - len(sampled_nodes))
            )
            sampled_nodes.update(additional_nodes)
        
        # 限制节点数量
        if len(sampled_nodes) > self.max_nodes:
            sampled_nodes = set(random.sample(list(sampled_nodes), self.max_nodes))
        
        return self._extract_subgraph(node_features, edge_index, edge_type, sampled_nodes)
    
    def k_hop_sampling(self,
                      node_features: torch.Tensor,
                      edge_index: torch.Tensor, 
                      edge_type: torch.Tensor,
                      center_nodes: List[int],
                      k: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        🚀 K跳邻居采样
        Args:
            node_features: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            center_nodes: 中心节点列表
            k: 跳数
        Returns:
            采样后的子图组件
        """
        if node_features.size(0) <= self.max_nodes:
            return node_features, edge_index, edge_type, list(range(node_features.size(0)))
        
        # 构建邻接表
        adj_list = self._build_adjacency_list(edge_index, node_features.size(0))
        
        # K跳邻居搜索
        sampled_nodes = set(center_nodes)
        current_nodes = set(center_nodes)
        
        for hop in range(k):
            next_nodes = set()
            for node in current_nodes:
                if node < len(adj_list):
                    neighbors = adj_list[node]
                    # 限制每个节点的邻居数量
                    if len(neighbors) > 20:
                        neighbors = random.sample(neighbors, 20)
                    next_nodes.update(neighbors)
            
            sampled_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            # 如果节点数量超过限制，提前停止
            if len(sampled_nodes) > self.max_nodes:
                break
        
        # 限制节点数量
        if len(sampled_nodes) > self.max_nodes:
            # 优先保留中心节点和它们的直接邻居
            priority_nodes = set(center_nodes)
            for center in center_nodes:
                if center < len(adj_list):
                    priority_nodes.update(adj_list[center][:10])  # 每个中心节点保留10个邻居
            
            remaining_quota = self.max_nodes - len(priority_nodes)
            other_nodes = sampled_nodes - priority_nodes
            
            if remaining_quota > 0 and other_nodes:
                additional_nodes = random.sample(list(other_nodes), min(remaining_quota, len(other_nodes)))
                priority_nodes.update(additional_nodes)
            
            sampled_nodes = priority_nodes
        
        return self._extract_subgraph(node_features, edge_index, edge_type, sampled_nodes)
    
    def importance_sampling(self,
                          node_features: torch.Tensor,
                          edge_index: torch.Tensor,
                          edge_type: torch.Tensor,
                          node_importance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        🚀 基于重要性的采样
        Args:
            node_features: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            node_importance: 节点重要性分数 [num_nodes]
        Returns:
            采样后的子图组件
        """
        if node_features.size(0) <= self.max_nodes:
            return node_features, edge_index, edge_type, list(range(node_features.size(0)))
        
        # 根据重要性分数采样
        _, top_indices = torch.topk(node_importance, min(self.max_nodes, len(node_importance)))
        sampled_nodes = set(top_indices.tolist())
        
        return self._extract_subgraph(node_features, edge_index, edge_type, sampled_nodes)
    
    def _build_adjacency_list(self, edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
        """构建邻接表"""
        adj_list = [[] for _ in range(num_nodes)]
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src < num_nodes and dst < num_nodes:
                adj_list[src].append(dst)
                adj_list[dst].append(src)  # 无向图
        
        return adj_list
    
    def _random_walk(self, adj_list: List[List[int]], start_node: int, walk_length: int) -> List[int]:
        """执行单次随机游走"""
        walk = [start_node]
        current_node = start_node
        
        for _ in range(walk_length):
            if current_node >= len(adj_list) or not adj_list[current_node]:
                break
            
            next_node = random.choice(adj_list[current_node])
            walk.append(next_node)
            current_node = next_node
        
        return walk
    
    def _extract_subgraph(self, 
                         node_features: torch.Tensor,
                         edge_index: torch.Tensor,
                         edge_type: torch.Tensor,
                         sampled_nodes: Set[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """从采样节点中提取子图"""
        sampled_nodes_list = sorted(list(sampled_nodes))
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sampled_nodes_list)}
        
        # 提取节点特征
        sampled_node_features = node_features[sampled_nodes_list]
        
        # 提取边
        sampled_edges = []
        sampled_edge_types = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in sampled_nodes and dst in sampled_nodes:
                new_src = node_mapping[src]
                new_dst = node_mapping[dst]
                sampled_edges.append([new_src, new_dst])
                sampled_edge_types.append(edge_type[i].item())
        
        # 限制边数量
        if len(sampled_edges) > self.max_edges:
            indices = random.sample(range(len(sampled_edges)), self.max_edges)
            sampled_edges = [sampled_edges[i] for i in indices]
            sampled_edge_types = [sampled_edge_types[i] for i in indices]
        
        if sampled_edges:
            sampled_edge_index = torch.tensor(sampled_edges, dtype=torch.long, device=edge_index.device).t()
            sampled_edge_type = torch.tensor(sampled_edge_types, dtype=torch.long, device=edge_type.device)
        else:
            sampled_edge_index = torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
            sampled_edge_type = torch.empty(0, dtype=torch.long, device=edge_type.device)
        
        return sampled_node_features, sampled_edge_index, sampled_edge_type, sampled_nodes_list


class AdaptiveGraphSampler(GraphSampler):
    """
    🚀 自适应图采样器 - 根据图的特性自动选择最佳采样策略
    """
    
    def __init__(self, max_nodes: int = 200, max_edges: int = 500):
        super().__init__(max_nodes, max_edges)
        
    def adaptive_sampling(self,
                         node_features: torch.Tensor,
                         edge_index: torch.Tensor,
                         edge_type: torch.Tensor,
                         query_nodes: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        🚀 自适应采样 - 根据图的特性选择最佳策略
        Args:
            node_features: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            query_nodes: 查询相关的节点（可选）
        Returns:
            采样后的子图组件
        """
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)
        
        if num_nodes <= self.max_nodes:
            return node_features, edge_index, edge_type, list(range(num_nodes))
        
        # 计算图的密度
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # 计算平均度
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        
        # 根据图特性选择采样策略
        if query_nodes and len(query_nodes) > 0:
            # 有查询节点时，使用K跳采样
            if avg_degree > 10:  # 高度连接的图
                return self.k_hop_sampling(node_features, edge_index, edge_type, query_nodes, k=1)
            else:  # 稀疏图
                return self.k_hop_sampling(node_features, edge_index, edge_type, query_nodes, k=2)
        
        elif density > 0.1:  # 密集图
            # 使用随机游走采样
            start_nodes = random.sample(range(num_nodes), min(5, num_nodes))
            return self.random_walk_sampling(node_features, edge_index, edge_type, start_nodes, walk_length=3, num_walks=5)
        
        else:  # 稀疏图
            # 使用更长的随机游走
            start_nodes = random.sample(range(num_nodes), min(3, num_nodes))
            return self.random_walk_sampling(node_features, edge_index, edge_type, start_nodes, walk_length=5, num_walks=8)
