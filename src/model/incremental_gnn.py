"""
增量学习图神经网络 - 支持动态图更新的GNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from typing import Dict, List, Tuple, Optional
import copy


class IncrementalGCNConv(MessagePassing):
    """
    🚀 增量学习的GCN层 - 支持动态添加节点和边
    """
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 可学习参数
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 🚀 增量学习相关
        self.node_embeddings_cache = {}  # 缓存节点嵌入
        self.edge_cache = set()  # 缓存边信息
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                node_ids: Optional[List[str]] = None,
                incremental: bool = False) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            node_ids: 节点ID列表（用于增量更新）
            incremental: 是否使用增量模式
        Returns:
            更新后的节点特征 [num_nodes, out_channels]
        """
        if incremental and node_ids is not None:
            return self._incremental_forward(x, edge_index, node_ids)
        else:
            return self._full_forward(x, edge_index)
    
    def _full_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """完整前向传播"""
        # 线性变换
        x = torch.matmul(x, self.weight)
        
        # 消息传递
        out = self.propagate(edge_index, x=x)
        
        if self.bias is not None:
            out += self.bias
        
        return out
    
    def _incremental_forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                           node_ids: List[str]) -> torch.Tensor:
        """
        🚀 增量前向传播 - 只更新受影响的节点
        """
        # 识别新节点和受影响的节点
        new_nodes = set()
        affected_nodes = set()
        
        for i, node_id in enumerate(node_ids):
            if node_id not in self.node_embeddings_cache:
                new_nodes.add(i)
                affected_nodes.add(i)
        
        # 识别新边和受影响的节点
        current_edges = set()
        for j in range(edge_index.size(1)):
            src, dst = edge_index[0, j].item(), edge_index[1, j].item()
            edge = (src, dst)
            current_edges.add(edge)
            
            if edge not in self.edge_cache:
                # 新边，影响相关节点
                affected_nodes.add(src)
                affected_nodes.add(dst)
        
        # 更新边缓存
        self.edge_cache = current_edges
        
        if not affected_nodes:
            # 没有受影响的节点，直接返回缓存结果
            cached_embeddings = []
            for node_id in node_ids:
                if node_id in self.node_embeddings_cache:
                    cached_embeddings.append(self.node_embeddings_cache[node_id])
                else:
                    # 新节点，需要计算
                    cached_embeddings.append(torch.zeros(self.out_channels, device=x.device))
            return torch.stack(cached_embeddings)
        
        # 只对受影响的节点进行完整计算
        affected_indices = list(affected_nodes)
        if affected_indices:
            # 提取受影响节点的子图
            subgraph_x = x[affected_indices]
            
            # 构建子图边索引
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(affected_indices)}
            subgraph_edges = []
            
            for j in range(edge_index.size(1)):
                src, dst = edge_index[0, j].item(), edge_index[1, j].item()
                if src in index_mapping and dst in index_mapping:
                    subgraph_edges.append([index_mapping[src], index_mapping[dst]])
            
            if subgraph_edges:
                subgraph_edge_index = torch.tensor(subgraph_edges, dtype=torch.long, device=x.device).t()
                # 对子图进行GCN计算
                subgraph_out = self._full_forward(subgraph_x, subgraph_edge_index)
                
                # 更新缓存
                for i, old_idx in enumerate(affected_indices):
                    if old_idx < len(node_ids):
                        self.node_embeddings_cache[node_ids[old_idx]] = subgraph_out[i]
        
        # 构建完整输出
        output_embeddings = []
        for i, node_id in enumerate(node_ids):
            if node_id in self.node_embeddings_cache:
                output_embeddings.append(self.node_embeddings_cache[node_id])
            else:
                # 新节点，使用线性变换
                linear_out = torch.matmul(x[i], self.weight)
                if self.bias is not None:
                    linear_out += self.bias
                output_embeddings.append(linear_out)
                self.node_embeddings_cache[node_id] = linear_out
        
        return torch.stack(output_embeddings)
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """消息函数"""
        return x_j
    
    def clear_cache(self):
        """清空缓存"""
        self.node_embeddings_cache.clear()
        self.edge_cache.clear()


class IncrementalGNN(nn.Module):
    """
    🚀 增量学习图神经网络 - 多层增量GCN
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 构建增量GCN层
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(IncrementalGCNConv(input_dim, hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(IncrementalGCNConv(hidden_dim, hidden_dim))
        
        # 输出层
        if num_layers > 1:
            self.layers.append(IncrementalGCNConv(hidden_dim, output_dim))
        else:
            self.layers[0] = IncrementalGCNConv(input_dim, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                node_ids: Optional[List[str]] = None,
                incremental: bool = False) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            node_ids: 节点ID列表
            incremental: 是否使用增量模式
        Returns:
            节点嵌入 [num_nodes, output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, node_ids, incremental)
            
            if i < len(self.layers) - 1:  # 不在最后一层应用激活和dropout
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x
    
    def update_incrementally(self, 
                           new_x: torch.Tensor,
                           new_edge_index: torch.Tensor,
                           new_node_ids: List[str]) -> torch.Tensor:
        """
        🚀 增量更新 - 只更新新增或修改的部分
        Args:
            new_x: 新的/更新的节点特征
            new_edge_index: 新的边索引
            new_node_ids: 新的节点ID
        Returns:
            更新后的节点嵌入
        """
        return self.forward(new_x, new_edge_index, new_node_ids, incremental=True)
    
    def clear_all_caches(self):
        """清空所有层的缓存"""
        for layer in self.layers:
            if hasattr(layer, 'clear_cache'):
                layer.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        stats = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'node_embeddings_cache'):
                stats[f'layer_{i}_cached_nodes'] = len(layer.node_embeddings_cache)
                stats[f'layer_{i}_cached_edges'] = len(layer.edge_cache)
        return stats


class AdaptiveIncrementalGNN(IncrementalGNN):
    """
    🚀 自适应增量GNN - 根据图变化程度自动选择更新策略
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自适应参数
        self.change_threshold = 0.1  # 变化阈值
        self.full_update_interval = 100  # 完整更新间隔
        self.update_count = 0
    
    def adaptive_forward(self,
                        x: torch.Tensor,
                        edge_index: torch.Tensor,
                        node_ids: Optional[List[str]] = None,
                        prev_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        🚀 自适应前向传播 - 根据变化程度选择更新策略
        """
        self.update_count += 1
        
        # 强制完整更新的条件
        force_full_update = (
            self.update_count % self.full_update_interval == 0 or
            prev_x is None or
            node_ids is None
        )
        
        if force_full_update:
            # 完整更新
            self.clear_all_caches()
            return self.forward(x, edge_index, node_ids, incremental=False)
        
        # 计算变化程度
        if prev_x.shape == x.shape:
            change_ratio = torch.norm(x - prev_x) / torch.norm(prev_x)
            
            if change_ratio > self.change_threshold:
                # 变化较大，使用完整更新
                self.clear_all_caches()
                return self.forward(x, edge_index, node_ids, incremental=False)
        
        # 变化较小，使用增量更新
        return self.forward(x, edge_index, node_ids, incremental=True)
