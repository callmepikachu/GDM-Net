"""
持久化图记忆系统 - 管理跨输入样本的全局图状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Any
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .relation_type_manager import get_global_relation_manager

# 🚀 尝试导入Faiss进行高效相似度搜索
try:
    import faiss
    FAISS_AVAILABLE = True
    print("✅ Faiss available for accelerated similarity search")
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ Faiss not available, using sklearn for similarity search")


class PersistentGraphMemory:
    """
    管理跨输入样本的持久化全局图记忆。
    使用内存存储节点和边，并提供更新和查询接口。
    """
    def __init__(self, node_dim: int = 768, device: torch.device = torch.device('cpu')):
        self.node_dim = node_dim
        self.device = device

        # 存储全局图状态
        # 使用字典存储节点，键为唯一ID，值为节点信息
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # 存储边，(src_id, rel_type, dst_id) 作为键，值为边信息（如权重、时间戳）
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        # 实体类型到ID的映射，用于快速查找
        self.entity_type_index: Dict[str, List[str]] = {}

        # 用于实体对齐的缓存（可选，存储最近访问的节点嵌入以加速）
        self._embedding_cache: Optional[np.ndarray] = None
        self._cache_node_ids: List[str] = []

        # 🚀 Faiss索引用于高效相似度搜索
        self.faiss_index: Optional[Any] = None
        self.faiss_node_ids: List[str] = []
        self.faiss_needs_update: bool = False

        # 🚀 关系类型管理器
        self.relation_manager = get_global_relation_manager()

    def add_or_update_nodes(self, local_entities: List[Dict], local_node_features: torch.Tensor, aligner: 'EntityAligner') -> Dict[int, str]:
        """
        将局部图中的节点添加到全局图或更新现有节点。
        Args:
            local_entities: 从当前块提取的实体列表。
            local_node_features: 对应的局部节点特征 [num_local_entities, node_dim]。
            aligner: 实体对齐器实例。
        Returns:
            local_to_global_map: 一个字典，将局部节点索引映射到全局节点ID。
        """
        local_to_global_map = {}
        updated_embeddings = []
        updated_node_ids = []

        for i, entity in enumerate(local_entities):
            local_feat = local_node_features[i].cpu().detach().numpy() # 转为numpy进行计算

            # 1. 实体对齐：查找全局图中是否已存在该实体
            global_node_id = aligner.align(entity, local_feat, self)

            if global_node_id is None:
                # 2a. 未找到匹配实体，创建新节点
                global_node_id = str(uuid.uuid4()) # 生成唯一ID
                # 可以考虑加入更多实体信息，如文本、类型等
                self.nodes[global_node_id] = {
                    'embedding': local_feat,
                    'type': entity.get('type', 'UNKNOWN'),
                    'text': entity.get('text', ''), # 需要从原始数据中获取或存储
                    'count': 1 # 记录出现次数
                }
                # 更新类型索引
                ent_type = entity.get('type', 'UNKNOWN')
                if ent_type not in self.entity_type_index:
                    self.entity_type_index[ent_type] = []
                self.entity_type_index[ent_type].append(global_node_id)
            else:
                # 2b. 找到匹配实体，更新其嵌入（例如，使用移动平均）
                existing_node = self.nodes[global_node_id]
                alpha = 0.1 # 更新率
                updated_embedding = (1 - alpha) * existing_node['embedding'] + alpha * local_feat
                self.nodes[global_node_id]['embedding'] = updated_embedding
                self.nodes[global_node_id]['count'] += 1

            local_to_global_map[i] = global_node_id
            updated_embeddings.append(self.nodes[global_node_id]['embedding'])
            updated_node_ids.append(global_node_id)

        # 更新缓存
        if updated_embeddings:
            self._embedding_cache = np.array(updated_embeddings)
            self._cache_node_ids = updated_node_ids
            # 标记Faiss索引需要更新
            self.faiss_needs_update = True

        return local_to_global_map

    def add_or_update_edges(self, local_relations: List[Dict], local_to_global_map: Dict[int, str]):
        """
        将局部图中的边添加到全局图或更新现有边。
        Args:
            local_relations: 从当前块提取的关系列表。
            local_to_global_map: 局部节点索引到全局节点ID的映射。
        """
        for relation in local_relations:
            local_head_idx = relation['head']
            local_tail_idx = relation['tail']

            # 检查头尾实体是否都在全局图中（理论上应该都在）
            if local_head_idx in local_to_global_map and local_tail_idx in local_to_global_map:
                global_head_id = local_to_global_map[local_head_idx]
                global_tail_id = local_to_global_map[local_tail_idx]

                # 🚀 使用关系类型管理器处理关系类型
                rel_type_str = str(relation['type'])
                rel_type_id = self.relation_manager.get_id(rel_type_str)

                edge_key = (global_head_id, rel_type_id, global_tail_id)

                if edge_key in self.edges:
                    # 更新现有边（例如，增加计数）
                    self.edges[edge_key]['count'] += 1
                else:
                    # 添加新边
                    self.edges[edge_key] = {
                        'count': 1,
                        # 可以添加其他边属性，如置信度、时间戳等
                    }

    def _update_faiss_index(self):
        """🚀 更新或构建Faiss索引以加速相似度搜索"""
        if not FAISS_AVAILABLE or not self.nodes:
            return

        try:
            # 收集所有节点嵌入
            embeddings = np.array([node['embedding'] for node in self.nodes.values()]).astype('float32')
            self.faiss_node_ids = list(self.nodes.keys())

            # 构建Faiss索引
            dimension = embeddings.shape[1]

            # 选择索引类型：对于中小规模使用FlatIP，大规模使用IVF
            if len(self.nodes) < 10000:
                self.faiss_index = faiss.IndexFlatIP(dimension)  # 精确搜索
            else:
                # 大规模数据使用近似搜索
                nlist = min(100, len(self.nodes) // 100)  # 聚类数量
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                # 训练索引（IVF需要训练）
                self.faiss_index.train(embeddings)

            # L2归一化以支持余弦相似度
            faiss.normalize_L2(embeddings)

            # 添加向量到索引
            self.faiss_index.add(embeddings)

            self.faiss_needs_update = False
            print(f"✅ Updated Faiss index with {len(self.nodes)} nodes")

        except Exception as e:
            print(f"⚠️ Failed to update Faiss index: {e}")
            self.faiss_index = None

    def get_subgraph_for_query(self, query_embedding: np.ndarray, top_k: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        🚀 根据查询嵌入检索最相关的节点子集及其连接的边，构建一个子图用于推理。
        使用Faiss加速相似度搜索。
        Args:
            query_embedding: 查询的嵌入向量 [node_dim]。
            top_k: 返回最相关的 top_k 个节点。
        Returns:
            node_features: 子图节点特征 [num_nodes, node_dim] (在device上)。
            edge_index: 子图边索引 [2, num_edges] (在device上)。
            edge_type: 子图边类型 [num_edges] (在device上)。
            node_ids: 子图中节点的全局ID列表。
        """
        if not self.nodes:
            # 如果全局图为空，返回空图
            return (torch.empty(0, self.node_dim, device=self.device),
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device),
                    [])

        # 🚀 使用Faiss加速搜索（如果可用）
        if FAISS_AVAILABLE and self.faiss_index is not None:
            if self.faiss_needs_update:
                self._update_faiss_index()

            try:
                # 准备查询向量
                query_vec = query_embedding.astype('float32').reshape(1, -1)
                faiss.normalize_L2(query_vec)  # L2归一化以支持余弦相似度

                # Faiss搜索
                k = min(top_k, len(self.faiss_node_ids))
                similarities, indices = self.faiss_index.search(query_vec, k)

                # 获取结果
                top_k_indices = indices[0]
                subgraph_node_ids = [self.faiss_node_ids[i] for i in top_k_indices]
                subgraph_node_features = torch.tensor(
                    np.array([self.nodes[node_id]['embedding'] for node_id in subgraph_node_ids]),
                    dtype=torch.float, device=self.device
                )

            except Exception as e:
                print(f"⚠️ Faiss search failed, falling back to sklearn: {e}")
                # 回退到sklearn方法
                return self._fallback_similarity_search(query_embedding, top_k)
        else:
            # 回退到sklearn方法
            return self._fallback_similarity_search(query_embedding, top_k)

        # 构建子图边（只包含子图节点之间的边）
        subgraph_edges = []
        subgraph_edge_types = []
        node_id_to_index = {nid: idx for idx, nid in enumerate(subgraph_node_ids)}

        for (src, rel_id, dst), edge_data in self.edges.items():
            if src in node_id_to_index and dst in node_id_to_index:
                subgraph_edges.append([node_id_to_index[src], node_id_to_index[dst]])
                # 🚀 直接使用关系类型ID（已经是整数）
                subgraph_edge_types.append(rel_id)

        if subgraph_edges:
            subgraph_edge_index = torch.tensor(subgraph_edges, dtype=torch.long, device=self.device).t().contiguous()
            subgraph_edge_type = torch.tensor(subgraph_edge_types, dtype=torch.long, device=self.device)
        else:
            subgraph_edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
            subgraph_edge_type = torch.empty(0, dtype=torch.long, device=self.device)

        return subgraph_node_features, subgraph_edge_index, subgraph_edge_type, subgraph_node_ids

    def _fallback_similarity_search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """回退到sklearn的相似度搜索方法"""
        all_node_embeddings = np.array([node['embedding'] for node in self.nodes.values()])
        all_node_ids = list(self.nodes.keys())

        # 计算查询与所有节点的相似度
        similarities = cosine_similarity(query_embedding.reshape(1, -1), all_node_embeddings).flatten()
        top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]  # 按相似度排序

        # 构建子图节点
        subgraph_node_ids = [all_node_ids[i] for i in top_k_indices]
        subgraph_node_features = torch.tensor(all_node_embeddings[top_k_indices], dtype=torch.float, device=self.device)

        # 构建子图边（只包含子图节点之间的边）
        subgraph_edges = []
        subgraph_edge_types = []
        node_id_to_index = {nid: idx for idx, nid in enumerate(subgraph_node_ids)}

        for (src, rel, dst), edge_data in self.edges.items():
            if src in node_id_to_index and dst in node_id_to_index:
                subgraph_edges.append([node_id_to_index[src], node_id_to_index[dst]])
                # 这里需要一个从字符串rel_type到整数ID的映射，假设在模型中已定义
                # 为简化，这里直接使用hash或预定义映射
                rel_id = hash(rel) % 1000000 # 示例，实际应使用预定义映射
                subgraph_edge_types.append(rel_id)

        if subgraph_edges:
            subgraph_edge_index = torch.tensor(subgraph_edges, dtype=torch.long, device=self.device).t().contiguous()
            subgraph_edge_type = torch.tensor(subgraph_edge_types, dtype=torch.long, device=self.device)
        else:
            subgraph_edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
            subgraph_edge_type = torch.empty(0, dtype=torch.long, device=self.device)

        return subgraph_node_features, subgraph_edge_index, subgraph_edge_type, subgraph_node_ids

    def get_all_nodes(self) -> Tuple[torch.Tensor, List[str]]:
        """获取所有节点特征和ID，用于调试或完整图分析。"""
        if not self.nodes:
             return (torch.empty(0, self.node_dim, device=self.device), [])
        all_node_embeddings = torch.tensor(np.array([node['embedding'] for node in self.nodes.values()]), dtype=torch.float, device=self.device)
        all_node_ids = list(self.nodes.keys())
        return all_node_embeddings, all_node_ids

    def save_to_disk(self, filepath: str):
        """将图状态保存到磁盘 (示例，实际可能需要更复杂的序列化)"""
        import pickle
        state = {
            'nodes': self.nodes,
            'edges': self.edges,
            'entity_type_index': self.entity_type_index
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_from_disk(self, filepath: str):
        """从磁盘加载图状态 (示例)"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.nodes = state['nodes']
        self.edges = state['edges']
        self.entity_type_index = state['entity_type_index']
        # 清除缓存
        self._embedding_cache = None
        self._cache_node_ids = []
