"""
分布式图存储接口 - 支持大规模图的分布式存储和查询
"""

import os
import pickle
import hashlib
import json
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import numpy as np


class GraphStorageInterface(ABC):
    """图存储接口基类"""
    
    @abstractmethod
    def store_node(self, node_id: str, node_data: Dict[str, Any]) -> bool:
        """存储节点"""
        pass
    
    @abstractmethod
    def store_edge(self, src_id: str, dst_id: str, rel_type: int, edge_data: Dict[str, Any]) -> bool:
        """存储边"""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点"""
        pass
    
    @abstractmethod
    def get_edges(self, node_id: str) -> List[Tuple[str, int, str, Dict[str, Any]]]:
        """获取节点的所有边"""
        pass
    
    @abstractmethod
    def query_nodes_by_type(self, node_type: str, limit: int = 100) -> List[Tuple[str, Dict[str, Any]]]:
        """按类型查询节点"""
        pass
    
    @abstractmethod
    def query_similar_nodes(self, embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, Dict[str, Any], float]]:
        """查询相似节点"""
        pass


class ShardedFileStorage(GraphStorageInterface):
    """
    🚀 基于文件的分片存储 - 适合中等规模的图
    """
    
    def __init__(self, storage_dir: str, num_shards: int = 16):
        self.storage_dir = storage_dir
        self.num_shards = num_shards
        
        # 创建存储目录
        os.makedirs(storage_dir, exist_ok=True)
        for i in range(num_shards):
            os.makedirs(os.path.join(storage_dir, f"shard_{i}"), exist_ok=True)
        
        # 内存缓存
        self.node_cache = {}
        self.edge_cache = {}
        self.cache_size_limit = 10000
    
    def _get_shard_id(self, node_id: str) -> int:
        """根据节点ID计算分片ID"""
        return int(hashlib.md5(node_id.encode()).hexdigest(), 16) % self.num_shards
    
    def _get_node_file_path(self, node_id: str) -> str:
        """获取节点文件路径"""
        shard_id = self._get_shard_id(node_id)
        return os.path.join(self.storage_dir, f"shard_{shard_id}", "nodes.pkl")
    
    def _get_edge_file_path(self, node_id: str) -> str:
        """获取边文件路径"""
        shard_id = self._get_shard_id(node_id)
        return os.path.join(self.storage_dir, f"shard_{shard_id}", "edges.pkl")
    
    def _load_shard_nodes(self, shard_id: int) -> Dict[str, Dict[str, Any]]:
        """加载分片的节点数据"""
        file_path = os.path.join(self.storage_dir, f"shard_{shard_id}", "nodes.pkl")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_shard_nodes(self, shard_id: int, nodes: Dict[str, Dict[str, Any]]):
        """保存分片的节点数据"""
        file_path = os.path.join(self.storage_dir, f"shard_{shard_id}", "nodes.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(nodes, f)
    
    def _load_shard_edges(self, shard_id: int) -> Dict[str, List[Tuple[str, int, str, Dict[str, Any]]]]:
        """加载分片的边数据"""
        file_path = os.path.join(self.storage_dir, f"shard_{shard_id}", "edges.pkl")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_shard_edges(self, shard_id: int, edges: Dict[str, List[Tuple[str, int, str, Dict[str, Any]]]]):
        """保存分片的边数据"""
        file_path = os.path.join(self.storage_dir, f"shard_{shard_id}", "edges.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(edges, f)
    
    def store_node(self, node_id: str, node_data: Dict[str, Any]) -> bool:
        """存储节点"""
        try:
            shard_id = self._get_shard_id(node_id)
            
            # 加载分片数据
            shard_nodes = self._load_shard_nodes(shard_id)
            shard_nodes[node_id] = node_data
            
            # 保存分片数据
            self._save_shard_nodes(shard_id, shard_nodes)
            
            # 更新缓存
            self.node_cache[node_id] = node_data
            self._manage_cache_size()
            
            return True
        except Exception as e:
            print(f"❌ Failed to store node {node_id}: {e}")
            return False
    
    def store_edge(self, src_id: str, dst_id: str, rel_type: int, edge_data: Dict[str, Any]) -> bool:
        """存储边"""
        try:
            # 边存储在源节点的分片中
            shard_id = self._get_shard_id(src_id)
            
            # 加载分片边数据
            shard_edges = self._load_shard_edges(shard_id)
            
            if src_id not in shard_edges:
                shard_edges[src_id] = []
            
            # 检查是否已存在相同的边
            edge_tuple = (src_id, rel_type, dst_id, edge_data)
            existing_edges = shard_edges[src_id]
            
            # 更新或添加边
            updated = False
            for i, (s, r, d, data) in enumerate(existing_edges):
                if s == src_id and r == rel_type and d == dst_id:
                    existing_edges[i] = edge_tuple
                    updated = True
                    break
            
            if not updated:
                existing_edges.append(edge_tuple)
            
            # 保存分片数据
            self._save_shard_edges(shard_id, shard_edges)
            
            return True
        except Exception as e:
            print(f"❌ Failed to store edge {src_id}->{dst_id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点"""
        # 先检查缓存
        if node_id in self.node_cache:
            return self.node_cache[node_id]
        
        try:
            shard_id = self._get_shard_id(node_id)
            shard_nodes = self._load_shard_nodes(shard_id)
            
            node_data = shard_nodes.get(node_id)
            if node_data:
                # 更新缓存
                self.node_cache[node_id] = node_data
                self._manage_cache_size()
            
            return node_data
        except Exception as e:
            print(f"❌ Failed to get node {node_id}: {e}")
            return None
    
    def get_edges(self, node_id: str) -> List[Tuple[str, int, str, Dict[str, Any]]]:
        """获取节点的所有边"""
        try:
            shard_id = self._get_shard_id(node_id)
            shard_edges = self._load_shard_edges(shard_id)
            
            return shard_edges.get(node_id, [])
        except Exception as e:
            print(f"❌ Failed to get edges for {node_id}: {e}")
            return []
    
    def query_nodes_by_type(self, node_type: str, limit: int = 100) -> List[Tuple[str, Dict[str, Any]]]:
        """按类型查询节点"""
        results = []
        
        try:
            for shard_id in range(self.num_shards):
                if len(results) >= limit:
                    break
                
                shard_nodes = self._load_shard_nodes(shard_id)
                
                for node_id, node_data in shard_nodes.items():
                    if len(results) >= limit:
                        break
                    
                    if node_data.get('type') == node_type:
                        results.append((node_id, node_data))
            
            return results[:limit]
        except Exception as e:
            print(f"❌ Failed to query nodes by type {node_type}: {e}")
            return []
    
    def query_similar_nodes(self, embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, Dict[str, Any], float]]:
        """查询相似节点"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        candidates = []
        
        try:
            for shard_id in range(self.num_shards):
                shard_nodes = self._load_shard_nodes(shard_id)
                
                for node_id, node_data in shard_nodes.items():
                    if 'embedding' in node_data:
                        node_embedding = np.array(node_data['embedding'])
                        similarity = cosine_similarity(
                            embedding.reshape(1, -1), 
                            node_embedding.reshape(1, -1)
                        )[0, 0]
                        candidates.append((node_id, node_data, similarity))
            
            # 按相似度排序并返回top_k
            candidates.sort(key=lambda x: x[2], reverse=True)
            return candidates[:top_k]
            
        except Exception as e:
            print(f"❌ Failed to query similar nodes: {e}")
            return []
    
    def _manage_cache_size(self):
        """管理缓存大小"""
        if len(self.node_cache) > self.cache_size_limit:
            # 简单的LRU：删除一半缓存
            items = list(self.node_cache.items())
            self.node_cache = dict(items[len(items)//2:])
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        total_nodes = 0
        total_edges = 0
        shard_sizes = []
        
        for shard_id in range(self.num_shards):
            try:
                shard_nodes = self._load_shard_nodes(shard_id)
                shard_edges = self._load_shard_edges(shard_id)
                
                shard_node_count = len(shard_nodes)
                shard_edge_count = sum(len(edges) for edges in shard_edges.values())
                
                total_nodes += shard_node_count
                total_edges += shard_edge_count
                shard_sizes.append({
                    'shard_id': shard_id,
                    'nodes': shard_node_count,
                    'edges': shard_edge_count
                })
            except:
                shard_sizes.append({
                    'shard_id': shard_id,
                    'nodes': 0,
                    'edges': 0
                })
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'num_shards': self.num_shards,
            'cache_size': len(self.node_cache),
            'shard_sizes': shard_sizes
        }


class DistributedGraphStorage:
    """
    🚀 分布式图存储管理器 - 统一管理不同的存储后端
    """
    
    def __init__(self, storage_type: str = "sharded_file", **kwargs):
        self.storage_type = storage_type
        
        if storage_type == "sharded_file":
            self.storage = ShardedFileStorage(**kwargs)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def migrate_from_memory(self, memory_data: Dict[str, Any]) -> bool:
        """
        从内存数据迁移到分布式存储
        Args:
            memory_data: 内存中的图数据
        Returns:
            bool: 迁移是否成功
        """
        try:
            nodes = memory_data.get('nodes', {})
            edges = memory_data.get('edges', {})
            
            print(f"🔄 Migrating {len(nodes)} nodes and {len(edges)} edges...")
            
            # 迁移节点
            for node_id, node_data in nodes.items():
                self.storage.store_node(node_id, node_data)
            
            # 迁移边
            for (src, rel_type, dst), edge_data in edges.items():
                self.storage.store_edge(src, dst, rel_type, edge_data)
            
            print("✅ Migration completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    def get_storage_interface(self) -> GraphStorageInterface:
        """获取存储接口"""
        return self.storage
