"""
批处理图记忆更新器 - 优化批量更新图记忆的性能
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from .persistent_graph_memory import PersistentGraphMemory
from .entity_aligner import EntityAligner


class BatchGraphUpdater:
    """
    批处理图记忆更新器，优化多样本同时更新图记忆的性能
    """
    
    def __init__(self, graph_memory: PersistentGraphMemory, entity_aligner: EntityAligner):
        self.graph_memory = graph_memory
        self.entity_aligner = entity_aligner
        
        # 批处理缓存
        self.batch_entities = []
        self.batch_relations = []
        self.batch_node_features = []
        
    def add_batch_sample(self, entities: List[Dict], relations: List[Dict], node_features: torch.Tensor):
        """
        添加一个样本到批处理缓存
        Args:
            entities: 实体列表
            relations: 关系列表  
            node_features: 节点特征 [num_entities, hidden_size]
        """
        self.batch_entities.append(entities)
        self.batch_relations.append(relations)
        self.batch_node_features.append(node_features.cpu().detach().numpy())
    
    def flush_batch(self) -> Dict[str, int]:
        """
        批量处理所有缓存的样本，更新图记忆
        Returns:
            Dict: 更新统计信息
        """
        if not self.batch_entities:
            return {'nodes_added': 0, 'nodes_updated': 0, 'edges_added': 0, 'edges_updated': 0}
        
        stats = {'nodes_added': 0, 'nodes_updated': 0, 'edges_added': 0, 'edges_updated': 0}
        
        # 🚀 批量处理实体对齐和更新
        all_entities = []
        all_node_features = []
        sample_entity_counts = []
        
        # 收集所有实体和特征
        for entities, node_features in zip(self.batch_entities, self.batch_node_features):
            all_entities.extend(entities)
            all_node_features.extend(node_features)
            sample_entity_counts.append(len(entities))
        
        if all_entities:
            # 🔧 确保实体和特征数量匹配
            if len(all_entities) != len(all_node_features):
                print(f"⚠️ Entity-feature mismatch: {len(all_entities)} entities vs {len(all_node_features)} features")
                # 取较小的数量以避免索引错误
                min_count = min(len(all_entities), len(all_node_features))
                all_entities = all_entities[:min_count]
                all_node_features = all_node_features[:min_count]

            # 批量实体对齐
            all_node_features_np = np.array(all_node_features)
            global_node_ids = self._batch_entity_alignment(all_entities, all_node_features_np, stats)
            
            # 重新分组为每个样本的映射
            sample_mappings = []
            start_idx = 0
            for count in sample_entity_counts:
                end_idx = start_idx + count
                # 🔧 修复索引越界：确保不超出global_node_ids的范围
                if end_idx <= len(global_node_ids):
                    sample_mapping = {i: global_node_ids[start_idx + i] for i in range(count)}
                else:
                    # 如果索引超出范围，只映射可用的部分
                    available_count = len(global_node_ids) - start_idx
                    sample_mapping = {i: global_node_ids[start_idx + i] for i in range(min(count, available_count))}
                    print(f"⚠️ Index mismatch: expected {count} entities, got {available_count}")

                sample_mappings.append(sample_mapping)
                start_idx = end_idx
            
            # 🚀 批量处理关系
            for relations, mapping in zip(self.batch_relations, sample_mappings):
                self._batch_relation_update(relations, mapping, stats)
        
        # 清空缓存
        self.clear_batch()
        
        # 标记Faiss索引需要更新
        self.graph_memory.faiss_needs_update = True
        
        return stats
    
    def _batch_entity_alignment(self, entities: List[Dict], node_features: np.ndarray, stats: Dict) -> List[str]:
        """
        批量实体对齐
        Args:
            entities: 所有实体列表
            node_features: 所有节点特征 [total_entities, hidden_size]
            stats: 统计信息字典
        Returns:
            List[str]: 对应的全局节点ID列表
        """
        global_node_ids = []
        
        # 🚀 如果图记忆为空，直接批量添加
        if not self.graph_memory.nodes:
            for i, (entity, features) in enumerate(zip(entities, node_features)):
                global_node_id = self._create_new_node(entity, features)
                global_node_ids.append(global_node_id)
                stats['nodes_added'] += 1
            return global_node_ids
        
        # 🚀 批量相似度计算
        existing_embeddings = np.array([node['embedding'] for node in self.graph_memory.nodes.values()])
        existing_node_ids = list(self.graph_memory.nodes.keys())
        
        if len(existing_embeddings) > 0:
            # 计算所有新实体与所有现有实体的相似度矩阵
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(node_features, existing_embeddings)
            
            # 对每个新实体找最相似的现有实体
            for i, (entity, features) in enumerate(zip(entities, node_features)):
                similarities = similarity_matrix[i]
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]
                
                if max_sim > self.entity_aligner.similarity_threshold:
                    # 找到匹配实体，更新
                    global_node_id = existing_node_ids[max_sim_idx]
                    self._update_existing_node(global_node_id, features)
                    global_node_ids.append(global_node_id)
                    stats['nodes_updated'] += 1
                else:
                    # 创建新实体
                    global_node_id = self._create_new_node(entity, features)
                    global_node_ids.append(global_node_id)
                    stats['nodes_added'] += 1
        else:
            # 没有现有实体，全部创建新的
            for entity, features in zip(entities, node_features):
                global_node_id = self._create_new_node(entity, features)
                global_node_ids.append(global_node_id)
                stats['nodes_added'] += 1
        
        return global_node_ids
    
    def _create_new_node(self, entity: Dict, features: np.ndarray) -> str:
        """创建新节点"""
        import uuid
        global_node_id = str(uuid.uuid4())
        
        self.graph_memory.nodes[global_node_id] = {
            'embedding': features,
            'type': entity.get('type', 'UNKNOWN'),
            'text': entity.get('text', ''),
            'count': 1
        }
        
        # 更新类型索引
        ent_type = entity.get('type', 'UNKNOWN')
        if ent_type not in self.graph_memory.entity_type_index:
            self.graph_memory.entity_type_index[ent_type] = []
        self.graph_memory.entity_type_index[ent_type].append(global_node_id)
        
        return global_node_id
    
    def _update_existing_node(self, global_node_id: str, features: np.ndarray):
        """更新现有节点"""
        existing_node = self.graph_memory.nodes[global_node_id]
        alpha = 0.1  # 更新率
        updated_embedding = (1 - alpha) * existing_node['embedding'] + alpha * features
        self.graph_memory.nodes[global_node_id]['embedding'] = updated_embedding
        self.graph_memory.nodes[global_node_id]['count'] += 1
    
    def _batch_relation_update(self, relations: List[Dict], local_to_global_map: Dict[int, str], stats: Dict):
        """批量更新关系"""
        for relation in relations:
            local_head_idx = relation['head']
            local_tail_idx = relation['tail']
            
            if local_head_idx in local_to_global_map and local_tail_idx in local_to_global_map:
                global_head_id = local_to_global_map[local_head_idx]
                global_tail_id = local_to_global_map[local_tail_idx]
                
                # 使用关系类型管理器
                rel_type_str = str(relation['type'])
                rel_type_id = self.graph_memory.relation_manager.get_id(rel_type_str)
                
                edge_key = (global_head_id, rel_type_id, global_tail_id)
                
                if edge_key in self.graph_memory.edges:
                    # 更新现有边
                    self.graph_memory.edges[edge_key]['count'] += 1
                    stats['edges_updated'] += 1
                else:
                    # 添加新边
                    self.graph_memory.edges[edge_key] = {'count': 1}
                    stats['edges_added'] += 1
    
    def clear_batch(self):
        """清空批处理缓存"""
        self.batch_entities.clear()
        self.batch_relations.clear()
        self.batch_node_features.clear()
    
    def get_batch_size(self) -> int:
        """获取当前批处理大小"""
        return len(self.batch_entities)
