"""
实体对齐器 - 负责将新提取的实体与全局图中的现有实体进行对齐
"""

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from .persistent_graph_memory import PersistentGraphMemory # 用于类型提示，避免循环导入


class EntityAligner:
    """
    🚀 多模态实体对齐器 - 结合嵌入、文本、类型等多种信息进行实体对齐
    """
    def __init__(self,
                 similarity_threshold: float = 0.85,
                 embedding_weight: float = 0.7,
                 text_weight: float = 0.2,
                 type_weight: float = 0.1):
        self.similarity_threshold = similarity_threshold

        # 🚀 多模态权重配置
        self.embedding_weight = embedding_weight
        self.text_weight = text_weight
        self.type_weight = type_weight

        # 确保权重和为1
        total_weight = embedding_weight + text_weight + type_weight
        self.embedding_weight /= total_weight
        self.text_weight /= total_weight
        self.type_weight /= total_weight

    def align(self, entity: Dict[str, Any], entity_embedding: np.ndarray, graph_memory: 'PersistentGraphMemory') -> Optional[str]:
        """
        尝试将一个新实体与全局图中的现有实体对齐。
        Args:
            entity: 新实体的字典信息。
            entity_embedding: 新实体的嵌入向量。
            graph_memory: 全局图记忆实例。
        Returns:
            str or None: 如果找到匹配，则返回全局节点ID；否则返回None。
        """
        # 策略1：使用缓存加速（如果可用且相关）
        if graph_memory._embedding_cache is not None and graph_memory._embedding_cache.size > 0:
            similarities = cosine_similarity(entity_embedding.reshape(1, -1), graph_memory._embedding_cache).flatten()
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]
            if max_sim > self.similarity_threshold:
                return graph_memory._cache_node_ids[max_sim_idx]

        # 策略2：🚀 多模态全局搜索
        if not graph_memory.nodes:
            return None

        # 优化：先按类型过滤候选节点
        entity_type = entity.get('type', 'UNKNOWN')
        candidate_nodes = {}

        # 首先尝试同类型匹配
        if entity_type in graph_memory.entity_type_index:
            for node_id in graph_memory.entity_type_index[entity_type]:
                if node_id in graph_memory.nodes:
                    candidate_nodes[node_id] = graph_memory.nodes[node_id]

        # 如果同类型候选太少，扩展到所有节点
        if len(candidate_nodes) < 10:
            candidate_nodes = graph_memory.nodes

        # 🚀 计算多模态相似度
        best_similarity = 0.0
        best_node_id = None

        for node_id, node_data in candidate_nodes.items():
            # 构造候选实体字典
            candidate_entity = {
                'type': node_data.get('type', 'UNKNOWN'),
                'text': node_data.get('text', ''),
            }

            # 计算多模态相似度
            similarity = self._compute_multimodal_similarity(
                entity, entity_embedding,
                candidate_entity, node_data['embedding']
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_node_id = node_id

        if best_similarity > self.similarity_threshold:
            return best_node_id

        return None # 未找到足够相似的实体

    def _compute_multimodal_similarity(self, entity1: Dict[str, Any], embedding1: np.ndarray,
                                     entity2: Dict[str, Any], embedding2: np.ndarray) -> float:
        """
        🚀 计算多模态相似度：结合嵌入、文本、类型等信息
        Args:
            entity1, entity2: 实体字典
            embedding1, embedding2: 实体嵌入向量
        Returns:
            float: 综合相似度分数
        """
        # 1. 嵌入相似度
        embedding_sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0, 0]

        # 2. 类型匹配
        type1 = entity1.get('type', 'UNKNOWN')
        type2 = entity2.get('type', 'UNKNOWN')
        type_sim = 1.0 if type1 == type2 else 0.0

        # 3. 文本相似度
        text1 = entity1.get('text', '').lower().strip()
        text2 = entity2.get('text', '').lower().strip()
        text_sim = self._compute_text_similarity(text1, text2)

        # 4. 综合相似度
        total_similarity = (self.embedding_weight * embedding_sim +
                          self.text_weight * text_sim +
                          self.type_weight * type_sim)

        return total_similarity

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        Args:
            text1, text2: 文本字符串
        Returns:
            float: 文本相似度 [0, 1]
        """
        if not text1 or not text2:
            return 0.0

        # 精确匹配
        if text1 == text2:
            return 1.0

        # 包含关系
        if text1 in text2 or text2 in text1:
            return 0.8

        # Jaccard相似度（基于词汇）
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard_sim = intersection / union if union > 0 else 0.0

        # 编辑距离相似度（对于短文本）
        if len(text1) <= 20 and len(text2) <= 20:
            edit_distance = self._levenshtein_distance(text1, text2)
            max_len = max(len(text1), len(text2))
            edit_sim = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0

            # 取较高的相似度
            return max(jaccard_sim, edit_sim)

        return jaccard_sim

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
