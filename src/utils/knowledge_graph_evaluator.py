"""
知识图谱质量评估器 - 评估持久化图记忆的质量和有效性
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict, Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KnowledgeGraphEvaluator:
    """
    知识图谱质量评估器
    """
    
    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.memory_data = None
        self.load_memory()
    
    def load_memory(self):
        """加载图记忆数据"""
        try:
            with open(self.memory_path, 'rb') as f:
                self.memory_data = pickle.load(f)
            print(f"✅ Loaded graph memory from {self.memory_path}")
        except Exception as e:
            print(f"❌ Failed to load memory: {e}")
            self.memory_data = {'nodes': {}, 'edges': {}, 'entity_type_index': {}}
    
    def evaluate_completeness(self) -> Dict[str, float]:
        """
        🚀 评估知识图谱的完整性
        Returns:
            完整性评估指标
        """
        nodes = self.memory_data.get('nodes', {})
        edges = self.memory_data.get('edges', {})
        
        if not nodes:
            return {'completeness_score': 0.0, 'coverage_ratio': 0.0, 'density': 0.0}
        
        # 1. 节点覆盖率（有文本信息的节点比例）
        nodes_with_text = sum(1 for node in nodes.values() if node.get('text', '').strip())
        coverage_ratio = nodes_with_text / len(nodes)
        
        # 2. 图密度
        num_nodes = len(nodes)
        num_edges = len(edges)
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # 3. 类型多样性
        entity_types = set(node.get('type', 'UNKNOWN') for node in nodes.values())
        type_diversity = len(entity_types) / max(len(nodes), 1)
        
        # 4. 综合完整性分数
        completeness_score = (coverage_ratio * 0.4 + density * 0.3 + type_diversity * 0.3)
        
        return {
            'completeness_score': completeness_score,
            'coverage_ratio': coverage_ratio,
            'density': density,
            'type_diversity': type_diversity,
            'num_entity_types': len(entity_types)
        }
    
    def evaluate_consistency(self) -> Dict[str, float]:
        """
        🚀 评估知识图谱的一致性
        Returns:
            一致性评估指标
        """
        nodes = self.memory_data.get('nodes', {})
        edges = self.memory_data.get('edges', {})
        
        if not nodes:
            return {'consistency_score': 0.0}
        
        # 1. 实体嵌入一致性（同类型实体的嵌入相似性）
        type_consistency_scores = []
        entity_types = self.memory_data.get('entity_type_index', {})
        
        for entity_type, node_ids in entity_types.items():
            if len(node_ids) >= 2:
                embeddings = []
                for node_id in node_ids:
                    if node_id in nodes:
                        embeddings.append(nodes[node_id]['embedding'])
                
                if len(embeddings) >= 2:
                    embeddings = np.array(embeddings)
                    # 计算类内相似度
                    similarities = cosine_similarity(embeddings)
                    # 排除对角线（自相似度）
                    mask = ~np.eye(similarities.shape[0], dtype=bool)
                    avg_similarity = similarities[mask].mean()
                    type_consistency_scores.append(avg_similarity)
        
        type_consistency = np.mean(type_consistency_scores) if type_consistency_scores else 0.0
        
        # 2. 关系一致性（相同关系类型的频率分布）
        relation_counts = Counter()
        for (src, rel_type, dst), edge_data in edges.items():
            relation_counts[rel_type] += edge_data.get('count', 1)
        
        if relation_counts:
            # 计算关系分布的熵（越低越一致）
            total_relations = sum(relation_counts.values())
            probabilities = [count / total_relations for count in relation_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(relation_counts))
            relation_consistency = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        else:
            relation_consistency = 0.0
        
        # 3. 综合一致性分数
        consistency_score = (type_consistency * 0.6 + relation_consistency * 0.4)
        
        return {
            'consistency_score': consistency_score,
            'type_consistency': type_consistency,
            'relation_consistency': relation_consistency,
            'num_relation_types': len(relation_counts)
        }
    
    def evaluate_connectivity(self) -> Dict[str, float]:
        """
        🚀 评估知识图谱的连通性
        Returns:
            连通性评估指标
        """
        nodes = self.memory_data.get('nodes', {})
        edges = self.memory_data.get('edges', {})
        
        if not nodes or not edges:
            return {'connectivity_score': 0.0, 'largest_component_ratio': 0.0}
        
        # 构建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        for node_id in nodes.keys():
            G.add_node(node_id)
        
        # 添加边
        for (src, rel_type, dst) in edges.keys():
            if src in nodes and dst in nodes:
                G.add_edge(src, dst)
        
        # 1. 连通分量分析
        connected_components = list(nx.connected_components(G))
        largest_component_size = max(len(comp) for comp in connected_components) if connected_components else 0
        largest_component_ratio = largest_component_size / len(nodes) if len(nodes) > 0 else 0
        
        # 2. 平均路径长度
        if largest_component_size > 1:
            largest_component = max(connected_components, key=len)
            subgraph = G.subgraph(largest_component)
            try:
                avg_path_length = nx.average_shortest_path_length(subgraph)
                # 归一化（较短的路径更好）
                normalized_path_length = 1.0 / (1.0 + avg_path_length)
            except:
                normalized_path_length = 0.0
        else:
            normalized_path_length = 0.0
        
        # 3. 聚类系数
        clustering_coefficient = nx.average_clustering(G) if len(G.nodes()) > 0 else 0.0
        
        # 4. 综合连通性分数
        connectivity_score = (
            largest_component_ratio * 0.4 +
            normalized_path_length * 0.3 +
            clustering_coefficient * 0.3
        )
        
        return {
            'connectivity_score': connectivity_score,
            'largest_component_ratio': largest_component_ratio,
            'avg_path_length': avg_path_length if 'avg_path_length' in locals() else float('inf'),
            'clustering_coefficient': clustering_coefficient,
            'num_components': len(connected_components)
        }
    
    def evaluate_redundancy(self) -> Dict[str, float]:
        """
        🚀 评估知识图谱的冗余性
        Returns:
            冗余性评估指标
        """
        nodes = self.memory_data.get('nodes', {})
        
        if len(nodes) < 2:
            return {'redundancy_score': 0.0, 'duplicate_ratio': 0.0}
        
        # 1. 基于嵌入的重复检测
        embeddings = []
        node_ids = []
        
        for node_id, node_data in nodes.items():
            embeddings.append(node_data['embedding'])
            node_ids.append(node_id)
        
        embeddings = np.array(embeddings)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 找到高相似度对（排除自相似）
        high_similarity_threshold = 0.95
        duplicate_pairs = 0
        total_pairs = 0
        
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                total_pairs += 1
                if similarity_matrix[i, j] > high_similarity_threshold:
                    duplicate_pairs += 1
        
        duplicate_ratio = duplicate_pairs / total_pairs if total_pairs > 0 else 0.0
        
        # 2. 基于文本的重复检测
        text_duplicates = 0
        text_pairs = 0
        
        texts = [node.get('text', '').lower().strip() for node in nodes.values()]
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if texts[i] and texts[j]:
                    text_pairs += 1
                    if texts[i] == texts[j]:
                        text_duplicates += 1
        
        text_duplicate_ratio = text_duplicates / text_pairs if text_pairs > 0 else 0.0
        
        # 3. 综合冗余性分数（越低越好，所以用1减去）
        redundancy_score = 1.0 - (duplicate_ratio * 0.6 + text_duplicate_ratio * 0.4)
        
        return {
            'redundancy_score': redundancy_score,
            'duplicate_ratio': duplicate_ratio,
            'text_duplicate_ratio': text_duplicate_ratio,
            'potential_duplicates': duplicate_pairs
        }
    
    def evaluate_overall_quality(self) -> Dict[str, Any]:
        """
        🚀 综合评估知识图谱质量
        Returns:
            综合质量评估报告
        """
        print("🔍 Evaluating knowledge graph quality...")
        
        # 各维度评估
        completeness = self.evaluate_completeness()
        consistency = self.evaluate_consistency()
        connectivity = self.evaluate_connectivity()
        redundancy = self.evaluate_redundancy()
        
        # 计算综合质量分数
        overall_score = (
            completeness['completeness_score'] * 0.25 +
            consistency['consistency_score'] * 0.25 +
            connectivity['connectivity_score'] * 0.25 +
            redundancy['redundancy_score'] * 0.25
        )
        
        # 质量等级
        if overall_score >= 0.8:
            quality_grade = "Excellent"
        elif overall_score >= 0.6:
            quality_grade = "Good"
        elif overall_score >= 0.4:
            quality_grade = "Fair"
        else:
            quality_grade = "Poor"
        
        return {
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'completeness': completeness,
            'consistency': consistency,
            'connectivity': connectivity,
            'redundancy': redundancy,
            'recommendations': self._generate_recommendations(
                completeness, consistency, connectivity, redundancy
            )
        }
    
    def _generate_recommendations(self, completeness: Dict, consistency: Dict, 
                                connectivity: Dict, redundancy: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if completeness['completeness_score'] < 0.6:
            recommendations.append("Improve data completeness by adding more entity text information")
        
        if consistency['consistency_score'] < 0.6:
            recommendations.append("Enhance entity alignment to improve type consistency")
        
        if connectivity['connectivity_score'] < 0.6:
            recommendations.append("Add more relationships to improve graph connectivity")
        
        if redundancy['redundancy_score'] < 0.7:
            recommendations.append("Remove duplicate entities to reduce redundancy")
        
        if completeness['density'] < 0.1:
            recommendations.append("Increase relationship extraction to improve graph density")
        
        return recommendations


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Graph Quality Evaluator")
    parser.add_argument("--memory_path", required=True, help="Path to graph memory file")
    parser.add_argument("--output", help="Output file for evaluation report")
    
    args = parser.parse_args()
    
    evaluator = KnowledgeGraphEvaluator(args.memory_path)
    quality_report = evaluator.evaluate_overall_quality()
    
    # 打印报告
    print("\n" + "="*50)
    print("📊 KNOWLEDGE GRAPH QUALITY REPORT")
    print("="*50)
    print(f"Overall Score: {quality_report['overall_score']:.3f}")
    print(f"Quality Grade: {quality_report['quality_grade']}")
    print()
    
    print("📈 Detailed Metrics:")
    for category, metrics in quality_report.items():
        if isinstance(metrics, dict) and category != 'recommendations':
            print(f"\n{category.title()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}")
    
    print("\n💡 Recommendations:")
    for i, rec in enumerate(quality_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # 保存报告
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        print(f"\n📄 Report saved to {args.output}")


if __name__ == "__main__":
    main()
