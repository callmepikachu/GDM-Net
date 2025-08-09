"""
图记忆管理工具 - 提供图记忆的分析、清理和管理功能
"""

import os
import pickle
import argparse
from typing import Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GraphMemoryManager:
    """图记忆管理器"""
    
    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.memory_data = None
        self.load_memory()
    
    def load_memory(self):
        """加载图记忆数据"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'rb') as f:
                self.memory_data = pickle.load(f)
            print(f"✅ Loaded graph memory from {self.memory_path}")
        else:
            print(f"❌ Memory file not found: {self.memory_path}")
            self.memory_data = {'nodes': {}, 'edges': {}, 'entity_type_index': {}}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图记忆统计信息"""
        if not self.memory_data:
            return {}
        
        nodes = self.memory_data.get('nodes', {})
        edges = self.memory_data.get('edges', {})
        entity_types = self.memory_data.get('entity_type_index', {})
        
        # 计算节点统计
        node_counts_by_type = {}
        for node_id, node_data in nodes.items():
            node_type = node_data.get('type', 'UNKNOWN')
            node_counts_by_type[node_type] = node_counts_by_type.get(node_type, 0) + 1
        
        # 计算边统计
        edge_counts_by_type = {}
        for (src, rel_type, dst), edge_data in edges.items():
            edge_counts_by_type[rel_type] = edge_counts_by_type.get(rel_type, 0) + 1
        
        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'entity_types': len(entity_types),
            'node_counts_by_type': node_counts_by_type,
            'edge_counts_by_type': edge_counts_by_type,
            'avg_node_degree': len(edges) * 2 / len(nodes) if len(nodes) > 0 else 0
        }
    
    def find_similar_entities(self, query_text: str, top_k: int = 10):
        """查找与查询文本相似的实体"""
        if not self.memory_data or not self.memory_data.get('nodes'):
            print("No nodes in memory")
            return []
        
        nodes = self.memory_data['nodes']
        similar_entities = []
        
        for node_id, node_data in nodes.items():
            entity_text = node_data.get('text', '')
            if query_text.lower() in entity_text.lower():
                similar_entities.append({
                    'id': node_id,
                    'text': entity_text,
                    'type': node_data.get('type', 'UNKNOWN'),
                    'count': node_data.get('count', 1)
                })
        
        # 按出现次数排序
        similar_entities.sort(key=lambda x: x['count'], reverse=True)
        return similar_entities[:top_k]
    
    def clean_low_frequency_nodes(self, min_count: int = 2):
        """清理低频节点"""
        if not self.memory_data:
            return
        
        nodes = self.memory_data['nodes']
        edges = self.memory_data['edges']
        
        # 找到低频节点
        low_freq_nodes = set()
        for node_id, node_data in nodes.items():
            if node_data.get('count', 1) < min_count:
                low_freq_nodes.add(node_id)
        
        # 删除低频节点
        for node_id in low_freq_nodes:
            del nodes[node_id]
        
        # 删除涉及低频节点的边
        edges_to_remove = []
        for (src, rel_type, dst) in edges.keys():
            if src in low_freq_nodes or dst in low_freq_nodes:
                edges_to_remove.append((src, rel_type, dst))
        
        for edge_key in edges_to_remove:
            del edges[edge_key]
        
        print(f"🗑️ Removed {len(low_freq_nodes)} low-frequency nodes and {len(edges_to_remove)} edges")
    
    def save_memory(self, output_path: str = None):
        """保存清理后的图记忆"""
        save_path = output_path or self.memory_path
        with open(save_path, 'wb') as f:
            pickle.dump(self.memory_data, f)
        print(f"💾 Saved cleaned memory to {save_path}")
    
    def export_to_text(self, output_path: str):
        """导出图记忆为可读文本格式"""
        if not self.memory_data:
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== Graph Memory Export ===\n\n")
            
            # 写入统计信息
            stats = self.get_statistics()
            f.write("Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 写入节点信息
            f.write("Nodes:\n")
            nodes = self.memory_data.get('nodes', {})
            for node_id, node_data in list(nodes.items())[:100]:  # 只显示前100个
                f.write(f"  {node_id}: {node_data.get('text', 'N/A')} ({node_data.get('type', 'UNKNOWN')})\n")
            
            if len(nodes) > 100:
                f.write(f"  ... and {len(nodes) - 100} more nodes\n")
            
            f.write("\n")
            
            # 写入边信息
            f.write("Edges (sample):\n")
            edges = self.memory_data.get('edges', {})
            for i, ((src, rel_type, dst), edge_data) in enumerate(list(edges.items())[:50]):
                src_text = nodes.get(src, {}).get('text', src[:8])
                dst_text = nodes.get(dst, {}).get('text', dst[:8])
                f.write(f"  {src_text} --[{rel_type}]--> {dst_text} (count: {edge_data.get('count', 1)})\n")
                if i >= 49:
                    break
            
            if len(edges) > 50:
                f.write(f"  ... and {len(edges) - 50} more edges\n")
        
        print(f"📄 Exported memory to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Graph Memory Manager")
    parser.add_argument("--memory_path", required=True, help="Path to graph memory file")
    parser.add_argument("--action", choices=['stats', 'clean', 'export', 'search'], 
                       default='stats', help="Action to perform")
    parser.add_argument("--min_count", type=int, default=2, 
                       help="Minimum count for cleaning (default: 2)")
    parser.add_argument("--output", help="Output path for export/save")
    parser.add_argument("--query", help="Query text for search")
    parser.add_argument("--top_k", type=int, default=10, help="Top K results for search")
    
    args = parser.parse_args()
    
    manager = GraphMemoryManager(args.memory_path)
    
    if args.action == 'stats':
        stats = manager.get_statistics()
        print("\n📊 Graph Memory Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.action == 'clean':
        manager.clean_low_frequency_nodes(args.min_count)
        if args.output:
            manager.save_memory(args.output)
        else:
            manager.save_memory()
    
    elif args.action == 'export':
        output_path = args.output or args.memory_path.replace('.pkl', '.txt')
        manager.export_to_text(output_path)
    
    elif args.action == 'search':
        if not args.query:
            print("❌ Please provide --query for search")
            return
        
        results = manager.find_similar_entities(args.query, args.top_k)
        print(f"\n🔍 Search results for '{args.query}':")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text']} ({result['type']}) - count: {result['count']}")


if __name__ == "__main__":
    main()
