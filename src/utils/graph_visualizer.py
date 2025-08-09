"""
图记忆可视化工具 - 可视化持久化图记忆的结构和内容
"""

import pickle
import argparse
import os
from typing import Dict, List, Tuple, Any
import numpy as np

# 尝试导入可视化库
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization libraries not available. Install: pip install networkx matplotlib seaborn scikit-learn")


class GraphMemoryVisualizer:
    """图记忆可视化器"""
    
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
            return
    
    def visualize_graph_structure(self, output_path: str = "graph_structure.png", max_nodes: int = 100):
        """
        可视化图结构
        Args:
            output_path: 输出图片路径
            max_nodes: 最大显示节点数（避免图过于复杂）
        """
        if not VISUALIZATION_AVAILABLE or not self.memory_data:
            print("❌ Cannot visualize: missing data or libraries")
            return
        
        nodes = self.memory_data.get('nodes', {})
        edges = self.memory_data.get('edges', {})
        
        if not nodes:
            print("❌ No nodes to visualize")
            return
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 选择最频繁的节点
        node_items = list(nodes.items())
        if len(node_items) > max_nodes:
            # 按出现次数排序，选择top节点
            node_items.sort(key=lambda x: x[1].get('count', 1), reverse=True)
            node_items = node_items[:max_nodes]
        
        selected_node_ids = {item[0] for item in node_items}
        
        # 添加节点
        for node_id, node_data in node_items:
            G.add_node(node_id, 
                      label=node_data.get('text', node_id[:8]),
                      type=node_data.get('type', 'UNKNOWN'),
                      count=node_data.get('count', 1))
        
        # 添加边（只包含选中节点之间的边）
        for (src, rel_id, dst), edge_data in edges.items():
            if src in selected_node_ids and dst in selected_node_ids:
                G.add_edge(src, dst, 
                          relation=str(rel_id),
                          count=edge_data.get('count', 1))
        
        # 设置图形大小
        plt.figure(figsize=(15, 10))
        
        # 计算布局
        if len(G.nodes()) > 50:
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = nx.spring_layout(G, k=2, iterations=100)
        
        # 根据节点类型设置颜色
        node_types = [G.nodes[node]['type'] for node in G.nodes()]
        unique_types = list(set(node_types))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        type_color_map = {t: colors[i] for i, t in enumerate(unique_types)}
        node_colors = [type_color_map[t] for t in node_types]
        
        # 根据出现次数设置节点大小
        node_sizes = [G.nodes[node]['count'] * 100 for node in G.nodes()]
        
        # 绘制图
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
        
        # 添加标签（只为重要节点）
        important_nodes = {node: G.nodes[node]['label'] for node in G.nodes() 
                          if G.nodes[node]['count'] > 2}
        nx.draw_networkx_labels(G, pos, important_nodes, font_size=8)
        
        plt.title(f"Graph Memory Structure ({len(G.nodes())} nodes, {len(G.edges())} edges)")
        plt.axis('off')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=type_color_map[t], markersize=10, label=t)
                          for t in unique_types[:10]]  # 最多显示10种类型
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Graph structure saved to {output_path}")
    
    def visualize_node_embeddings(self, output_path: str = "node_embeddings.png", method: str = "tsne"):
        """
        可视化节点嵌入的分布
        Args:
            output_path: 输出图片路径
            method: 降维方法 ('tsne' 或 'pca')
        """
        if not VISUALIZATION_AVAILABLE or not self.memory_data:
            print("❌ Cannot visualize: missing data or libraries")
            return
        
        nodes = self.memory_data.get('nodes', {})
        if not nodes:
            print("❌ No nodes to visualize")
            return
        
        # 提取嵌入和标签
        embeddings = []
        labels = []
        texts = []
        
        for node_id, node_data in nodes.items():
            embeddings.append(node_data['embedding'])
            labels.append(node_data.get('type', 'UNKNOWN'))
            texts.append(node_data.get('text', node_id[:8]))
        
        embeddings = np.array(embeddings)
        
        # 降维
        if method == "tsne":
            if len(embeddings) > 1000:
                # 对于大数据集，先用PCA降维再用t-SNE
                pca = PCA(n_components=50)
                embeddings_reduced = pca.fit_transform(embeddings)
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
                embeddings_2d = tsne.fit_transform(embeddings_reduced)
            else:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
                embeddings_2d = tsne.fit_transform(embeddings)
        else:  # PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        
        # 绘制散点图
        plt.figure(figsize=(12, 8))
        
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7, s=50)
        
        plt.title(f"Node Embeddings Visualization ({method.upper()})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Node embeddings saved to {output_path}")
    
    def generate_statistics_plot(self, output_path: str = "graph_statistics.png"):
        """生成图记忆统计图表"""
        if not VISUALIZATION_AVAILABLE or not self.memory_data:
            print("❌ Cannot visualize: missing data or libraries")
            return
        
        nodes = self.memory_data.get('nodes', {})
        edges = self.memory_data.get('edges', {})
        
        # 统计数据
        node_type_counts = {}
        node_frequency_dist = []
        edge_type_counts = {}
        
        for node_data in nodes.values():
            node_type = node_data.get('type', 'UNKNOWN')
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            node_frequency_dist.append(node_data.get('count', 1))
        
        for (src, rel_id, dst), edge_data in edges.items():
            edge_type_counts[str(rel_id)] = edge_type_counts.get(str(rel_id), 0) + 1
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 节点类型分布
        if node_type_counts:
            types, counts = zip(*sorted(node_type_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            ax1.bar(types, counts)
            ax1.set_title("Top 10 Node Types")
            ax1.set_xlabel("Node Type")
            ax1.set_ylabel("Count")
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. 节点频率分布
        if node_frequency_dist:
            ax2.hist(node_frequency_dist, bins=20, alpha=0.7)
            ax2.set_title("Node Frequency Distribution")
            ax2.set_xlabel("Node Frequency")
            ax2.set_ylabel("Count")
        
        # 3. 边类型分布
        if edge_type_counts:
            edge_types, edge_counts = zip(*sorted(edge_type_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            ax3.bar(edge_types, edge_counts)
            ax3.set_title("Top 10 Edge Types")
            ax3.set_xlabel("Edge Type")
            ax3.set_ylabel("Count")
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 总体统计
        stats_text = f"""
        Total Nodes: {len(nodes)}
        Total Edges: {len(edges)}
        Node Types: {len(node_type_counts)}
        Edge Types: {len(edge_type_counts)}
        Avg Node Degree: {len(edges) * 2 / len(nodes) if len(nodes) > 0 else 0:.2f}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        ax4.set_title("Overall Statistics")
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Statistics plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Graph Memory Visualizer")
    parser.add_argument("--memory_path", required=True, help="Path to graph memory file")
    parser.add_argument("--output_dir", default="visualizations", help="Output directory")
    parser.add_argument("--max_nodes", type=int, default=100, help="Maximum nodes to display")
    parser.add_argument("--embedding_method", choices=['tsne', 'pca'], default='tsne',
                       help="Dimensionality reduction method for embeddings")
    
    args = parser.parse_args()
    
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization libraries not available")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建可视化器
    visualizer = GraphMemoryVisualizer(args.memory_path)
    
    # 生成可视化
    print("🎨 Generating visualizations...")
    
    visualizer.visualize_graph_structure(
        os.path.join(args.output_dir, "graph_structure.png"),
        max_nodes=args.max_nodes
    )
    
    visualizer.visualize_node_embeddings(
        os.path.join(args.output_dir, "node_embeddings.png"),
        method=args.embedding_method
    )
    
    visualizer.generate_statistics_plot(
        os.path.join(args.output_dir, "graph_statistics.png")
    )
    
    print(f"✅ All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
