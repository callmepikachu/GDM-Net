import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple, Optional

from .bert_encoder import DocumentEncoder, StructureExtractor
from .graph_memory import GraphWriter, GraphMemory
from .dual_memory import DualMemorySystem
from .reasoning_module import ReasoningModule
from .persistent_graph_memory import PersistentGraphMemory
from .entity_aligner import EntityAligner
from .batch_graph_updater import BatchGraphUpdater


class GDMNet(nn.Module):
    """Graph-Augmented Dual Memory Network with dual-path processing."""

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_entities: int = 9,
        num_relations: int = 10,
        num_classes: int = 5,
        gnn_type: str = "rgcn",
        num_gnn_layers: int = 3,
        num_reasoning_hops: int = 4,
        fusion_method: str = "gate",
        dropout_rate: float = 0.1,
        entity_loss_weight: float = 0.1,
        relation_loss_weight: float = 0.1,
        freeze_bert: bool = True,  # 默认冻结BERT
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.entity_loss_weight = entity_loss_weight
        self.relation_loss_weight = relation_loss_weight

        # Document encoder (BERT-based)
        self.document_encoder = DocumentEncoder(
            model_name=bert_model_name,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            freeze_bert=freeze_bert
        )

        # Structure extractor for entities and relations
        self.structure_extractor = StructureExtractor(
            hidden_size=hidden_size,
            num_entity_types=num_entities,
            num_relation_types=num_relations,
            dropout_rate=dropout_rate
        )

        # Graph writer to convert structures to graph format
        self.graph_writer = GraphWriter(
            hidden_size=hidden_size,
            num_entity_types=num_entities,
            num_relation_types=num_relations
        )

        # Graph memory for processing structured knowledge
        self.graph_memory = GraphMemory(
            hidden_size=hidden_size,
            num_relation_types=num_relations,
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            dropout_rate=dropout_rate
        )

        # Dual memory system (optional, can be integrated later)
        self.dual_memory = DualMemorySystem(
            hidden_size=hidden_size,
            memory_size=hidden_size,
            dropout_rate=dropout_rate
        )

        # Reasoning module with PathFinder and GraphReader
        self.reasoning_module = ReasoningModule(
            hidden_size=hidden_size,
            max_hops=num_reasoning_hops,
            fusion_method=fusion_method,
            dropout_rate=dropout_rate
        )

        # 🚀 持久化图记忆系统 (新增)
        self.persistent_graph_memory = PersistentGraphMemory(
            node_dim=hidden_size,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 🚀 实体对齐器 (新增)
        self.entity_aligner = EntityAligner(similarity_threshold=0.85)

        # 🚀 批处理图更新器 (新增)
        self.batch_graph_updater = BatchGraphUpdater(
            self.persistent_graph_memory, self.entity_aligner
        )
        
        # Stable classification head with normalization
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),  # 添加层归一化
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 4),  # 中间层
            nn.GELU(),  # 平滑激活函数
            nn.Dropout(dropout_rate * 0.5),  # 较小的dropout
            nn.Linear(hidden_size // 4, num_classes)
        )

        # Initialize all model weights properly
        self.apply(self._init_weights)

        # Loss functions
        self.main_loss_fn = nn.CrossEntropyLoss()
        self.entity_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.relation_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def _init_weights(self, module):
        """Initialize weights for all modules."""
        if isinstance(module, nn.Linear):
            # Use normal initialization with proper scaling
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        entity_spans: torch.Tensor,
        entity_labels: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GDM-Net with dual-path processing.

        # Check all inputs for NaN/Inf at the very beginning
        inputs_to_check = {
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'doc_input_ids': doc_input_ids,
            'doc_attention_mask': doc_attention_mask,
            'entity_spans': entity_spans
        }

        for name, tensor in inputs_to_check.items():
            if torch.isnan(tensor).any():
                print(f"CRITICAL: NaN detected in input {name}")
                print(f"  Shape: {tensor.shape}")
                print(f"  NaN count: {torch.isnan(tensor).sum()}")
                # Replace NaN with safe values
                if 'input_ids' in name:
                    tensor = torch.where(torch.isnan(tensor), torch.ones_like(tensor), tensor)
                elif 'attention_mask' in name:
                    tensor = torch.where(torch.isnan(tensor), torch.ones_like(tensor), tensor)
                elif 'spans' in name:
                    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

                # Update the actual variables
                if name == 'query_input_ids':
                    query_input_ids = tensor
                elif name == 'query_attention_mask':
                    query_attention_mask = tensor
                elif name == 'doc_input_ids':
                    doc_input_ids = tensor
                elif name == 'doc_attention_mask':
                    doc_attention_mask = tensor
                elif name == 'entity_spans':
                    entity_spans = tensor

            if torch.isinf(tensor).any():
                print(f"CRITICAL: Inf detected in input {name}")
                print(f"  Shape: {tensor.shape}")
                print(f"  Inf count: {torch.isinf(tensor).sum()}")
                # Replace Inf with safe values
                if 'input_ids' in name:
                    tensor = torch.where(torch.isinf(tensor), torch.ones_like(tensor), tensor)
                elif 'attention_mask' in name:
                    tensor = torch.where(torch.isinf(tensor), torch.ones_like(tensor), tensor)
                elif 'spans' in name:
                    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)

                # Update the actual variables
                if name == 'query_input_ids':
                    query_input_ids = tensor
                elif name == 'query_attention_mask':
                    query_attention_mask = tensor
                elif name == 'doc_input_ids':
                    doc_input_ids = tensor
                elif name == 'doc_attention_mask':
                    doc_attention_mask = tensor
                elif name == 'entity_spans':
                    entity_spans = tensor

        Args:
            query_input_ids: [batch_size, query_len]
            query_attention_mask: [batch_size, query_len]
            doc_input_ids: [batch_size, doc_len]
            doc_attention_mask: [batch_size, doc_len]
            entity_spans: [batch_size, num_entities, 2]
            entity_labels: [batch_size, seq_len] (optional, for training)
            relation_labels: [batch_size, num_pairs] (optional, for training)

        Returns:
            Dictionary containing logits and intermediate representations
        """

        # Step 1: Document encoding (implicit path)
        doc_pooled, doc_sequence = self.document_encoder.encode_document(
            doc_input_ids, doc_attention_mask
        )

        # Step 2: Query encoding
        query_pooled, query_sequence = self.document_encoder.encode_query(
            query_input_ids, query_attention_mask
        ) if query_input_ids is not None else (doc_pooled, doc_sequence)

        # Step 3: Structure extraction (explicit path) - 使用SpaCy + 适配器
        # 提取原始文本用于SpaCy处理
        doc_texts = kwargs.get('doc_texts', None)

        # 简化的调试信息
        if doc_texts is None:
            print("⚠️ No doc_texts received in GDMNet")

        entity_logits, relation_logits, entities_batch, relations_batch = self.structure_extractor(
            doc_sequence, doc_attention_mask, entity_spans, input_texts=doc_texts
        )

        # 🔍 调试信息：检查实体关系提取质量
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        self._debug_step += 1

        if self._debug_step % 100 == 0:
            avg_entities = sum(len(entities) for entities in entities_batch) / len(entities_batch) if entities_batch else 0
            avg_relations = sum(len(relations) for relations in relations_batch) / len(relations_batch) if relations_batch else 0
            print(f"🔍 Step {self._debug_step}: Avg entities={avg_entities:.1f}, Avg relations={avg_relations:.1f}")

        # Step 4: Graph construction
        node_features, edge_index, edge_type, batch_indices = self.graph_writer(
            entities_batch, relations_batch, doc_sequence
        )

        # 🔍 调试信息：检查图构建结果
        if hasattr(self, '_debug_graph_count'):
            self._debug_graph_count += 1
        else:
            self._debug_graph_count = 1

        if self._debug_graph_count <= 3:  # 只打印前3次的调试信息
            print(f"🔍 Graph Construction Debug {self._debug_graph_count}:")
            print(f"  - node_features shape: {node_features.shape}")
            print(f"  - batch_indices length: {len(batch_indices)}")
            print(f"  - edge_index shape: {edge_index.shape}")
            print(f"  - Shapes match: {node_features.size(0) == len(batch_indices)}")

        # Step 5: Graph memory processing
        updated_node_features = self.graph_memory(
            node_features, edge_index, edge_type, batch_indices
        )

        # 🚀 Step 5.5: 持久化图记忆更新和查询 (新增)
        if self.training:
            # 🚀 训练时：使用批处理更新器更新全局图记忆
            current_node_idx = 0
            for i, (entities, relations) in enumerate(zip(entities_batch, relations_batch)):
                num_entities = len(entities) if entities else 0
                if num_entities > 0:
                    # 🔧 确保不超出节点特征的范围
                    end_idx = current_node_idx + num_entities
                    if end_idx <= updated_node_features.size(0):
                        sample_node_features = updated_node_features[current_node_idx:end_idx]

                        # 添加到批处理更新器
                        self.batch_graph_updater.add_batch_sample(
                            entities, relations, sample_node_features
                        )
                    else:
                        print(f"⚠️ Node feature index out of range: {end_idx} > {updated_node_features.size(0)}")

                    current_node_idx = end_idx

            # 每隔一定步数或batch结束时批量更新
            if self.batch_graph_updater.get_batch_size() >= 2:  # 🔧 减少批量大小以避免累积过多数据
                try:
                    update_stats = self.batch_graph_updater.flush_batch()
                    if hasattr(self, '_debug_step') and self._debug_step % 1000 == 0:
                        print(f"🔄 Graph update: +{update_stats['nodes_added']} nodes, +{update_stats['edges_added']} edges")
                except Exception as e:
                    print(f"❌ Graph update failed: {e}")
                    # 清空批处理队列以避免重复错误
                    self.batch_graph_updater.clear_batch()

        # 查询相关的全局图子图用于推理增强
        query_embedding = query_pooled[0].cpu().detach().numpy()  # 使用第一个查询
        global_node_features, global_edge_index, global_edge_type, global_node_ids = \
            self.persistent_graph_memory.get_subgraph_for_query(query_embedding, top_k=20)

        # 将全局图信息与局部图信息结合 (简单拼接)
        if global_node_features.size(0) > 0:
            # 拼接局部和全局节点特征
            combined_node_features = torch.cat([updated_node_features, global_node_features], dim=0)

            # 🔧 关键修复：同步调整batch_indices
            # 为全局节点创建batch_indices（使用-1表示全局节点）
            global_batch_indices = torch.full(
                (global_node_features.size(0),),
                -1,  # 使用-1标记全局节点
                dtype=batch_indices.dtype,
                device=batch_indices.device
            )
            combined_batch_indices = torch.cat([batch_indices, global_batch_indices], dim=0)

            # 调整全局边索引以适应拼接后的节点索引
            if global_edge_index.size(1) > 0:
                global_edge_index_adjusted = global_edge_index + updated_node_features.size(0)
                combined_edge_index = torch.cat([edge_index, global_edge_index_adjusted], dim=1)
                combined_edge_type = torch.cat([edge_type, global_edge_type], dim=0)
            else:
                combined_edge_index = edge_index
                combined_edge_type = edge_type
        else:
            combined_node_features = updated_node_features
            combined_edge_index = edge_index
            combined_edge_type = edge_type
            combined_batch_indices = batch_indices

        # Step 6: Multi-hop reasoning (使用组合的局部+全局图特征)
        fused_representation, path_representation, graph_representation = self.reasoning_module(
            query_pooled, doc_pooled, combined_node_features, combined_edge_index, combined_edge_type, combined_batch_indices
        )

        # Step 7: NEW - Dual Memory Processing (使用推理结果)
        memory_output, episodic_output, semantic_output = self.dual_memory(
            doc_representation=doc_pooled,
            query_representation=query_pooled,
            graph_representation=graph_representation,
            path_representation=path_representation
        )

        # Step 8: Enhanced Final Classification with Memory Integration
        # 将记忆输出与原始融合表示结合
        enhanced_representation = fused_representation + memory_output

        # 确保维度正确
        if enhanced_representation.size(-1) != self.hidden_size:
            print(f"WARNING: enhanced_representation dimension mismatch! Expected {self.hidden_size}, got {enhanced_representation.size(-1)}")
            # Project to correct dimension
            if not hasattr(self, 'emergency_projection'):
                self.emergency_projection = nn.Linear(enhanced_representation.size(-1), self.hidden_size).to(enhanced_representation.device)
            enhanced_representation = self.emergency_projection(enhanced_representation)

        # 🔥 记忆增强的分类器应用
        # 标准化输入
        enhanced_representation = F.layer_norm(enhanced_representation, [enhanced_representation.size(-1)])

        # 直接应用分类器
        logits = self.classifier(enhanced_representation)

        # Prepare comprehensive outputs showcasing dual-path processing
        outputs = {
            # 最终输出
            'logits': logits,

            # 辅助任务输出
            'entity_logits': entity_logits,
            'relation_logits': relation_logits,

            # 隐式路径 (Implicit Path) - BERT编码结果
            'doc_representation': doc_pooled,
            'query_representation': query_pooled,

            # 显式路径 (Explicit Path) - 结构化知识
            'entities_batch': entities_batch,
            'relations_batch': relations_batch,
            'updated_node_features': updated_node_features,

            # 图神经网络处理结果
            'graph_representation': graph_representation,

            # 多跳推理结果
            'path_representation': path_representation,

            # 双记忆系统输出
            'memory_output': memory_output,
            'episodic_output': episodic_output,
            'semantic_output': semantic_output,

            # 最终融合表示
            'fused_representation': fused_representation,

            # 图构建信息
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'batch_indices': batch_indices
        }

        return outputs

    # 🚀 持久化图记忆管理方法 (新增)
    def save_graph_memory(self, filepath: str):
        """保存持久化图记忆到磁盘"""
        self.persistent_graph_memory.save_to_disk(filepath)
        print(f"✅ Graph memory saved to {filepath}")

    def load_graph_memory(self, filepath: str):
        """从磁盘加载持久化图记忆"""
        self.persistent_graph_memory.load_from_disk(filepath)
        print(f"✅ Graph memory loaded from {filepath}")

    def get_graph_memory_stats(self) -> dict:
        """获取图记忆统计信息"""
        return {
            'num_nodes': len(self.persistent_graph_memory.nodes),
            'num_edges': len(self.persistent_graph_memory.edges),
            'num_entity_types': len(self.persistent_graph_memory.entity_type_index),
            'batch_queue_size': self.batch_graph_updater.get_batch_size()
        }

    def flush_graph_memory_batch(self):
        """强制清空批处理队列"""
        if self.batch_graph_updater.get_batch_size() > 0:
            update_stats = self.batch_graph_updater.flush_batch()
            print(f"🔄 Final graph update: +{update_stats['nodes_added']} nodes, +{update_stats['edges_added']} edges")
            return update_stats
        return {'nodes_added': 0, 'nodes_updated': 0, 'edges_added': 0, 'edges_updated': 0}

    def _stable_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        手动实现的数值稳定cross entropy损失

        Args:
            logits: [batch_size, num_classes] 原始logits
            labels: [batch_size] 标签索引

        Returns:
            scalar loss tensor
        """
        # 1. 基本检查
        batch_size, num_classes = logits.shape

        # 2. 数值稳定的softmax计算
        # 减去最大值防止exp溢出
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        shifted_logits = logits - max_logits

        # 计算exp，添加小的epsilon防止log(0)
        exp_logits = torch.exp(shifted_logits)
        sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)

        # 计算log概率
        log_probs = shifted_logits - torch.log(sum_exp + 1e-8)

        # 3. 手动实现negative log likelihood
        # 创建one-hot编码
        labels_one_hot = torch.zeros_like(logits)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # 计算负对数似然
        nll = -torch.sum(labels_one_hot * log_probs, dim=1)

        # 4. 返回平均损失
        loss = torch.mean(nll)

        # 5. 最终安全检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"🚨 CRITICAL: Even stable cross entropy failed!")
            print(f"  Logits stats: min={logits.min():.6f}, max={logits.max():.6f}, std={logits.std():.6f}")
            print(f"  Labels: {labels}")
            # 返回一个合理的常数损失
            return torch.tensor(1.609, device=logits.device, requires_grad=True)  # ln(5)

        return loss

    def _stable_cross_entropy_multi(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        稳定的多类别cross entropy，用于辅助任务
        处理ignore_index=0的情况
        """
        # 过滤掉ignore_index=0的样本
        valid_mask = (labels != 0)
        if not valid_mask.any():
            # 如果没有有效标签，返回零损失
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # 使用稳定的cross entropy
        return self._stable_cross_entropy(valid_logits, valid_labels)
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        entity_labels: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with main task and auxiliary tasks.

        Args:
            outputs: Model outputs dictionary
            labels: Main task labels [batch_size]
            entity_labels: Entity labels [batch_size, seq_len] (optional)
            relation_labels: Relation labels [batch_size, num_pairs] (optional)

        Returns:
            Dictionary containing different loss components
        """

        # 🔥 完全重写的数值稳定损失计算
        logits = outputs['logits']

        # 手动实现最稳定的cross entropy
        main_loss = self._stable_cross_entropy(logits, labels)

        # 简单调试（只在前3次）
        if not hasattr(self, '_loss_debug_count'):
            self._loss_debug_count = 0
        if self._loss_debug_count < 3:
            print(f"✅ Stable loss {self._loss_debug_count}: {main_loss.item():.6f}")
            self._loss_debug_count += 1

        # 🎯 重新启用辅助损失，提升整体性能
        total_loss = main_loss
        loss_dict = {
            'main_loss': main_loss
        }

        # Entity extraction auxiliary loss
        if entity_labels is not None and 'entity_logits' in outputs:
            entity_logits = outputs['entity_logits']
            batch_size, seq_len, num_classes = entity_logits.shape

            # Flatten for loss computation
            entity_logits_flat = entity_logits.view(-1, num_classes)
            entity_labels_flat = entity_labels.view(-1)

            entity_loss = self._stable_cross_entropy_multi(entity_logits_flat, entity_labels_flat)
            total_loss += self.entity_loss_weight * entity_loss
            loss_dict['entity_loss'] = entity_loss

        # Relation extraction auxiliary loss
        if relation_labels is not None and 'relation_logits' in outputs:
            relation_logits = outputs['relation_logits']
            if relation_logits.numel() > 0 and relation_labels.numel() > 0:
                # Ensure shapes match
                min_size = min(relation_logits.size(1), relation_labels.size(1))
                if min_size > 0:
                    relation_logits_subset = relation_logits[:, :min_size]
                    relation_labels_subset = relation_labels[:, :min_size]

                    relation_logits_flat = relation_logits_subset.view(-1, relation_logits.size(-1))
                    relation_labels_flat = relation_labels_subset.view(-1)

                    relation_loss = self._stable_cross_entropy_multi(relation_logits_flat, relation_labels_flat)
                    total_loss += self.relation_loss_weight * relation_loss
                    loss_dict['relation_loss'] = relation_loss

        # 可选：记忆一致性损失（确保情节记忆和语义记忆的协调）
        if 'episodic_output' in outputs and 'semantic_output' in outputs:
            episodic_out = outputs['episodic_output']  # [batch_size, memory_size]
            semantic_out = outputs['semantic_output']  # [batch_size, memory_size]

            # 计算记忆间的相似性损失（鼓励适度的一致性，但不完全相同）
            memory_consistency_loss = F.mse_loss(episodic_out, semantic_out)
            # 使用较小的权重，因为我们希望记忆有所区别
            total_loss += 0.01 * memory_consistency_loss
            loss_dict['memory_consistency_loss'] = memory_consistency_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def predict(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        entity_spans: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Make predictions."""

        with torch.no_grad():
            outputs = self.forward(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                doc_input_ids=doc_input_ids,
                doc_attention_mask=doc_attention_mask,
                entity_spans=entity_spans,
                **kwargs
            )

            predictions = torch.argmax(outputs['logits'], dim=1)

        return predictions
