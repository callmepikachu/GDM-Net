import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DualMemorySystem(nn.Module):
    """Dual memory system combining episodic and semantic memory."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        memory_size: int = 512,
        num_memory_slots: int = 64,
        dropout_rate: float = 0.1,
        use_gating: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_memory_slots = num_memory_slots
        self.use_gating = use_gating
        
        # Episodic memory - stores specific instances and experiences
        self.episodic_memory = nn.Parameter(
            torch.randn(num_memory_slots, memory_size) * 0.1
        )
        
        # Semantic memory - stores general knowledge and patterns
        self.semantic_memory = nn.Parameter(
            torch.randn(num_memory_slots, memory_size) * 0.1
        )
        
        # Memory access mechanisms
        self.episodic_query_projection = nn.Linear(hidden_size, memory_size)
        self.semantic_query_projection = nn.Linear(hidden_size, memory_size)
        
        # Memory update mechanisms
        self.episodic_update_gate = nn.Linear(hidden_size + memory_size, memory_size)
        self.semantic_update_gate = nn.Linear(hidden_size + memory_size, memory_size)
        
        # Fusion mechanisms
        if self.use_gating:
            self.fusion_gate = nn.Linear(memory_size * 2, 1)

        # The fusion projection needs to handle concatenated episodic + semantic + combined features
        # This can be memory_size * 2 (episodic + semantic) or memory_size * 3 depending on gating
        self.fusion_projection = nn.Linear(memory_size * 3, hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize memory
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize memory slots."""
        nn.init.xavier_uniform_(self.episodic_memory)
        nn.init.xavier_uniform_(self.semantic_memory)
    
    def forward(
        self,
        query_representations: torch.Tensor,
        update_memory: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through dual memory system.
        
        Args:
            query_representations: [batch_size, hidden_size]
            update_memory: Whether to update memory slots
        
        Returns:
            fused_output: [batch_size, hidden_size]
            episodic_output: [batch_size, memory_size]
            semantic_output: [batch_size, memory_size]
        """
        
        batch_size = query_representations.size(0)
        
        # Project queries for memory access
        episodic_query = self.episodic_query_projection(query_representations)
        semantic_query = self.semantic_query_projection(query_representations)
        
        # Access episodic memory with enhanced numerical stability
        episodic_scores = torch.matmul(episodic_query, self.episodic_memory.t())
        episodic_scores = torch.clamp(episodic_scores, min=-5, max=5)  # 更保守的范围
        # 添加温度缩放提高稳定性
        episodic_attention = F.softmax(episodic_scores / 1.0, dim=1)
        episodic_output = torch.matmul(episodic_attention, self.episodic_memory)

        # Access semantic memory with enhanced numerical stability
        semantic_scores = torch.matmul(semantic_query, self.semantic_memory.t())
        semantic_scores = torch.clamp(semantic_scores, min=-5, max=5)  # 更保守的范围
        semantic_attention = F.softmax(semantic_scores / 1.0, dim=1)
        semantic_output = torch.matmul(semantic_attention, self.semantic_memory)

        # 检查输出的数值稳定性
        if torch.isnan(episodic_output).any() or torch.isinf(episodic_output).any():
            print("WARNING: NaN/Inf in episodic_output")
            episodic_output = torch.zeros_like(episodic_output)
        if torch.isnan(semantic_output).any() or torch.isinf(semantic_output).any():
            print("WARNING: NaN/Inf in semantic_output")
            semantic_output = torch.zeros_like(semantic_output)
        
        # Update memory if specified
        if update_memory and self.training:
            self._update_memory(
                query_representations,
                episodic_output,
                semantic_output,
                episodic_attention,
                semantic_attention
            )
        
        # Fuse episodic and semantic outputs
        fused_output = self._fuse_memories(episodic_output, semantic_output)
        
        return fused_output, episodic_output, semantic_output
    
    def _update_memory(
        self,
        query_representations: torch.Tensor,
        episodic_output: torch.Tensor,
        semantic_output: torch.Tensor,
        episodic_attention: torch.Tensor,
        semantic_attention: torch.Tensor
    ):
        """Update memory slots based on current inputs."""
        
        # Compute update vectors
        episodic_input = torch.cat([query_representations, episodic_output], dim=1)
        episodic_update = torch.tanh(self.episodic_update_gate(episodic_input))
        
        semantic_input = torch.cat([query_representations, semantic_output], dim=1)
        semantic_update = torch.tanh(self.semantic_update_gate(semantic_input))
        
        # Update episodic memory (more dynamic, instance-specific)
        episodic_update_weights = episodic_attention.mean(dim=0)  # [num_memory_slots]
        episodic_update_vector = episodic_update.mean(dim=0)  # [memory_size]
        
        self.episodic_memory.data = (
            self.episodic_memory.data * (1 - episodic_update_weights.unsqueeze(1)) +
            episodic_update_vector.unsqueeze(0) * episodic_update_weights.unsqueeze(1)
        )
        
        # Update semantic memory (more stable, general knowledge)
        semantic_update_weights = semantic_attention.mean(dim=0) * 0.1  # Slower update
        semantic_update_vector = semantic_update.mean(dim=0)
        
        self.semantic_memory.data = (
            self.semantic_memory.data * (1 - semantic_update_weights.unsqueeze(1)) +
            semantic_update_vector.unsqueeze(0) * semantic_update_weights.unsqueeze(1)
        )
    
    def _fuse_memories(
        self,
        episodic_output: torch.Tensor,
        semantic_output: torch.Tensor
    ) -> torch.Tensor:
        """Fuse episodic and semantic memory outputs."""

        # Concatenate memory outputs
        combined = torch.cat([episodic_output, semantic_output], dim=1)

        if self.use_gating:
            # Compute fusion gate with numerical stability
            gate_input = torch.clamp(combined, min=-5, max=5)
            gate = torch.sigmoid(self.fusion_gate(gate_input))

            # Weighted combination with clamping
            weighted_combination = gate * episodic_output + (1 - gate) * semantic_output
            weighted_combination = torch.clamp(weighted_combination, min=-5, max=5)

            # Final fusion: episodic + semantic + weighted_combination
            fused = torch.cat([episodic_output, semantic_output, weighted_combination], dim=1)
        else:
            # Simple concatenation: episodic + semantic + zeros for consistency
            zeros = torch.zeros_like(episodic_output)
            fused = torch.cat([episodic_output, semantic_output, zeros], dim=1)

        # 检查融合结果的数值稳定性
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            print("WARNING: NaN/Inf in fused memory output")
            fused = torch.zeros_like(fused)

        # Project to output size
        output = self.fusion_projection(fused)
        output = self.layer_norm(self.dropout(output))

        return output
    
    def reset_memory(self):
        """Reset memory to initial state."""
        self._initialize_memory()
    
    def get_memory_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current memory state."""
        return self.episodic_memory.clone(), self.semantic_memory.clone()
