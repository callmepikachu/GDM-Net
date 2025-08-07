"""
GDM-Net: Graph-Augmented Dual Memory Network

A neural architecture that combines explicit graph memory with dual-path processing 
for structured information extraction and multi-hop reasoning over documents.
"""

from .model import GDMNet
from .encoder import DocumentEncoder
from .extractor import StructureExtractor
from .graph_memory import GraphMemory, GraphWriter
from .reasoning import PathFinder, GraphReader, ReasoningFusion

__version__ = "0.1.0"
__author__ = "GDM-Net Team"

__all__ = [
    "GDMNet",
    "DocumentEncoder", 
    "StructureExtractor",
    "GraphMemory",
    "GraphWriter",
    "PathFinder",
    "GraphReader", 
    "ReasoningFusion"
]
