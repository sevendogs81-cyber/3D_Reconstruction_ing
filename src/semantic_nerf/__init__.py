"""训练时语义 NeRF：轻量体渲染 + 语义头，用 2D 标签监督。"""

from .dataset import SemanticNeRFDataset
from .model import SemanticNeRF

__all__ = ["SemanticNeRF", "SemanticNeRFDataset"]
