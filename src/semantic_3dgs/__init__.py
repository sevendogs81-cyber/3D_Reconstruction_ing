"""稠密 3DGS + 语义：从 splatfacto 加载高斯，仅训语义头，gsplat 渲染。"""

from .gaussian_model import DenseSemanticGaussianModel
from .view_utils import get_viewmat

__all__ = ["DenseSemanticGaussianModel", "get_viewmat"]
