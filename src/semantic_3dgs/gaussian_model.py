"""稠密 3DGS + 语义：从 splatfacto 加载的 means/quats/scales/opacities/rgb，仅语义 logits 可训。"""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseSemanticGaussianModel(nn.Module):
    """
    稠密高斯 + 语义：几何与 RGB 来自 splatfacto（固定），仅 semantic_logits (N, K) 可学习。
    供 gsplat 渲染：scales/opacities 为线性值（已 exp/sigmoid），colors 为 [0,1] RGB。
    """

    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales_linear: torch.Tensor,
        opacities_linear: torch.Tensor,
        rgb: torch.Tensor,
        num_classes: int,
        init_semantic: str = "zeros",
    ):
        super().__init__()
        N = means.shape[0]
        device = means.device
        self.num_classes = num_classes

        self.register_buffer("_means", means.float())
        self.register_buffer("_quats", quats.float() / (quats.norm(dim=-1, keepdim=True) + 1e-8))
        self.register_buffer("_scales", scales_linear.float().clamp(min=1e-6))
        self.register_buffer("_opacities", opacities_linear.float().clamp(0, 1))
        self.register_buffer("_rgb", rgb.float().clamp(0, 1))

        if init_semantic == "zeros":
            sem = torch.zeros(N, num_classes, device=device, dtype=torch.float32)
        else:
            sem = torch.randn(N, num_classes, device=device, dtype=torch.float32) * 0.01
        self._semantic_logits = nn.Parameter(sem)

    @property
    def means(self) -> torch.Tensor:
        return self._means

    @property
    def quats(self) -> torch.Tensor:
        return self._quats

    @property
    def scales(self) -> torch.Tensor:
        return self._scales

    @property
    def opacities(self) -> torch.Tensor:
        return self._opacities

    @property
    def rgb(self) -> torch.Tensor:
        return self._rgb

    @property
    def semantic_logits(self) -> torch.Tensor:
        return self._semantic_logits

    def get_rgb_features(self) -> torch.Tensor:
        """(N, 3) for RGB render."""
        return self._rgb

    def get_semantic_features(self) -> torch.Tensor:
        """(N, K) for semantic render."""
        return self._semantic_logits
