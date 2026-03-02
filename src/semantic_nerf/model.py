"""轻量 Semantic NeRF：MLP 输出密度、RGB、语义 logits；体渲染 RGB + 语义。"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    pe = []
    for i in range(L):
        pe.append(torch.sin(2 ** i * math.pi * x))
        pe.append(torch.cos(2 ** i * math.pi * x))
    return torch.cat(pe, dim=-1)


class SemanticNeRF(nn.Module):
    """密度 + RGB + 语义 logits；输入 3D 点与方向，输出 (density, rgb, sem_logits)。"""

    def __init__(
        self,
        num_classes: int = 150,
        pos_L: int = 10,
        dir_L: int = 4,
        hidden: int = 256,
        num_layers: int = 6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pos_L = pos_L
        self.dir_L = dir_L
        pos_dim = 3 * 2 * pos_L
        dir_dim = 3 * 2 * dir_L
        self.pts_enc_dim = pos_dim
        self.dir_enc_dim = dir_dim
        dims = [pos_dim] + [hidden] * num_layers
        self.pts_linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.density_head = nn.Linear(hidden, 1)
        self.rgb_feat = nn.Linear(hidden, hidden)
        self.rgb_head = nn.Linear(hidden + dir_dim, 3)
        self.sem_head = nn.Linear(hidden, num_classes)

    def forward(
        self,
        pts: torch.Tensor,
        dirs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pts: (N, 3), dirs: (N, 3) 已单位化
        返回 density (N,1), rgb (N,3), sem_logits (N, num_classes)
        """
        pe_pts = positional_encoding(pts, self.pos_L)
        pe_dir = positional_encoding(dirs, self.dir_L)
        h = pe_pts
        for lin in self.pts_linears:
            h = torch.relu(lin(h))
        density = torch.nn.functional.softplus(self.density_head(h))
        rgb_feat = self.rgb_feat(h)
        rgb = torch.sigmoid(self.rgb_head(torch.cat([rgb_feat, pe_dir], dim=-1)))
        sem_logits = self.sem_head(h)
        return density, rgb, sem_logits


def volume_rendering(
    density: torch.Tensor,
    rgb: torch.Tensor,
    sem_logits: torch.Tensor,
    t_vals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    density (N,S,1), rgb (N,S,3), sem_logits (N,S,C), t_vals (N,S)
    返回 rgb_rendered (N,3), sem_rendered (N,C), weights (N,S)
    """
    dt = t_vals[:, 1:] - t_vals[:, :-1]
    dt = torch.cat([dt, dt[:, -1:]], dim=1)
    alpha = 1 - torch.exp(-density.squeeze(-1) * dt.clamp(min=1e-6))
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1] + 1e-8], dim=1),
        dim=1,
    )
    weights = alpha * trans
    rgb_rendered = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    sem_rendered = (weights.unsqueeze(-1) * sem_logits).sum(dim=1)
    return rgb_rendered, sem_rendered, weights
