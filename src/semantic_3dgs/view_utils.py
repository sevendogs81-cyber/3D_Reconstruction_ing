"""与 splatfacto/gsplat 一致的 view 矩阵：c2w 转 world2camera（含 Y/Z 翻转）。"""

from __future__ import annotations

import numpy as np
import torch


def get_viewmat(c2w: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    将 Nerfstudio 的 camera_to_world (4x4) 转为 gsplat 的 world2camera。
    与 nerfstudio.models.splatfacto.get_viewmat 一致：对 R 应用 [1, -1, -1] 后取逆。
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w.astype(np.float32))
    c2w = c2w.float()
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    R = c2w[:, :3, :3]
    T = c2w[:, :3, 3:4]
    # splatfacto: flip y and z
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat
