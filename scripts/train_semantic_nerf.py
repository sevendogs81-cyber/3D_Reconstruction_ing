#!/usr/bin/env python3
"""
训练时语义 NeRF：用 2D 语义标签监督体渲染的语义输出。

要求：已运行 run_semantic_labeling.py 得到 semantic_scene.json 与 label_maps/*.npy，
且 ns_processed 与 semantic 目录在约定路径下。

用法：
  python scripts/train_semantic_nerf.py \\
    --processed-dir mipnerf360/db/playroom/ns_processed \\
    --semantic-dir mipnerf360/db/playroom/semantic \\
    --output-dir mipnerf360/db/playroom/semantic_nerf_runs \\
    --num-classes 150 \\
    --steps 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from src.semantic_nerf.dataset import SemanticNeRFDataset
from src.semantic_nerf.model import SemanticNeRF, volume_rendering


def get_rays(c2w: torch.Tensor, K: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """c2w (4,4), K (3,3). 返回 origins (H*W,3), dirs (H*W,3) 单位向量。"""
    device = c2w.device
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()
    u = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    v = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    u, v = torch.meshgrid(u, v, indexing="xy")
    u = u.reshape(-1)
    v = v.reshape(-1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)
    dir_cam = torch.stack([x, y, z], dim=-1)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    dir_world = (R @ dir_cam.T).T
    dir_world = F.normalize(dir_world, dim=-1)
    origins = t.unsqueeze(0).expand(dir_world.shape[0], -1)
    return origins, dir_world


def sample_along_rays(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    near: float = 0.1,
    far: float = 6.0,
    N_samples: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """origins/dirs (N,3). 返回 pts (N,S,3), t_vals (N,S)。"""
    N = origins.shape[0]
    device = origins.device
    t_vals = torch.linspace(near, far, N_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(N, -1)
    jitter = torch.rand_like(t_vals, device=device) * (far - near) / N_samples
    t_vals = t_vals + jitter
    pts = origins.unsqueeze(1) + t_vals.unsqueeze(2) * dirs.unsqueeze(1)
    return pts, t_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--semantic-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-rays", type=int, default=1024)
    parser.add_argument("--N-samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = REPO_ROOT / processed_dir
    semantic_dir = Path(args.semantic_dir) if args.semantic_dir else processed_dir.parent / "semantic"
    if not semantic_dir.is_absolute():
        semantic_dir = REPO_ROOT / semantic_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = SemanticNeRFDataset(
        processed_dir,
        semantic_root=semantic_dir,
        num_classes=args.num_classes,
        split="train",
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    model = SemanticNeRF(num_classes=args.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_iter = iter(train_loader)
    for step in range(args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        image = batch["image"][0].to(device)
        c2w = batch["c2w"][0].to(device)
        K = batch["K"][0].to(device)
        semantics = batch["semantics"][0].to(device)
        H, W = batch["height"][0].item(), batch["width"][0].item()

        origins, dirs = get_rays(c2w, K, H, W)
        total_rays = H * W
        perm = torch.randperm(total_rays, device=device)[: args.batch_rays]
        origins = origins[perm]
        dirs = dirs[perm]

        pts, t_vals = sample_along_rays(origins, dirs, N_samples=args.N_samples)
        N, S, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = dirs.unsqueeze(1).expand(-1, S, -1).reshape(-1, 3)

        density, rgb, sem_logits = model(pts_flat, dirs_flat)
        density = density.reshape(N, S, 1)
        rgb = rgb.reshape(N, S, 3)
        sem_logits = sem_logits.reshape(N, S, args.num_classes)

        rgb_rendered, sem_rendered, _ = volume_rendering(density, rgb, sem_logits, t_vals)

        # 取对应像素的 GT
        v = perm // W
        u = perm % W
        gt_rgb = image[v, u]
        gt_sem = semantics[v, u]

        loss_rgb = mse(rgb_rendered, gt_rgb)
        loss_sem = ce(sem_rendered, gt_sem)
        loss = loss_rgb + 0.5 * loss_sem

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 200 == 0:
            print(f"step {step+1}/{args.steps} loss_rgb={loss_rgb.item():.4f} loss_sem={loss_sem.item():.4f}")

    ckpt_path = output_dir / "semantic_nerf.pt"
    torch.save({"model": model.state_dict(), "num_classes": args.num_classes}, ckpt_path)
    print(f"保存 checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
