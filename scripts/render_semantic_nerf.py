#!/usr/bin/env python3
"""
从训练好的语义 NeRF 渲染指定视角的 RGB 与按类别着色的语义图，保存为图片便于在 Viewer 中查看。

用法：
  python scripts/render_semantic_nerf.py \\
    --checkpoint mipnerf360/db/playroom/semantic_nerf_runs/semantic_nerf.pt \\
    --processed-dir mipnerf360/db/playroom/ns_processed \\
    --output-dir mipnerf360/db/playroom/semantic_nerf_runs/renders \\
    --frame-idx 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_nerf.model import SemanticNeRF, volume_rendering


def get_class_colormap(num_classes: int) -> np.ndarray:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if num_classes <= 20:
            cmap = plt.get_cmap("tab20")
            colors = np.array([cmap(i % 20)[:3] for i in range(num_classes)], dtype=np.float64)
        else:
            cmap = plt.get_cmap("turbo")
            colors = np.array([cmap(i / max(num_classes - 1, 1))[:3] for i in range(num_classes)], dtype=np.float64)
        return colors
    except Exception:
        palette = [
            (1, 0, 0), (0, 0.8, 0), (0, 0, 1), (1, 0.8, 0), (1, 0, 0.8),
            (0, 0.8, 0.8), (0.6, 0.2, 0.8), (0.8, 0.6, 0), (0.2, 0.6, 0.4), (0.8, 0.2, 0.2),
        ]
        return np.array([palette[i % len(palette)] for i in range(num_classes)], dtype=np.float64)


def get_rays(c2w: torch.Tensor, K: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = c2w.device
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()
    u = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    v = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    u, v = torch.meshgrid(u, v, indexing="xy")
    u, v = u.reshape(-1), v.reshape(-1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)
    dir_cam = torch.stack([x, y, z], dim=-1)
    R, t = c2w[:3, :3], c2w[:3, 3]
    dir_world = (R @ dir_cam.T).T
    dir_world = F.normalize(dir_world, dim=-1)
    origins = t.unsqueeze(0).expand(dir_world.shape[0], -1)
    return origins, dir_world


def sample_along_rays(origins: torch.Tensor, dirs: torch.Tensor, near: float = 0.1, far: float = 6.0, N_samples: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    N = origins.shape[0]
    device = origins.device
    t_vals = torch.linspace(near, far, N_samples, device=device).unsqueeze(0).expand(N, -1)
    pts = origins.unsqueeze(1) + t_vals.unsqueeze(2) * dirs.unsqueeze(1)
    return pts, t_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--ray-chunk", type=int, default=8192)
    parser.add_argument("--N-samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"Checkpoint 不存在: {ckpt_path}")
        sys.exit(1)

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = REPO_ROOT / processed_dir
    with open(processed_dir / "transforms.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta["frames"]
    H, W = int(meta["h"]), int(meta["w"])
    fl_x, fl_y = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    K = torch.tensor([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=torch.float32)
    idx = min(args.frame_idx, len(frames) - 1)
    frame = frames[idx]
    c2w = torch.from_numpy(np.array(frame["transform_matrix"], dtype=np.float32))

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "renders"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = int(ckpt.get("num_classes", 150))
    model = SemanticNeRF(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    c2w = c2w.to(device)
    K = K.to(device)
    origins, dirs = get_rays(c2w, K, H, W)
    total = origins.shape[0]
    rgb_list, sem_list = [], []
    with torch.no_grad():
        for start in range(0, total, args.ray_chunk):
            end = min(start + args.ray_chunk, total)
            o = origins[start:end]
            d = dirs[start:end]
            pts, t_vals = sample_along_rays(o, d, N_samples=args.N_samples)
            N, S, _ = pts.shape
            density, rgb, sem_logits = model(pts.reshape(-1, 3), d.unsqueeze(1).expand(-1, S, -1).reshape(-1, 3))
            density = density.reshape(N, S, 1)
            rgb = rgb.reshape(N, S, 3)
            sem_logits = sem_logits.reshape(N, S, num_classes)
            r, s, _ = volume_rendering(density, rgb, sem_logits, t_vals)
            rgb_list.append(r.cpu())
            sem_list.append(s.cpu())
    rgb_full = torch.cat(rgb_list, dim=0)
    sem_full = torch.cat(sem_list, dim=0)
    rgb_img = rgb_full.reshape(H, W, 3).numpy().clip(0, 1)
    sem_ids = sem_full.argmax(dim=-1).reshape(H, W).numpy()
    colormap = get_class_colormap(num_classes)
    sem_rgb = colormap[np.clip(sem_ids, 0, num_classes - 1)]

    Image.fromarray((rgb_img * 255).astype(np.uint8)).save(output_dir / "rgb.png")
    Image.fromarray((sem_rgb * 255).astype(np.uint8)).save(output_dir / "semantic_colormap.png")
    sidebyside = np.concatenate([rgb_img, sem_rgb], axis=1)
    Image.fromarray((sidebyside * 255).astype(np.uint8)).save(output_dir / "rgb_semantic_sidebyside.png")
    print(f"已保存: {output_dir / 'rgb.png'}, {output_dir / 'semantic_colormap.png'}, {output_dir / 'rgb_semantic_sidebyside.png'}")


if __name__ == "__main__":
    main()
