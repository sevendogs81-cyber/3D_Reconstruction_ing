#!/usr/bin/env python3
"""
从训练好的语义 3DGS 渲染指定视角的 RGB 与按类别着色的语义图，保存为图片便于在 Viewer 中查看。

用法：
  python scripts/render_semantic_3dgs.py \\
    --checkpoint mipnerf360/db/playroom/semantic_3dgs_runs/semantic_3dgs.pt \\
    --processed-dir mipnerf360/db/playroom/ns_processed \\
    --output-dir mipnerf360/db/playroom/semantic_3dgs_runs/renders \\
    --frame-idx 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_3dgs.view_utils import get_viewmat

try:
    from gsplat.rendering import rasterization
    HAS_GSPLAT = True
except ImportError:
    HAS_GSPLAT = False


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not HAS_GSPLAT:
        print("需要 gsplat（conda activate worldrecon）")
        sys.exit(1)

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
    W, H = int(meta["w"]), int(meta["h"])
    fl_x, fl_y = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
    idx = min(args.frame_idx, len(frames) - 1)
    frame = frames[idx]
    c2w = np.array(frame["transform_matrix"], dtype=np.float32)
    # 与 splatfacto/训练时一致：get_viewmat(c2w)
    viewmat = get_viewmat(c2w)

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "renders"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = int(ckpt["num_classes"])
    means = ckpt["means"].to(device)
    quats = ckpt["quats"].to(device)
    scales = ckpt["scales"].to(device)
    opacities = ckpt["opacities"].to(device)
    rgb = ckpt["rgb"].to(device)
    semantic_logits = ckpt["semantic_logits"].to(device)

    viewmat = viewmat.to(device)
    K_t = torch.from_numpy(K).unsqueeze(0).to(device)

    with torch.no_grad():
        out_rgb, out_alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=rgb,
            viewmats=viewmat,
            Ks=K_t,
            width=W,
            height=H,
            sh_degree=None,
        )
        out_sem, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=semantic_logits,
            viewmats=viewmat,
            Ks=K_t,
            width=W,
            height=H,
            sh_degree=None,
        )

    rgb_img = out_rgb[0].clamp(0, 1).cpu().numpy()
    alpha = out_alpha[0].squeeze().cpu().numpy()
    sem_logits_img = out_sem[0].cpu().numpy()
    sem_ids = np.argmax(sem_logits_img, axis=-1)
    sem_conf = np.max(np.exp(sem_logits_img) / (np.exp(sem_logits_img).sum(axis=-1, keepdims=True) + 1e-8), axis=-1)
    colormap = get_class_colormap(num_classes)
    sem_rgb = colormap[np.clip(sem_ids, 0, num_classes - 1)]

    alpha_thresh = 0.2
    low_alpha = alpha < alpha_thresh
    sem_rgb[low_alpha] = 0.55
    sem_rgb = (sem_rgb * 255).astype(np.uint8)

    blend = 0.45
    overlay = rgb_img * (1 - blend) + (sem_rgb.astype(np.float64) / 255.0) * blend
    overlay[low_alpha] = rgb_img[low_alpha]
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    Image.fromarray((rgb_img * 255).astype(np.uint8)).save(output_dir / "rgb.png")
    Image.fromarray(sem_rgb).save(output_dir / "semantic_colormap.png")
    Image.fromarray(overlay).save(output_dir / "semantic_overlay_on_rgb.png")
    sidebyside = np.concatenate([rgb_img, sem_rgb.astype(np.float64) / 255.0], axis=1)
    Image.fromarray((sidebyside * 255).astype(np.uint8)).save(output_dir / "rgb_semantic_sidebyside.png")
    print(f"已保存: rgb.png, semantic_colormap.png（低 alpha 处为灰）, semantic_overlay_on_rgb.png（语义叠在 RGB 上）, rgb_semantic_sidebyside.png")
    print(f"目录: {output_dir}")


if __name__ == "__main__":
    main()
