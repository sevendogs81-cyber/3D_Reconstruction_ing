#!/usr/bin/env python3
"""
在 splatfacto 训好的稠密 3DGS 上挂语义头并训练，用 2D 标签监督。几何与 RGB 冻结，仅训 semantic_logits。

要求：
  - 已用 ns-train splatfacto 训好场景，得到 config.yml（含 load_dir）；
  - 已运行 run_semantic_labeling.py，有 semantic/label_maps/*.npy；
  - ns_processed/transforms.json 与 splatfacto 使用同一数据。

用法：
  python scripts/train_semantic_3dgs.py \\
    --splatfacto-config /path/to/ns_outs/.../config.yml \\
    --processed-dir mipnerf360/db/playroom/ns_processed \\
    --semantic-dir mipnerf360/db/playroom/semantic \\
    --output-dir mipnerf360/db/playroom/semantic_3dgs_runs \\
    --steps 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_3dgs.gaussian_model import DenseSemanticGaussianModel
from src.semantic_3dgs.view_utils import get_viewmat

try:
    from gsplat.rendering import rasterization
    HAS_GSPLAT = True
except ImportError:
    HAS_GSPLAT = False


def load_cameras(processed_dir: Path) -> Tuple[List[dict], int, int, float, float, float, float]:
    with open(processed_dir / "transforms.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta["frames"]
    w, h = int(meta["w"]), int(meta["h"])
    fl_x, fl_y = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    return frames, w, h, fl_x, fl_y, cx, cy


def get_K_tensor(fl_x: float, fl_y: float, cx: float, cy: float, device: torch.device) -> torch.Tensor:
    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
    return torch.from_numpy(K).unsqueeze(0).to(device)


def load_splatfacto_gaussians(config_path: Path, device: torch.device):
    """通过 nerfstudio eval_setup 加载 splatfacto pipeline，提取稠密高斯参数。"""
    try:
        from nerfstudio.utils.eval_utils import eval_setup
        from nerfstudio.models.splatfacto import SplatfactoModel
    except ImportError as e:
        raise ImportError("需要 nerfstudio（conda activate worldrecon）") from e

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"splatfacto config 不存在: {config_path}")

    _, pipeline, _, _ = eval_setup(config_path, test_mode="inference")
    model = pipeline.model
    if not isinstance(model, SplatfactoModel):
        raise TypeError("config 对应的模型不是 SplatfactoModel")

    with torch.no_grad():
        means = model.means.detach().to(device)
        quats = (model.quats / (model.quats.norm(dim=-1, keepdim=True) + 1e-8)).detach().to(device)
        scales_linear = torch.exp(model.scales.detach()).clamp(min=1e-6).to(device)
        opacities_linear = torch.sigmoid(model.opacities.detach()).squeeze(-1).clamp(0, 1).to(device)
        rgb = torch.clamp(model.colors.detach(), 0.0, 1.0).to(device)

    return means, quats, scales_linear, opacities_linear, rgb


def main():
    parser = argparse.ArgumentParser(description="在 splatfacto 稠密 3DGS 上训练语义头")
    parser.add_argument("--splatfacto-config", type=str, required=True, help="splatfacto 运行目录下的 config.yml")
    parser.add_argument("--processed-dir", type=str, required=True, help="ns_processed 目录（transforms.json）")
    parser.add_argument("--semantic-dir", type=str, required=True, help="语义目录，含 label_maps/*.npy")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录，保存 semantic_3dgs.pt")
    parser.add_argument("--num-classes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not HAS_GSPLAT:
        print("需要安装 gsplat（conda activate worldrecon）")
        sys.exit(1)

    splatfacto_config = Path(args.splatfacto_config)
    if not splatfacto_config.is_absolute():
        splatfacto_config = REPO_ROOT / splatfacto_config
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = REPO_ROOT / processed_dir
    semantic_dir = Path(args.semantic_dir)
    if not semantic_dir.is_absolute():
        semantic_dir = REPO_ROOT / semantic_dir
    label_maps_dir = semantic_dir / "label_maps"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("正在从 splatfacto 加载稠密高斯...")
    means, quats, scales_linear, opacities_linear, rgb = load_splatfacto_gaussians(splatfacto_config, device)
    N = means.shape[0]
    print(f"  高斯数量: {N}")

    model = DenseSemanticGaussianModel(
        means=means,
        quats=quats,
        scales_linear=scales_linear,
        opacities_linear=opacities_linear,
        rgb=rgb,
        num_classes=args.num_classes,
        init_semantic="zeros",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    frames, W, H, fl_x, fl_y, cx, cy = load_cameras(processed_dir)
    K_t = get_K_tensor(fl_x, fl_y, cx, cy, device)

    for step in range(args.steps):
        idx = step % len(frames)
        frame = frames[idx]
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        viewmat = get_viewmat(c2w).to(device)

        rel = frame["file_path"]
        stem = Path(rel).stem
        sem_path_npy = label_maps_dir / f"{stem}.npy"
        if not sem_path_npy.exists():
            continue
        sem_gt = np.load(sem_path_npy).astype(np.int64)
        sem_gt = np.clip(sem_gt, 0, args.num_classes - 1)
        sem_gt_t = torch.from_numpy(sem_gt).long().to(device)

        # 渲染语义（几何与 RGB 固定，不参与 backward）
        sem_feat = model.get_semantic_features()
        out_sem, _, _ = rasterization(
            means=model.means,
            quats=model.quats,
            scales=model.scales,
            opacities=model.opacities,
            colors=sem_feat,
            viewmats=viewmat,
            Ks=K_t,
            width=W,
            height=H,
            sh_degree=None,
        )
        sem_rendered = out_sem[0]

        loss = F.cross_entropy(
            sem_rendered.permute(2, 0, 1).unsqueeze(0),
            sem_gt_t.unsqueeze(0),
            ignore_index=255,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 200 == 0:
            print(f"step {step+1}/{args.steps} loss_sem={loss.item():.4f}")

    ckpt = {
        "means": model.means.detach().cpu(),
        "quats": model.quats.detach().cpu(),
        "scales": model.scales.detach().cpu(),
        "opacities": model.opacities.detach().cpu(),
        "rgb": model.rgb.detach().cpu(),
        "semantic_logits": model.semantic_logits.detach().cpu(),
        "num_classes": args.num_classes,
    }
    torch.save(ckpt, output_dir / "semantic_3dgs.pt")
    print(f"已保存: {output_dir / 'semantic_3dgs.pt'}")


if __name__ == "__main__":
    main()
