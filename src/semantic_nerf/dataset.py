"""加载 ns_processed + 语义 label_maps，返回图像、相机与每像素语义标签。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def load_transforms(processed_dir: Path) -> Tuple[Dict, List[Dict], float, float, float, float]:
    with open(processed_dir / "transforms.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta["frames"]
    w = int(meta["w"])
    h = int(meta["h"])
    fl_x = float(meta["fl_x"])
    fl_y = float(meta["fl_y"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    return meta, frames, fl_x, fl_y, cx, cy


def load_semantic_maps(
    processed_dir: Path,
    semantic_root: Path,
    frames: List[Dict],
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    """frames 中 file_path -> (H,W) long tensor 语义标签。缺失的用 -1 或 0。"""
    out: Dict[str, torch.Tensor] = {}
    for f in frames:
        rel = f.get("file_path", "")
        if not rel:
            continue
        stem = Path(rel).stem
        npy_path = semantic_root / "label_maps" / f"{stem}.npy"
        if npy_path.exists():
            lab = np.load(npy_path).astype(np.int64)
            # 过滤非法类别
            lab = np.clip(lab, 0, num_classes - 1)
            out[rel] = torch.from_numpy(lab)
        else:
            # 无标签时用 0（可后续 mask 掉）
            out[rel] = None  # 调用方用 mask 跳过
    return out


class SemanticNeRFDataset(torch.utils.data.Dataset):
    """单场景：transforms.json + images + semantic label_maps（.npy）。"""

    def __init__(
        self,
        processed_dir: Path,
        semantic_root: Optional[Path] = None,
        num_classes: int = 150,
        split: str = "train",
        train_frac: float = 0.9,
    ):
        self.processed_dir = Path(processed_dir)
        self.semantic_root = Path(semantic_root) if semantic_root else self.processed_dir.parent / "semantic"
        self.num_classes = num_classes
        meta, frames, fl_x, fl_y, cx, cy = load_transforms(self.processed_dir)
        self.meta = meta
        self.frames = frames
        self.w = int(meta["w"])
        self.h = int(meta["h"])
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy
        n = len(frames)
        idx = np.random.permutation(n) if split == "train" else np.arange(n)
        if split == "train":
            idx = idx[: int(n * train_frac)]
        else:
            idx = idx[int(n * train_frac) :]
        self.indices = idx.tolist()
        self.semantic_maps = load_semantic_maps(
            self.processed_dir, self.semantic_root, self.frames, self.num_classes
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _get_camera(self, frame: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """c2w 4x4（相机到世界）与 K 3x3。"""
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        K = np.array([
            [self.fl_x, 0, self.cx],
            [0, self.fl_y, self.cy],
            [0, 0, 1],
        ], dtype=np.float32)
        return torch.from_numpy(c2w), torch.from_numpy(K)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = self.indices[i]
        frame = self.frames[idx]
        rel = frame["file_path"]
        img_path = self.processed_dir / rel
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image)
        c2w, K = self._get_camera(frame)
        semantics = self.semantic_maps.get(rel)
        if semantics is None:
            semantics = torch.zeros(self.h, self.w, dtype=torch.long)
        return {
            "image": image,
            "c2w": c2w,
            "K": K,
            "semantics": semantics,
            "height": self.h,
            "width": self.w,
        }
