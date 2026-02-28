#!/usr/bin/env python3
"""
对场景图像运行 2D 语义分割，并可选地将语义融合到 COLMAP 稀疏点上，
使场景可查询（按类别、区域、点）。

用法示例：
  # 仅 2D 语义（需提供 ns_processed 目录，内含 transforms.json 和 images/）
  python scripts/run_semantic_labeling.py \\
    --processed-dir mipnerf360/db/playroom/ns_processed \\
    --output-dir mipnerf360/db/playroom/semantic \\
    --scene-id mipnerf360/db/playroom

  # 2D + 3D 融合（需额外提供 COLMAP sparse 目录）
  python scripts/run_semantic_labeling.py \\
    --processed-dir mipnerf360/db/playroom/ns_processed \\
    --sparse-dir mipnerf360/db/playroom/sparse/0 \\
    --output-dir mipnerf360/db/playroom/semantic \\
    --scene-id mipnerf360/db/playroom

依赖：worldrecon 环境中的 transformers, torch, PIL。模型默认使用 SegFormer (ADE20K)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from src.world_model.semantics import (
    SemanticScene,
    ImageSemantics,
    PointSemantics,
    save_semantic_scene,
)


def get_segformer_processor_and_model(model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
    """加载 SegFormer 的 processor 与 model，并得到 id2label。"""
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    raw = getattr(model.config, "id2label", None) or {}
    # HuggingFace 通常为 "0": "wall", "1": "building", ...
    id2label = {int(k): str(v) for k, v in raw.items()}
    return processor, model, id2label


def run_2d_semantic(
    processed_dir: Path,
    output_dir: Path,
    scene_id: str,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    device: str = "cuda",
    batch_size: int = 1,
) -> Tuple[SemanticScene, Dict[int, str]]:
    """
    对 processed_dir 下 transforms.json 中的每张图跑语义分割，
    写出 label_maps 到 output_dir/label_maps/，并构建 SemanticScene（2D 部分）。
    """
    transforms_path = processed_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"需要 {transforms_path}")

    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    if not frames:
        raise ValueError("transforms.json 中 frames 为空")

    processor, model, id2label = get_segformer_processor_and_model(model_name)
    model = model.to(device)
    model.eval()

    # 统一类别表（SegFormer ADE20K 含 0 为 wall 等，保留原始 id）
    classes = dict(id2label) if id2label else {}
    if not classes:
        classes = {i: f"class_{i}" for i in range(151)}

    label_maps_dir = output_dir / "label_maps"
    label_maps_dir.mkdir(parents=True, exist_ok=True)

    scene = SemanticScene(scene_id=scene_id, classes=classes, semantic_root=output_dir)
    image_key_to_colmap_id: Dict[str, int] = {}

    import torch
    for i, frame in enumerate(frames):
        rel_path = frame.get("file_path", "")
        if not rel_path:
            continue
        img_path = processed_dir / rel_path
        if not img_path.exists():
            print(f"跳过不存在的图像: {img_path}")
            continue
        colmap_im_id = frame.get("colmap_im_id")
        if colmap_im_id is not None:
            image_key_to_colmap_id[rel_path] = int(colmap_im_id)

        pil = Image.open(img_path).convert("RGB")
        w, h = pil.size
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits  # (1, C, H, W)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        # 上采样到原图尺寸
        pred_pil = Image.fromarray(pred.astype(np.uint16))
        pred_resized = pred_pil.resize((w, h), Image.NEAREST)
        pred_final = np.array(pred_resized)

        # 保存 label map（npy 或 png 存 id）
        stem = Path(rel_path).stem
        label_filename = f"{stem}.npy"
        label_path = label_maps_dir / label_filename
        np.save(label_path, pred_final)

        # class_counts
        unique, counts = np.unique(pred_final, return_counts=True)
        class_counts = {int(u): int(c) for u, c in zip(unique, counts)}

        rel_label = f"label_maps/{label_filename}"
        im_sem = ImageSemantics(
            image_path=rel_path,
            image_id=image_key_to_colmap_id.get(rel_path),
            height=h,
            width=w,
            label_map_path=rel_label,
            class_counts=class_counts,
        )
        scene.images[rel_path] = im_sem

        if (i + 1) % 10 == 0:
            print(f"  2D 语义已处理 {i + 1}/{len(frames)} 张")

    scene.meta["model"] = model_name
    scene.meta["source"] = "run_semantic_labeling_2d"
    return scene, image_key_to_colmap_id


def parse_colmap_images(images_txt: Path) -> Dict[int, str]:
    """COLMAP images.txt: 每两行一组，第一行 IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME。"""
    name_by_id: Dict[int, str] = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                im_id = int(parts[0])
                name = parts[9]
                name_by_id[im_id] = name
            next(f)  # 跳过第二行（2D points）
    return name_by_id


def parse_colmap_points(points_txt: Path) -> Dict[int, Tuple[float, float, float, List[Tuple[int, int]]]]:
    """
    COLMAP points3D.txt: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) ...
    返回 point_id -> (x, y, z, [(image_id, point2d_idx), ...])
    """
    points: Dict[int, Tuple[float, float, float, List[Tuple[int, int]]]] = {}
    with open(points_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            track = []
            for i in range(8, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    track.append((int(parts[i]), int(parts[i + 1])))
            points[point_id] = (x, y, z, track)
    return points


def get_camera_intrinsics(transforms: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """从 transforms.json 取 fl_x, fl_y, cx, cy。"""
    fl_x = float(transforms.get("fl_x", 1.0))
    fl_y = float(transforms.get("fl_y", 1.0))
    cx = float(transforms.get("cx", 0.0))
    cy = float(transforms.get("cy", 0.0))
    return fl_x, fl_y, cx, cy


def get_image_name_to_file_path(frames: List[Dict]) -> Dict[str, str]:
    """从 frames 建立 COLMAP 图像 NAME 到 transforms 中 file_path 的映射。"""
    # COLMAP NAME 通常是文件名，如 frame_00008.jpg；transforms 里是 images/frame_00008.jpg
    name_to_path: Dict[str, str] = {}
    for f in frames:
        path = f.get("file_path", "")
        if path:
            name = Path(path).name
            name_to_path[name] = path
    return name_to_path


def fuse_3d_semantics(
    scene: SemanticScene,
    processed_dir: Path,
    sparse_dir: Path,
    image_key_to_colmap_id: Dict[str, int],
) -> None:
    """
    用 COLMAP 稀疏点与 images.txt、points3D.txt，将 2D 语义投票到 3D 点。
    """
    images_txt = sparse_dir / "images.txt"
    points_txt = sparse_dir / "points3D.txt"
    if not images_txt.exists() or not points_txt.exists():
        print("未找到 COLMAP sparse images.txt 或 points3D.txt，跳过 3D 融合")
        return

    with open(processed_dir / "transforms.json", "r", encoding="utf-8") as f:
        transforms = json.load(f)
    frames = transforms.get("frames", [])
    fl_x, fl_y, cx, cy = get_camera_intrinsics(transforms)
    name_by_id = parse_colmap_images(images_txt)
    colmap_id_to_file_path: Dict[int, str] = {}
    name_to_path = get_image_name_to_file_path(frames)
    for im_id, name in name_by_id.items():
        if name in name_to_path:
            colmap_id_to_file_path[im_id] = name_to_path[name]

    # 构建 (colmap_im_id, file_path) -> 该图的 label map numpy
    def load_label_map(rel_path: str) -> Optional[np.ndarray]:
        # scene.images 的 key 是 file_path (e.g. images/frame_00008.jpg)
        if rel_path not in scene.images:
            return None
        im_sem = scene.images[rel_path]
        if not im_sem.label_map_path or not scene.semantic_root:
            return None
        path = scene.semantic_root / im_sem.label_map_path
        if not path.exists():
            return None
        return np.load(path)

    points_data = parse_colmap_points(points_txt)
    # 相机位姿：从 transforms 的 transform_matrix 取 R, t；需与 colmap_im_id 对应
    frame_by_path: Dict[str, Dict] = {f["file_path"]: f for f in frames}
    path_to_colmap_id = {v: k for k, v in colmap_id_to_file_path.items()}

    def world_to_camera_and_project(x: float, y: float, z: float, frame: Dict) -> Optional[Tuple[int, int]]:
        """世界坐标转相机平面，再投影到像素。返回 (px, py) 或 None。"""
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        R, t = mat[:3, :3], mat[:3, 3]
        p_w = np.array([x, y, z, 1.0])
        p_cam = (R @ p_w[:3]) + t
        if p_cam[2] <= 0:
            return None
        u = fl_x * p_cam[0] / p_cam[2] + cx
        v = fl_y * p_cam[1] / p_cam[2] + cy
        return int(round(u)), int(round(v))

    voted: Dict[int, List[int]] = {}  # point_id -> [class_id from each view]
    for point_id, (x, y, z, track) in points_data.items():
        class_ids: List[int] = []
        for im_id, point2d_idx in track:
            path = colmap_id_to_file_path.get(im_id)
            if not path:
                continue
            frame = frame_by_path.get(path)
            if not frame:
                continue
            label_map = load_label_map(path)
            if label_map is None:
                continue
            h, w = label_map.shape
            # point2d_idx 是 COLMAP 中该图像上的 2D 点索引；我们没有直接 2D 坐标，
            # 需要用世界点反投影到该图得到 (u,v)
            uv = world_to_camera_and_project(x, y, z, frame)
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < w and 0 <= v < h:
                cid = int(label_map[v, u])
                class_ids.append(cid)
        if not class_ids:
            continue
        # 多数投票
        from collections import Counter
        (most_id, count) = Counter(class_ids).most_common(1)[0]
        confidence = count / len(class_ids)
        ps = PointSemantics(
            point_id=point_id,
            x=x, y=y, z=z,
            class_id=most_id,
            class_name=scene.class_id_to_name(most_id),
            confidence=round(confidence, 4),
            num_views=len(class_ids),
        )
        scene.points[point_id] = ps

    print(f"  3D 融合得到 {len(scene.points)} 个带语义的稀疏点")


def main():
    parser = argparse.ArgumentParser(description="场景 2D/3D 语义标注，使场景可查询")
    parser.add_argument("--processed-dir", type=str, required=True,
                        help="ns_processed 目录（含 transforms.json 和 images/）")
    parser.add_argument("--sparse-dir", type=str, default=None,
                        help="COLMAP sparse 目录（含 images.txt, points3D.txt），可选，用于 3D 融合")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="语义输出目录（将写入 semantic_scene.json 与 label_maps/）")
    parser.add_argument("--scene-id", type=str, default="",
                        help="场景 ID，写入 semantic_scene.json")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512",
                        help="HuggingFace 语义分割模型")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-3d", action="store_true", help="不做 3D 融合，仅 2D")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = (REPO_ROOT / processed_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_id = args.scene_id or str(processed_dir)
    print("运行 2D 语义分割...")
    scene, image_key_to_colmap_id = run_2d_semantic(
        processed_dir=processed_dir,
        output_dir=output_dir,
        scene_id=scene_id,
        model_name=args.model,
        device=args.device,
    )

    if not args.no_3d and args.sparse_dir:
        sparse_dir = Path(args.sparse_dir)
        if not sparse_dir.is_absolute():
            sparse_dir = (REPO_ROOT / sparse_dir).resolve()
        print("运行 3D 语义融合...")
        fuse_3d_semantics(scene, processed_dir, sparse_dir, image_key_to_colmap_id)
    else:
        print("跳过 3D 融合（未指定 --sparse-dir 或使用了 --no-3d）")

    out_json = output_dir / "semantic_scene.json"
    save_semantic_scene(scene, out_json)
    print(f"已保存: {out_json}")
    print("可通过 SceneState.load_semantic_scene() 与 query_by_class / query_region 进行查询。")


if __name__ == "__main__":
    main()
