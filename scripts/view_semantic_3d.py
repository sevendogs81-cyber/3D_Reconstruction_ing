#!/usr/bin/env python3
"""
在 3D Viewer 中按语义类别着色显示点云。

使用 Open3D 打开带语义的稀疏点云，每个类别一种颜色；支持按键高亮单类或显示全部。

用法：
  # 使用语义场景 JSON（需已运行 3D 融合，即 semantic_scene.json 内含 points）
  python scripts/view_semantic_3d.py --semantic-json mipnerf360/db/playroom/semantic/semantic_scene.json

  # 或通过 world_state 指定场景
  python scripts/view_semantic_3d.py --world-state mipnerf360/db/playroom/world_state.playroom.json

  # 无显示器（SSH/服务器）时导出彩色 PLY，在本地用 MeshLab/CloudCompare 或 Open3D 打开
  python scripts/view_semantic_3d.py --semantic-json ... --export-ply semantic_colored.ply

依赖：open3d（worldrecon 环境已包含）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.world_model.semantics import load_semantic_scene, SemanticScene
from src.world_model.scene_state import load_scene_state


def get_class_colormap(num_classes: int) -> np.ndarray:
    """为 0..num_classes-1 生成区分度高的 RGB 颜色 (num_classes, 3)，值域 [0,1]。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab20")
        # 若类别多则用 turbo 再采样
        if num_classes <= 20:
            colors = np.array([cmap(i % 20)[:3] for i in range(num_classes)], dtype=np.float64)
        else:
            cmap = plt.get_cmap("turbo")
            colors = np.array([cmap(i / max(num_classes - 1, 1))[:3] for i in range(num_classes)], dtype=np.float64)
        return colors
    except Exception:
        # 无 matplotlib 时用简单离散色
        palette = [
            (1, 0, 0), (0, 0.8, 0), (0, 0, 1), (1, 0.8, 0), (1, 0, 0.8),
            (0, 0.8, 0.8), (0.6, 0.2, 0.8), (0.8, 0.6, 0), (0.2, 0.6, 0.4), (0.8, 0.2, 0.2),
        ]
        return np.array([palette[i % len(palette)] for i in range(num_classes)], dtype=np.float64)


def build_point_cloud_from_semantic_scene(scene: SemanticScene) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]], list]:
    """
    从 SemanticScene 的 3D 点构建点云坐标与颜色。
    返回: points (N,3), colors (N,3), legend [(class_id, class_name), ...], point_semantics_list (与 points 同序)
    """
    if not scene.points:
        return np.zeros((0, 3)), np.zeros((0, 3)), [], []

    # 收集所有出现过的 class_id，保持顺序
    class_ids_seen: list[int] = []
    for ps in scene.points.values():
        if ps.class_id not in class_ids_seen:
            class_ids_seen.append(ps.class_id)
    num_classes = len(class_ids_seen)
    cid_to_idx = {cid: i for i, cid in enumerate(class_ids_seen)}
    colormap = get_class_colormap(num_classes)

    points_list: list[list[float]] = []
    colors_list: list[list[float]] = []
    point_semantics_list: list = []
    for ps in scene.points.values():
        points_list.append([ps.x, ps.y, ps.z])
        idx = cid_to_idx[ps.class_id]
        colors_list.append(colormap[idx].tolist())
        point_semantics_list.append(ps)

    points = np.array(points_list, dtype=np.float64)
    colors = np.array(colors_list, dtype=np.float64)
    legend = [(cid, scene.class_id_to_name(cid)) for cid in class_ids_seen]
    return points, colors, legend, point_semantics_list


def export_colored_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """导出按类别着色的点云为 PLY，便于在无显示器环境下用 MeshLab/CloudCompare 等查看。"""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def run_open3d_viewer(
    points: np.ndarray,
    colors: np.ndarray,
    legend: list[tuple[int, str]],
    point_semantics_list: list,
) -> None:
    """用 Open3D 打开可交互窗口（draw_geometries，兼容无 register_key_callback 的构建）；无显示器时请用 --export-ply。"""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("--- 类别图例 ---")
    for cid, cname in legend:
        print(f"  {cid}: {cname}")
    print("--- 无显示器时请使用: --export-ply out.ply 导出后在本地用 MeshLab/CloudCompare 打开 ---\n")

    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="语义 3D 查看器 — 按类别着色",
            point_show_normal=False,
        )
    except Exception as e:
        print(f"无法打开图形窗口: {e}")
        print("请使用: --export-ply semantic_colored.ply 导出后下载到本地查看。")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D 语义查看器：按类别着色点云")
    parser.add_argument("--semantic-json", type=str, help="semantic_scene.json 路径")
    parser.add_argument("--world-state", type=str, help="或 world_state.*.json 路径（从中读取语义）")
    parser.add_argument("--export-ply", type=str, metavar="PATH", help="不打开窗口，导出彩色 PLY 到指定路径（适用于无显示器/SSH）")
    args = parser.parse_args()

    scene: SemanticScene | None = None
    if args.world_state:
        path = Path(args.world_state)
        if not path.is_absolute():
            path = REPO_ROOT / path
        state = load_scene_state(path)
        scene = state.load_semantic_scene()
        if scene is None:
            sem_path = state.get_semantic_path()
            if sem_path and sem_path.exists():
                scene = load_semantic_scene(sem_path)
            if scene is None:
                print("未找到语义数据：请先运行 run_semantic_labeling.py 并指定 --sparse-dir 做 3D 融合。")
                sys.exit(1)
    elif args.semantic_json:
        path = Path(args.semantic_json)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            print(f"文件不存在: {path}")
            sys.exit(1)
        scene = load_semantic_scene(path)
    else:
        parser.error("请指定 --semantic-json 或 --world-state")
        return

    if not scene.points:
        print("当前语义场景中没有 3D 点（未做 3D 融合）。")
        print("请先运行: python scripts/run_semantic_labeling.py --processed-dir ... --sparse-dir ... --output-dir ...")
        sys.exit(1)

    points, colors, legend, point_semantics_list = build_point_cloud_from_semantic_scene(scene)
    print(f"共 {len(points)} 个带语义的 3D 点，{len(legend)} 个类别。")

    if args.export_ply:
        out_path = Path(args.export_ply)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_colored_ply(points, colors, out_path)
        print(f"已导出彩色点云: {out_path}")
        return

    run_open3d_viewer(points, colors, legend, point_semantics_list)


if __name__ == "__main__":
    main()
