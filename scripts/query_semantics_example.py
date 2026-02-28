#!/usr/bin/env python3
"""
语义层查询示例：从 world_state 加载场景与语义，按类别/区域/点查询。

用法（需先对场景运行 run_semantic_labeling.py 生成 semantic_scene.json）：
  python scripts/query_semantics_example.py --world-state mipnerf360/db/playroom/world_state.playroom.json
  或直接指定语义 JSON：
  python scripts/query_semantics_example.py --semantic-json mipnerf360/db/playroom/semantic/semantic_scene.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.world_model import load_scene_state
from src.world_model.semantics import load_semantic_scene, SemanticScene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-state", type=str, help="world_state.*.json 路径")
    parser.add_argument("--semantic-json", type=str, help="或直接指定 semantic_scene.json 路径")
    parser.add_argument("--class", dest="class_name", type=str, default="chair",
                        help="要查询的类别名（默认 chair）")
    args = parser.parse_args()

    sem: SemanticScene | None = None
    if args.world_state:
        path = Path(args.world_state)
        if not path.is_absolute():
            path = REPO_ROOT / path
        state = load_scene_state(path)
        sem = state.load_semantic_scene()
        if sem is None:
            print("该 world_state 未配置语义或 semantic_scene.json 尚不存在。")
            print("请先运行: python scripts/run_semantic_labeling.py --processed-dir ... --output-dir ...")
            return
    elif args.semantic_json:
        path = Path(args.semantic_json)
        if not path.is_absolute():
            path = REPO_ROOT / path
        sem = load_semantic_scene(path)
    else:
        parser.error("请指定 --world-state 或 --semantic-json")
        return

    print("=== 场景中出现的类别（前 20 个）===")
    for cid, cname in sem.list_classes()[:20]:
        print(f"  {cid}: {cname}")
    if len(sem.classes) > 20:
        print(f"  ... 共 {len(sem.classes)} 类")

    print(f"\n=== 按类别查询: {args.class_name} ===")
    result = sem.query_by_class(args.class_name, include_2d=True, include_3d=True)
    if result["class_id"] is None:
        print(f"  未找到类别 '{args.class_name}'，请用上面列表中的英文类名重试。")
    else:
        print(f"  包含该类别的图像数: {len(result['images'])}")
        print(f"  包含该类别的 3D 点数: {len(result['points'])}")
        if result["images"]:
            print("  示例图像:", result["images"][0].get("image_path"))

    print("\n完成。")


if __name__ == "__main__":
    main()
