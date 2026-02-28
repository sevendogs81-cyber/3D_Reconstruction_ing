"""
语义层：使场景可查询/可理解（世界模型增量，ROI 最大）。

提供：
- 2D 语义：每张图像的分割结果（类别 + 可选 mask）
- 3D 语义：稀疏点云上的类别（由 2D 融合得到）
- 查询 API：按类别、按 3D 区域、按点查询
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]


@dataclass
class ImageSemantics:
    """单张图像的语义标注。"""

    image_path: str  # 相对于场景根或 ns_processed
    image_id: Optional[int] = None  # COLMAP image_id
    height: int = 0
    width: int = 0
    # 每像素类别 id，或存为 numpy 文件路径（相对 semantic_root）
    label_map_path: Optional[str] = None
    # 可选：类别 id -> 该图内像素数 / 占比，用于快速“该图有哪些类”
    class_counts: Dict[int, int] = field(default_factory=dict)
    # 可选：检测框或实例 id 映射
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_id": self.image_id,
            "height": self.height,
            "width": self.width,
            "label_map_path": self.label_map_path,
            "class_counts": self.class_counts,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageSemantics":
        return cls(
            image_path=d.get("image_path", ""),
            image_id=d.get("image_id"),
            height=int(d.get("height", 0)),
            width=int(d.get("width", 0)),
            label_map_path=d.get("label_map_path"),
            class_counts={int(k): int(v) for k, v in (d.get("class_counts") or {}).items()},
            extra=d.get("extra") or {},
        )


@dataclass
class PointSemantics:
    """单个 3D 点的语义（由多视图投票得到）。"""

    point_id: int
    x: float
    y: float
    z: float
    class_id: int
    class_name: Optional[str] = None
    confidence: float = 1.0  # 投票置信度或平均概率
    num_views: int = 0  # 参与投票的视图数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "num_views": self.num_views,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PointSemantics":
        return cls(
            point_id=int(d["point_id"]),
            x=float(d["x"]),
            y=float(d["y"]),
            z=float(d["z"]),
            class_id=int(d["class_id"]),
            class_name=d.get("class_name"),
            confidence=float(d.get("confidence", 1.0)),
            num_views=int(d.get("num_views", 0)),
        )


@dataclass
class SemanticScene:
    """
    场景的语义表示：类别表 + 2D 标注 + 可选 3D 点语义。
    使场景可查询（按类别、区域、点）。
    """

    scene_id: str = ""
    # class_id -> class_name（如 ADE20K 150 类）
    classes: Dict[int, str] = field(default_factory=dict)
    # 图像 key（路径或 image_id 字符串）-> ImageSemantics
    images: Dict[str, ImageSemantics] = field(default_factory=dict)
    # point_id -> PointSemantics（可选，由 2D 融合得到）
    points: Dict[int, PointSemantics] = field(default_factory=dict)
    # 语义数据根目录（用于解析相对路径）
    semantic_root: Optional[Path] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def class_id_to_name(self, class_id: int) -> str:
        return self.classes.get(class_id, f"class_{class_id}")

    def class_name_to_id(self, name: str) -> Optional[int]:
        name_lower = name.strip().lower()
        for cid, cname in self.classes.items():
            if cname.lower() == name_lower:
                return cid
        return None

    def query_by_class(
        self,
        class_name: str,
        *,
        include_2d: bool = True,
        include_3d: bool = True,
    ) -> Dict[str, Any]:
        """
        按类别名查询：返回包含该类别的图像列表与 3D 点列表。
        """
        cid = self.class_name_to_id(class_name)
        if cid is None:
            return {"class_id": None, "class_name": class_name, "images": [], "points": []}

        result: Dict[str, Any] = {
            "class_id": cid,
            "class_name": self.class_id_to_name(cid),
            "images": [],
            "points": [],
        }

        if include_2d:
            for key, im_sem in self.images.items():
                if cid in (im_sem.class_counts or {}):
                    result["images"].append({
                        "image_path": im_sem.image_path,
                        "image_id": im_sem.image_id,
                        "pixel_count": im_sem.class_counts.get(cid, 0),
                    })

        if include_3d and self.points:
            for pid, ps in self.points.items():
                if ps.class_id == cid:
                    result["points"].append({
                        "point_id": pid,
                        "xyz": (ps.x, ps.y, ps.z),
                        "confidence": ps.confidence,
                    })

        return result

    def query_region(
        self,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        *,
        class_filter: Optional[str] = None,
    ) -> List[PointSemantics]:
        """
        查询 3D 包围盒内的语义点；可选按类别过滤。
        """
        x0, y0, z0 = bbox_min
        x1, y1, z1 = bbox_max
        cid = self.class_name_to_id(class_filter) if class_filter else None

        out: List[PointSemantics] = []
        for ps in self.points.values():
            if not (x0 <= ps.x <= x1 and y0 <= ps.y <= y1 and z0 <= ps.z <= z1):
                continue
            if cid is not None and ps.class_id != cid:
                continue
            out.append(ps)
        return out

    def get_semantic_at_point(self, x: float, y: float, z: float) -> Optional[PointSemantics]:
        """
        最近邻：找到离 (x,y,z) 最近的带语义的 3D 点并返回其语义。
        若无 3D 点语义则返回 None。
        """
        if not self.points:
            return None
        best: Optional[PointSemantics] = None
        best_d2 = float("inf")
        for ps in self.points.values():
            d2 = (ps.x - x) ** 2 + (ps.y - y) ** 2 + (ps.z - z) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = ps
        return best

    def list_classes(self) -> List[Tuple[int, str]]:
        """返回 (class_id, class_name) 列表，便于“场景中有哪些类”查询。"""
        return sorted(self.classes.items())

    def to_dict(self) -> Dict[str, Any]:
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        return {
            "scene_id": self.scene_id,
            "classes": self.classes,
            "images": {k: v.to_dict() for k, v in self.images.items()},
            "points": {str(k): v.to_dict() for k, v in self.points.items()},
            "semantic_root": str(self.semantic_root) if self.semantic_root else None,
            "meta": _convert(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticScene":
        classes = {int(k): str(v) for k, v in (data.get("classes") or {}).items()}
        images = {
            k: ImageSemantics.from_dict(v)
            for k, v in (data.get("images") or {}).items()
        }
        points = {}
        for k, v in (data.get("points") or {}).items():
            points[int(k)] = PointSemantics.from_dict(v)
        root = data.get("semantic_root")
        return cls(
            scene_id=data.get("scene_id", ""),
            classes=classes,
            images=images,
            points=points,
            semantic_root=Path(root) if root else None,
            meta=data.get("meta") or {},
        )


def load_semantic_scene(path: PathLike) -> SemanticScene:
    """从 JSON 文件加载语义场景。"""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scene = SemanticScene.from_dict(data)
    if not scene.semantic_root:
        scene.semantic_root = p.parent
    return scene


def save_semantic_scene(scene: SemanticScene, path: PathLike) -> None:
    """将语义场景保存为 JSON。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(scene.to_dict(), f, ensure_ascii=False, indent=2)
