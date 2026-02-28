from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .semantics import SemanticScene, load_semantic_scene

PathLike = Union[str, Path]


@dataclass
class SceneState:
    """轻量级的静态场景状态表示（world model scene state）。

    约定：
    - 所有路径字段在内存中用 Path 存储；
    - 序列化到 JSON 时统一转为相对于仓库根目录（或某个自定义 root）的字符串。
    """

    scene_id: str
    root: Path
    coordinate_system: str = "colmap"
    representations: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, relative: PathLike) -> Path:
        """将相对路径解析到场景根目录下的绝对路径。"""

        rel = Path(relative)
        if rel.is_absolute():
            return rel
        return (self.root / rel).resolve()

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典（内部 Path → str）。"""

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
            "root": str(self.root),
            "coordinate_system": self.coordinate_system,
            "representations": _convert(self.representations),
            "meta": _convert(self.meta),
        }

    def get_semantic_path(self) -> Optional[Path]:
        """返回语义层 JSON 的绝对路径（若已配置）。"""
        sem = self.representations.get("semantics") or {}
        rel = sem.get("semantic_scene_json")
        if not rel:
            return None
        return self.resolve(rel)

    def load_semantic_scene(self) -> Optional[SemanticScene]:
        """若场景已配置语义层，则加载并返回 SemanticScene，否则返回 None。"""
        path = self.get_semantic_path()
        if path is None or not path.exists():
            return None
        scene = load_semantic_scene(path)
        if not scene.semantic_root:
            scene.semantic_root = path.parent
        return scene

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneState":
        """从 JSON 友好的字典构建 SceneState。"""

        scene_id = data.get("scene_id", "")
        root = Path(data.get("root", "."))
        coordinate_system = data.get("coordinate_system", "colmap")
        representations = data.get("representations", {}) or {}
        meta = data.get("meta", {}) or {}

        return cls(
            scene_id=scene_id,
            root=root,
            coordinate_system=coordinate_system,
            representations=representations,
            meta=meta,
        )


def load_scene_state(path: PathLike) -> SceneState:
    """从 JSON 文件加载静态场景状态。"""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    state = SceneState.from_dict(data)

    # 若 JSON 中未显式设置 root，则默认用 JSON 文件所在目录作为场景根
    if "root" not in data or not data.get("root"):
        state.root = json_path.parent
    return state


def save_scene_state(state: SceneState, path: PathLike) -> None:
    """将静态场景状态保存为 JSON 文件。"""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


def example_playroom_state(repo_root: PathLike) -> SceneState:
    """基于当前仓库的 playroom 目录构造一个示例场景状态。

    注意：这里只是示例，某些 3DGS 路径可能尚未训练完成，仍然处于占位状态。
    """

    repo_root = Path(repo_root)
    scene_root = repo_root / "mipnerf360" / "db" / "playroom"

    representations: Dict[str, Any] = {
        "colmap": {
            "sparse_root": "mipnerf360/db/playroom/sparse/0",
            # 若后续生成 playroom 的 dense 结果，可在 world_state JSON 中补充该字段
            "dense_root": None,
        },
        "nerfstudio": {
            "processed_data": "mipnerf360/db/playroom/ns_processed",
            "runs": {
                "nerfacto": {
                    "config": (
                        "mipnerf360/db/playroom/ns_runs/"
                        "playroom_nerfacto/ns_processed/nerfacto/"
                        "2026-02-27_121137/config.yml"
                    )
                }
            },
        },
        "semantics": {
            # 运行 scripts/run_semantic_labeling.py 后生成（路径相对于场景根）
            "semantic_scene_json": "semantic/semantic_scene.json",
        },
        "gaussians": {
            # splatfacto（Nerfstudio 内置 3DGS 实现）
            "splatfacto": {
                "run_dir": None,
                "status": "not_trained",
            },
            # 原生 3DGS（如官方仓库），预留 checkpoint 字段
            "native_3dgs": {
                "checkpoint": None,
                "status": "not_trained",
            },
        },
    }

    meta: Dict[str, Any] = {
        "description": "Static world model scene state for mipnerf360/db/playroom.",
        "version": 1,
    }

    return SceneState(
        scene_id="mipnerf360/db/playroom",
        root=scene_root,
        coordinate_system="colmap",
        representations=representations,
        meta=meta,
    )

