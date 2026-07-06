#!/usr/bin/env python3
"""从 .tex 编译 PDF，并生成配套 Markdown（pandoc）。"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent

DOC_SPECS = [
    {
        "tex": "review_world_model_theory.tex",
        "md": "review_world_model_theory.md",
        "pdf": "review_world_model_theory.pdf",
        "title": "从三维重建到世界模型的技术路径综述",
        "subtitle": "几何、辐射场与显式表示",
        "abstract": (
            "本综述基于 `3D_Reconstruction_ing` 仓库中的文档与代码，系统梳理从传统三维重建到面向世界模型"
            "（World Model）的 3D/360 表示技术路径。文章从几何层、辐射场层、显式高效表示层，以及语义与世界"
            "记忆接口四个层面展开，给出关键数学公式与算法要点，为后续工程实践文档与实验提供理论背景。"
        ),
    },
    {
        "tex": "review_world_model_engineering.tex",
        "md": "review_world_model_engineering.md",
        "pdf": "review_world_model_engineering.pdf",
        "title": "从 Mip-NeRF 360 到世界模型的工程实践指南",
        "subtitle": "基于 `3D_Reconstruction_ing` 仓库",
        "abstract": (
            "本文面向工程实践，基于 `3D_Reconstruction_ing` 仓库，给出从 Mip-NeRF 360 数据集出发，经 COLMAP "
            "几何重建、Nerfstudio NeRF 训练、3D Gaussian Splatting（3DGS）以及语义标注与世界模型场景状态构建的"
            "完整 Pipeline。文中所有命令与代码示例均可在仓库结构与环境配置下直接运行或微调使用，与配套的理论"
            "综述文档构成“原理 + 实践”完整一套笔记。"
        ),
    },
]


def find_xelatex() -> str | None:
    for candidate in (
        shutil.which("xelatex"),
        Path.home() / ".TinyTeX/bin/x86_64-linux/xelatex",
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def compile_pdf(tex_path: Path, xelatex: str) -> None:
    for _ in range(2):
        result = subprocess.run(
            [xelatex, "-interaction=nonstopmode", tex_path.name],
            cwd=DOCS,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and not tex_path.with_suffix(".pdf").exists():
            print(result.stdout[-2000:], file=sys.stderr)
            print(result.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"xelatex failed for {tex_path.name}")


def postprocess_markdown(body: str) -> str:
    body = re.sub(r"``` \{[^`]*\}", "```", body)
    body = re.sub(r"``` (\w+)", r"```\1", body)
    body = body.replace("文献综述_3D_与_全景重建.md", "review_world_model_theory.md")
    body = body.replace("COLMAP_与_Nerfstudio_教程.md", "review_world_model_engineering.md")
    body = re.sub(r"^1\.  \\\n", "1.  \n", body, flags=re.MULTILINE)
    return body.strip() + "\n"


def build_markdown(spec: dict, tex_path: Path, md_path: Path) -> None:
    import pypandoc

    body = pypandoc.convert_file(str(tex_path), "gfm", format="latex", extra_args=["--wrap=none"])
    body = postprocess_markdown(body)

    header = (
        f"# {spec['title']}\n\n"
        f"## {spec['subtitle']}\n\n"
        f"> 3D_Reconstruction_ing 项目笔记\n\n"
        f"**摘要：** {spec['abstract']}\n\n"
        f"---\n\n"
    )
    md_path.write_text(header + body, encoding="utf-8")
    print(f"Wrote {md_path.name}")


def main() -> int:
    xelatex = find_xelatex()
    if xelatex is None:
        print("Warning: xelatex not found; skipping PDF compilation.", file=sys.stderr)

    for spec in DOC_SPECS:
        tex_path = DOCS / spec["tex"]
        md_path = DOCS / spec["md"]
        pdf_path = DOCS / spec["pdf"]

        if xelatex:
            print(f"Compiling {tex_path.name} -> {pdf_path.name}")
            compile_pdf(tex_path, xelatex)

        build_markdown(spec, tex_path, md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
