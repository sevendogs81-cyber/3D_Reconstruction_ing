# 三维与全景重建（3D / 360 Reconstruction）世界模型项目

本仓库 `3D_Reconstruction_ing` 的目标是：  
**从多视角图像 / 视频（尤其是 Mip-NeRF 360 等数据）出发，搭建一条通向“世界模型”的 3D / 360 重建技术路径。**

核心能力包括：

- 基于 **COLMAP** 的传统 SfM / MVS 几何重建（相机位姿 + 稀疏/稠密点云）；
- 基于 **Nerfstudio** 的 NeRF 训练与渲染；
- 后续扩展到 **3DGS / 4DGS 等显式高效表示**，支撑虚实融合与世界模型实验。

---

## 仓库结构概览

```text
3D_Reconstruction_ing/
├── README.md                     # 本说明文档
├── requirements.txt              # 最小 Python 依赖（pycolmap 等）
├── env/
│   └── worldrecon.yml            # Nerfstudio + gsplat 等完整环境（conda）
├── docs/
│   ├── review_world_model_theory.tex/.md   # 理论综述（几何 / NeRF / 3DGS / 语义）
│   ├── review_world_model_engineering.tex/.md  # 工程实践（COLMAP → Nerfstudio → 语义）
│   └── build_docs.py                     # 从 .tex 编译 PDF 并导出 Markdown
├── mipnerf360/                   # 示例数据（如 db/drjohnson、db/playroom 等）
├── scripts/                      # 语义标注、训练与渲染脚本
│   ├── run_semantic_labeling.py  # 2D 分割 + 可选 3D 点语义融合
│   ├── train_semantic_nerf.py    # 语义 NeRF 训练
│   ├── train_semantic_3dgs.py    # 在 splatfacto 上挂语义头
│   ├── render_semantic_nerf.py   # 语义 NeRF 渲染
│   ├── render_semantic_3dgs.py   # 语义 3DGS 渲染
│   ├── view_semantic_3d.py       # 3D 语义点云查看
│   └── query_semantics_example.py
└── src/
    ├── world_model/              # 场景状态、语义场景（SceneState, SemanticScene）
    ├── semantic_nerf/            # 语义 NeRF 模型与数据集
    └── semantic_3dgs/            # 稠密 3DGS + 语义（splatfacto 加载与 view 约定）
```

当前已实现重点：

- **文档**：`docs/review_world_model_engineering.md` 覆盖 COLMAP 稀疏/稠密、Nerfstudio（nerfacto / splatfacto）、语义层（2D 标注、3D 融合、语义 NeRF/3DGS 训练与渲染）；`docs/review_world_model_theory.md` 梳理技术路径与数学原理。源文件为同名 `.tex`，可用 `python docs/build_docs.py` 重新编译 PDF 并导出 Markdown。
- **语义 pipeline**：`run_semantic_labeling.py` → `label_maps/` + 可选 `semantic_scene.json`；语义 NeRF（`train_semantic_nerf.py` / `render_semantic_nerf.py`）；语义 3DGS（先 `ns-train splatfacto`，再 `train_semantic_3dgs.py` / `render_semantic_3dgs.py`），详见教程 §9.6。

---

## 环境与依赖

### 1. 几何实验（只用 COLMAP / pycolmap）

如果你只想学习传统 SfM / MVS 和 `pycolmap` 的基本用法：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows 使用 .venv\\Scripts\\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 目前包含：

- `pycolmap==3.13.0`
- `numpy`
- `scipy`
- `opencv-python`

你可以在此基础上编写自己的 **COLMAP / pycolmap 实验脚本**（后续会在 `src/` 下补充示例）。

### 2. 完整世界重建环境（Nerfstudio + 3DGS 等）

若你希望沿着文档走完 **Mip-NeRF 360 → COLMAP → Nerfstudio** 甚至 3DGS 的整条链路，建议使用项目提供的 `worldrecon` 环境：

```bash
conda env create -f env/worldrecon.yml
conda activate worldrecon
```

`env/worldrecon.yml` 中包含：

- 基础：`python=3.10`, `ffmpeg`, `cmake`, `ninja`, `git`；
- 深度学习与重建栈（通过 pip）：
  - `nerfstudio`, `gsplat`, `opencv-python`, `open3d`, `trimesh`；
  - `numpy`, `scipy`, `matplotlib`, `tqdm`, `rich`, `imageio`, `imageio-ffmpeg`, `scikit-image`, `pandas`；
  - 语义与世界模型相关的可选库：`transformers`, `timm`, `einops`, `accelerate`, `supervision` 等。

> 注意：`worldrecon` 环境假定你本机已有合适版本的 CUDA / NVIDIA 驱动。  
> 若安装 GPU 相关包时遇到冲突，可参考 Nerfstudio 与 PyTorch 官方安装指南调整。

---

## 从 COLMAP 到 Nerfstudio 的推荐实践路径

详细步骤请阅读 `docs/review_world_model_engineering.md`，这里给一个**高度概括**，方便你建立整体印象：

1. **准备数据（以 Mip-NeRF 360 为例）**
   - 将某个场景（如 `db/drjohnson`）放到 `mipnerf360/` 目录下；
   - 确保有 `images/` 子目录存放原始多视角图像。

2. **理解 & 练习 COLMAP 几何重建**
   - 按教程第 6–8 节，在命令行跑完：
     - `feature_extractor`、`exhaustive_matcher`、`mapper`；
     - （可选）`image_undistorter`、`patch_match_stereo`、`stereo_fusion`；
   - 学会查看 `cameras.txt` / `images.txt` / `points3D.txt`。

3. **用 Nerfstudio 自动调用 COLMAP & 训练 NeRF**
   - 在 `worldrecon` 环境中：
     - `ns-process-data images --data ... --output-dir ...`  → 生成 `transforms.json` 等；
     - `ns-train nerfacto --data ...`  → 训练 NeRF；
     - `ns-render ...`  → 导出视频或图像序列。

4. **（可选）引入 3DGS / 4DGS 等显式表示**
   - 使用 `gsplat` 或 Nerfstudio 的 `splatfacto`，将同一场景转换为 3D 高斯云表示；
   - 比较 NeRF 与 3DGS 在质量、效率、编辑友好性等方面的差异；
   - 将这几种表示统一挂接到一个“静态 world model 场景状态”（`world_state.*.json`）上，便于复用。

---

## 静态 World Model 场景状态（Scene State）

为了在“世界模型”视角下复用同一真实场景，本仓库推荐为每个场景维护一个**静态场景状态描述文件**，例如：

- `mipnerf360/db/playroom/world_state.playroom.json`

其核心思想是：

- **统一坐标系**：通常以 COLMAP 的世界坐标系（`cameras.txt` / `images.txt` / `transforms.json`）为基准；
- **挂接多种表示**：在同一 JSON 中记录：
  - `representations.colmap.*`：几何层（稀疏/稠密点云）；
  - `representations.nerfstudio.*`：NeRF / Nerfacto 的数据与训练结果；
  - `representations.gaussians.*`：3DGS（如 `splatfacto` 或原生 3DGS）的运行目录或 checkpoint；
  - `representations.semantics.*`：语义层（`semantic_scene_json`），使场景可查询/可理解。

在代码层面，可以使用 `src/world_model/` 中的：

- `scene_state`：`SceneState`、`load_scene_state` / `save_scene_state`、`example_playroom_state`；
- `semantics`：`SemanticScene`、`load_semantic_scene` / `save_semantic_scene`，以及按类别/区域/点的查询 API（`query_by_class`、`query_region`、`get_semantic_at_point`）。

运行 `scripts/run_semantic_labeling.py` 可为场景生成 2D 语义与可选 3D 点语义，详见 `docs/review_world_model_engineering.md` 语义 Pipeline 章节；语义 3DGS 为先 splatfacto 再挂语义头，见该文档语义 3DGS 小节。

这样，你可以把 **COLMAP poses → Nerfstudio / NeRF → 3DGS → 语义层** 的所有中间结果，都纳入到一个统一的“世界记忆单元”中，后续无论做渲染、编辑还是语义查询，都只需先加载 `SceneState`。

---

## 下一步工作（Roadmap 简要）

- 在 `src/` 目录中补充：
  - `pipelines/colmap_pipeline.py`：封装 SfM / MVS 命令行与 `pycolmap` 调用；
  - `pipelines/nerfstudio_pipeline.py`：一键从场景目录跑完 `ns-process-data` → `ns-train` → `ns-render`；
  - `world_model/` 子模块：探索将 NeRF / 3DGS / 语义标签统一到一个“世界记忆库”接口中。
- 在 `docs/` 中继续扩展文献阅读笔记：
  - 3DGS / 4DGS 核心论文与实现对比；
  - 世界模型（World Model）在 3D 场景中的应用案例整理。

如果你有新的实验想法或更好的结构设计，欢迎直接在本仓库基础上拓展，或提交 PR/Issue 一起打磨这个“世界重建”学习与实验项目。
