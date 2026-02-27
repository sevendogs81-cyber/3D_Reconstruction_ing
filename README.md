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
├── README.md                     # 当前说明文档
├── requirements.txt              # 最小 Python 依赖（pycolmap 等）
├── env/
│   └── worldrecon.yml            # Nerfstudio + 3DGS 等完整环境（conda）
├── docs/
│   ├── 文献综述_3D_与_全景重建.md          # 3D/360 重建 & 世界模型的整体技术路径
│   └── COLMAP_与_Nerfstudio_教程.md       # 从 COLMAP 到 Nerfstudio 的详细实践手册
├── mipnerf360/                   # 示例数据目录（如 db/drjohnson 等场景）
│   └── ...                       # 建议只保留少量示例或用 .gitignore 控制
└── src/（预留）
    ├── pipelines/                # 封装 COLMAP / Nerfstudio / 3DGS 的 Python 管线
    ├── viz/                      # 可视化与评估脚本
    └── world_model/              # 世界模型相关实验代码
```

当前重点已经实现的是：

- `docs/COLMAP_与_Nerfstudio_教程.md`：  
  - 用 Mip-NeRF 360 场景示范 **COLMAP 稀疏/稠密重建完整流程**；  
  - 在后续章节详细给出 **Nerfstudio + COLMAP** 的联动实践（`ns-process-data` / `ns-train` / `ns-render`）。
- `docs/文献综述_3D_与_全景重建.md`：  
  - 从世界模型视角梳理 3D / 360 重建、NeRF、3DGS / 4DGS 的关系与演进；  
  - 给出本项目未来的扩展路线图。

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

详细步骤请阅读 `docs/COLMAP_与_Nerfstudio_教程.md`，这里给一个**高度概括**，方便你建立整体印象：

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
   - 使用 `gsplat` 等工具，将同一场景转换为高斯点云表示；
   - 比较 NeRF 与 3DGS 在质量、效率、编辑友好性等方面的差异。

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
