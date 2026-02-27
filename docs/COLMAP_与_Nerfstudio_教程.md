## 用 Mip-NeRF 360 学 COLMAP 与 Nerfstudio：原理与完整 Pipeline

本教程结合本仓库结构，从 **Mip-NeRF 360 数据集** 出发，系统讲解：

- **COLMAP 的核心原理**（SfM / MVS）与命令行重建流程；
- **Nerfstudio 的基本原理**（NeRF 数据格式与训练）；
- 如何在本仓库中跑通：  
  **Mip-NeRF 360 → COLMAP → Nerfstudio 渲染** 的一条完整 Pipeline。

示例场景统一使用仓库内的路径约定：

```bash
REPO_ROOT=/home/wenhanxiao/code/3D_Reconstruction_ing
DATA_ROOT=$REPO_ROOT/mipnerf360          # 本仓库内的数据根目录
SCENE_NAME=db/drjohnson                  # 示例场景，可换成 playroom 等
SCENE_ROOT=$DATA_ROOT/$SCENE_NAME
```

---

## 一、整体架构：几何层 + 辐射场层

从“世界模型”的角度，可以把 3D / 360 重建拆成两层：

- **几何层（Geometry）**：  
  - 工具：COLMAP（SfM + MVS）  
  - 目标：估计相机位姿和 3D 点云（稀疏或稠密）  
  - 输出：`cameras.txt`、`images.txt`、`points3D.txt` 以及对应的二进制形式

- **辐射场层（Radiance Field / NeRF）**：  
  - 工具：Nerfstudio（`nerfacto` 等）  
  - 目标：学习一个连续辐射场，给定任意相机位姿都能渲染高质量图像  
  - 输入：几何层提供的相机位姿 + 原始图像  

本仓库的实践路线：

```text
Mip-NeRF 360 多视角图像
    └─ COLMAP（SfM/MVS）
         └─ 相机位姿 + 点云（几何层）
              └─ Nerfstudio（NeRF 训练与渲染）
                   └─ 可视化 / 视频导出 / 后续 3DGS 扩展
```

---

## 二、环境与数据准备

### 2.1 环境：worldrecon（推荐）

本仓库提供了一个完整的 3D 重建 / NeRF 环境配置：

```bash
cd $REPO_ROOT
conda env create -f env/worldrecon.yml
conda activate worldrecon
```

`worldrecon` 环境中包含：

- 基础工具：`python=3.10`, `ffmpeg`, `cmake`, `ninja`, `git`；
- 深度栈：`nerfstudio`, `gsplat`, `torch`（通过 Nerfstudio 依赖安装）、`open3d`, `trimesh` 等；
- 视觉/科学计算：`numpy`, `scipy`, `opencv-python`, `scikit-image`, `matplotlib` 等；
- 语义/世界模型相关（可选）：`transformers`, `timm`, `einops`, `accelerate`, `supervision` 等。

> 建议：  
> - `colmap` 自己单独用一个环境（如 `colmap`）或系统安装即可；  
> - `worldrecon` 只负责 Nerfstudio / NeRF / 3DGS 等高层建模。

### 2.2 数据：Mip-NeRF 360 场景布局

在本仓库中建议将 Mip-NeRF 360 的场景组织为：

```text
3D_Reconstruction_ing/
└── mipnerf360/
    ├── db/
    │   └── drjohnson/
    │       ├── images/          # 原始多视角图像
    │       ├── sparse/0/        # COLMAP SfM 输出（可已有）
    │       └── dense/0/         # COLMAP 稠密输出（可选）
    └── ...
```

如果你已有官方数据，只需保证：

```bash
ls $SCENE_ROOT/images
```

能看到该场景下的所有图像文件即可。

---

## 三、COLMAP 原理概览（SfM / MVS）

### 3.1 稀疏重建（SfM）的核心步骤

COLMAP 的稀疏重建（Structure-from-Motion）主要包括：

1. **特征提取（Feature Extraction）**
   - 对每张图像提取局部特征（默认 SIFT），得到关键点和描述子。
2. **特征匹配（Feature Matching）**
   - 在图像之间进行特征匹配，得出候选对应点对。
3. **初始双视图重建**
   - 选两张合适的图（视差充足）作为初始视图，估计它们的相对位姿，三角化出第一批 3D 点。
4. **增量式重建 + 捆绑调整（Bundle Adjustment）**
   - 逐步加入新图像并优化已有相机与 3D 点参数，最小化重投影误差。

结果就是：

- 每张图的相机位姿（R, t）和内参（f, cx, cy 等）；
- 一个稀疏的 3D 点云，每个点由多个观测支持。

### 3.2 稠密重建（MVS）的核心步骤

在已有相机位姿的前提下，COLMAP 使用多视图立体（MVS）生成稠密深度：

1. 图像去畸变（`image_undistorter`），将图像统一到理想 pinhole 模型；
2. 使用 PatchMatch Stereo 估计每个像素的视差/深度；
3. 融合多视角深度图得到稠密点云（`stereo_fusion`）。

对于 NeRF / Nerfstudio 使用场景，**稀疏层的相机位姿就已经足够**，稠密层可选。

---

## 四、COLMAP 操作 Pipeline（命令行）

本节给出一个标准的 COLMAP 命令行流程，以 `db/drjohnson` 为例；如果你只打算用 Nerfstudio 的封装，可以把这部分当作“理解原理”来阅读。

### 4.1 变量约定

```bash
REPO_ROOT=/home/wenhanxiao/code/3D_Reconstruction_ing
DATA_ROOT=$REPO_ROOT/mipnerf360
SCENE_NAME=db/drjohnson
SCENE_ROOT=$DATA_ROOT/$SCENE_NAME

PROJECT_PATH=$SCENE_ROOT/colmap_project
IMAGE_PATH=$SCENE_ROOT/images
mkdir -p $PROJECT_PATH
```

### 4.2 特征提取

```bash
colmap feature_extractor \
  --database_path $PROJECT_PATH/database.db \
  --image_path $IMAGE_PATH \
  --ImageReader.single_camera 0 \
  --ImageReader.camera_model SIMPLE_RADIAL
```

### 4.3 特征匹配

```bash
colmap exhaustive_matcher \
  --database_path $PROJECT_PATH/database.db
```

> 图像较多时，可以改用 `sequential_matcher` 或 `vocab_tree_matcher`。

### 4.4 稀疏重建（Mapper）

```bash
mkdir -p $PROJECT_PATH/sparse/0

colmap mapper \
  --database_path $PROJECT_PATH/database.db \
  --image_path $IMAGE_PATH \
  --output_path $PROJECT_PATH/sparse
```

完成后，`$PROJECT_PATH/sparse/0/` 中会包含：

- `cameras.bin` / `images.bin` / `points3D.bin`；
- 以及 `project.ini` 等项目配置。

### 4.5 转为 TXT 与可视化

```bash
colmap model_converter \
  --input_path $PROJECT_PATH/sparse/0 \
  --output_path $PROJECT_PATH/sparse/0 \
  --output_type TXT
```

得到 `cameras.txt` / `images.txt` / `points3D.txt` 后，可以：

- 用文本编辑器阅读相机与点云信息；
- 用 `colmap gui` 加载模型进行 3D 交互浏览。

### 4.6 稠密重建（可选）

如需稠密点云，可继续执行：

```bash
mkdir -p $PROJECT_PATH/dense
cp -r $PROJECT_PATH/sparse/0 $PROJECT_PATH/dense/

colmap image_undistorter \
  --image_path $IMAGE_PATH \
  --input_path $PROJECT_PATH/dense/0 \
  --output_path $PROJECT_PATH/dense/0_undistort \
  --output_type COLMAP

colmap patch_match_stereo \
  --workspace_path $PROJECT_PATH/dense/0_undistort \
  --workspace_format COLMAP

colmap stereo_fusion \
  --workspace_path $PROJECT_PATH/dense/0_undistort \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path $PROJECT_PATH/dense/fused.ply
```

---

## 五、Nerfstudio 原理概览

Nerfstudio 是一个围绕 NeRF 及其变体的训练/可视化框架，它的核心思想：

1. **数据解析（Data Parser）**
   - 负责读取一组图像 + 对应相机参数（内外参），统一成 Nerfstudio 内部格式；
   - 对我们来说，最关键的是 `transforms.json`：记录每张图的位姿与相机参数。

2. **模型（如 nerfacto）**
   - 基于神经网络表示一个连续场函数 \( f(\mathbf{x}, \mathbf{d}) \)，输入空间点与视线方向，输出颜色与密度；
   - 使用体渲染积分近似真实图像，与输入图像做误差回传优化。

3. **训练与渲染**
   - `ns-train`：训练某一类模型（如 `nerfacto`）；
   - `ns-render`：在训练好的模型上按给定相机轨迹渲染图片或视频。

在与 COLMAP 结合时，Nerfstudio 会：

- 调用 COLMAP（或读取其输出）估计相机位姿；
- 将结果转换为 `transforms.json` 等；
- 在此基础上进行 NeRF 训练。

---

## 六、Nerfstudio 操作 Pipeline（本仓库）

本节给出在 `worldrecon` 环境中，基于 `mipnerf360/db/drjohnson` 跑通 Nerfstudio 的完整流程。

### 6.1 用 `ns-process-data` 预处理（内部调用 COLMAP）

```bash
conda activate worldrecon

REPO_ROOT=/home/wenhanxiao/code/3D_Reconstruction_ing
DATA_ROOT=$REPO_ROOT/mipnerf360
SCENE_NAME=db/drjohnson
SCENE_ROOT=$DATA_ROOT/$SCENE_NAME
OUT_DIR=$SCENE_ROOT/ns_processed

ns-process-data images \
  --data $SCENE_ROOT/images \
  --output-dir $OUT_DIR \
  --matching-method exhaustive
```

内部会执行：

- COLMAP 特征提取 / 匹配 / SfM；
- 生成 Nerfstudio 所需的：
  - `OUT_DIR/images/`：训练用图像；
  - `OUT_DIR/transforms.json`：相机与位姿；
  - 其它元数据与日志。

> 这一步是“**COLMAP 原理 + Nerfstudio 数据格式**”之间的桥梁：  
> 你已经理解了第 3、4 节的流程，这里让 `ns-process-data` 帮你自动完成。

### 6.2 训练 NeRF 模型（以 `nerfacto` 为例）

```bash
conda activate worldrecon

ns-train nerfacto \
  --data $OUT_DIR \
  --output-dir $SCENE_ROOT/ns_runs/drjohnson_nerfacto
```

训练过程中：

- 终端会输出训练日志与损失曲线；
- 默认会启动一个 Web Viewer（地址通常为 `http://127.0.0.1:7007`）；
- 可以在浏览器中实时查看渲染结果、调整渲染质量、录制相机路径等。

训练输出目录大致为：

```text
$SCENE_ROOT/ns_runs/drjohnson_nerfacto/
├── config.yml            # 训练配置（后续渲染会用到）
├── nerfstudio_models/    # 模型 checkpoint
└── logs / 其它文件
```

### 6.3 用 `ns-render` 导出视频

1. **使用默认相机轨迹渲染一个环绕视频**：

   ```bash
   RUN_DIR=$SCENE_ROOT/ns_runs/drjohnson_nerfacto

   ns-render dataset \
     --load-config $RUN_DIR/config.yml \
     --output-path $RUN_DIR/render_orbit.mp4
   ```

   - 需要 `ffmpeg` 支持（worldrecon 环境已安装）。

2. **使用自定义相机路径渲染**：

   - 在 Web Viewer 中录制并保存一条相机轨迹（导出一个 `camera_path.json`）；
   - 然后执行：

     ```bash
     RUN_DIR=$SCENE_ROOT/ns_runs/drjohnson_nerfacto

     ns-render camera-path \
       --load-config $RUN_DIR/config.yml \
       --camera-path $RUN_DIR/camera_path.json \
       --output-path $RUN_DIR/render_custom.mp4
     ```

   - 这样可以实现更精细的镜头设计（如环绕、推进、俯视等）。

---

## 七、端到端 Pipeline 总结（命令速查版）

以 `db/drjohnson` 为例，完整的从原始图像到 NeRF 渲染可总结为：

```bash
# 0. 环境
conda activate worldrecon

# 1. 数据路径
REPO_ROOT=/home/wenhanxiao/code/3D_Reconstruction_ing
DATA_ROOT=$REPO_ROOT/mipnerf360
SCENE_NAME=db/drjohnson
SCENE_ROOT=$DATA_ROOT/$SCENE_NAME

# 2. Nerfstudio 预处理（内部调用 COLMAP）
OUT_DIR=$SCENE_ROOT/ns_processed
ns-process-data images \
  --data $SCENE_ROOT/images \
  --output-dir $OUT_DIR

# 3. 训练 NeRF（nerfacto）
ns-train nerfacto \
  --data $OUT_DIR \
  --output-dir $SCENE_ROOT/ns_runs/drjohnson_nerfacto

# 4. 渲染默认相机轨迹视频
RUN_DIR=$SCENE_ROOT/ns_runs/drjohnson_nerfacto
ns-render dataset \
  --load-config $RUN_DIR/config.yml \
  --output-path $RUN_DIR/render_orbit.mp4
```

若需要深入理解各步骤的几何含义与实现细节，可返回本教程的第 3–4 节，以及 `docs/文献综述_3D_与_全景重建.md` 中的理论综述部分。

---

## 八、从 Nerfstudio 到 3DGS（splatfacto）与静态场景状态

在完成 **COLMAP → Nerfstudio** 之后，可以进一步使用 Nerfstudio 内置的 3DGS 实现（`splatfacto`）或其它工具，将同一场景转换为**高斯点云表示**，构成一个可复用的“静态 world model 场景状态”。

### 8.1 使用 splatfacto 训练 3D 高斯（示例）

以 `db/drjohnson` 为例，假设已经完成第 6 节中的 `ns-process-data`，并得到了：

- `$SCENE_ROOT/ns_processed/`（包含 `transforms.json` 等）。

则可以直接在 `worldrecon` 环境中训练 `splatfacto`：

```bash
conda activate worldrecon

REPO_ROOT=/home/wenhanxiao/code/3D_Reconstruction_ing
DATA_ROOT=$REPO_ROOT/mipnerf360
SCENE_NAME=db/drjohnson
SCENE_ROOT=$DATA_ROOT/$SCENE_NAME
OUT_DIR=$SCENE_ROOT/ns_processed

ns-train splatfacto \
  --data $OUT_DIR \
  --output-dir $SCENE_ROOT/ns_runs/drjohnson_splatfacto
```

训练输出目录中同样会包含一个 `config.yml` 和若干 checkpoint，可用于后续渲染：

```bash
RUN_DIR=$SCENE_ROOT/ns_runs/drjohnson_splatfacto

ns-render dataset \
  --load-config $RUN_DIR/config.yml \
  --output-path $RUN_DIR/render_orbit.mp4
```

> 提示：可以根据显存情况选择 `splatfacto` 或 `splatfacto-big`，也可以通过命令行参数调节质量与速度。

### 8.2 将多种表示统一到静态 Scene State

为了在“世界模型”的视角下复用同一场景，本仓库推荐为每个场景维护一个 **world state JSON**，例如：

- `mipnerf360/db/playroom/world_state.playroom.json`

它通常包含：

- `scene_id` / `root` / `coordinate_system`：场景标识、根目录与坐标系说明（一般采用 COLMAP 世界坐标）；
- `representations.colmap.*`：几何层结果（`sparse/0`、`dense/0` 等路径）；
- `representations.nerfstudio.*`：NeRF / Nerfacto 的 `ns_processed` 与各个 run 的 `config.yml`；
- `representations.gaussians.*`：3DGS（如 `splatfacto` 或原生 3DGS）训练结果的目录或 checkpoint。

在 Python 中，可以使用 `src/world_model/scene_state.py` 提供的：

- `SceneState` 数据结构；
- `load_scene_state(path)` / `save_scene_state(state, path)`；
- 以及 `example_playroom_state(repo_root)`（用于生成或参考 `playroom` 场景的 world state）。

这样，**COLMAP poses → Nerfstudio / NeRF → 3DGS** 这条链路上的所有中间表示，都可以在一个统一的“静态场景状态”对象中被索引和复用，为后续世界模型实验打下基础。

