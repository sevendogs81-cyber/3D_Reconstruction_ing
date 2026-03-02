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

#### 3.1.1 SfM 关键技术原理（数学简述）

- **特征提取（以 SIFT 为例）**：在尺度空间上检测极值点，得到位置 \((x,y)\)、尺度 \(\sigma\)、主方向 \(\theta\)；在邻域内统计梯度直方图得到 128 维描述子，具有尺度与旋转不变性。
- **对极几何**：两视图下，对应点 \(\mathbf{p}_1 \leftrightarrow \mathbf{p}_2\)（齐次像素坐标）满足 \(\mathbf{p}_2^\top \mathbf{F} \mathbf{p}_1 = 0\)，其中基础矩阵 \(\mathbf{F} = \mathbf{K}_2^{-\top} \mathbf{E} \mathbf{K}_1^{-1}\)，本质矩阵 \(\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}\) 编码相对位姿 \((\mathbf{R}, \mathbf{t})\)。通过八点法或 RANSAC 估计 \(\mathbf{F}/\mathbf{E}\)，再分解得到 \(\mathbf{R},\mathbf{t}\)（含尺度歧义，由三角化后的 3D 点尺度固定）。
- **三角化**：已知两相机投影矩阵 \(\mathbf{P}_1, \mathbf{P}_2\) 与匹配点 \(\mathbf{p}_1, \mathbf{p}_2\)，求 3D 点 \(\mathbf{X}\) 使 \(\mathbf{p}_i \propto \mathbf{P}_i \mathbf{X}\)。常用 DLT（线性 SVD）或中点法（在两射线间取最近点）。
- **捆绑调整（Bundle Adjustment）**：优化所有相机参数与 3D 点，最小化重投影误差：
  \[
  \min_{\{\mathbf{R}_i,\mathbf{t}_i\},\{\mathbf{X}_j\}} \sum_{i,j} \rho \bigl( \| \pi(\mathbf{R}_i \mathbf{X}_j + \mathbf{t}_i) - \mathbf{p}_{ij} \|^2 \bigr),
  \]
  其中 \(\pi\) 为相机投影（含内参），\(\rho\) 为鲁棒核（如 Huber）。增量式 SfM 每加入一张新图后做局部或全局 BA。

### 3.2 稠密重建（MVS）的核心步骤

在已有相机位姿的前提下，COLMAP 使用多视图立体（MVS）生成稠密深度：

1. 图像去畸变（`image_undistorter`），将图像统一到理想 pinhole 模型；
2. 使用 PatchMatch Stereo 估计每个像素的视差/深度；
3. 融合多视角深度图得到稠密点云（`stereo_fusion`）。

对于 NeRF / Nerfstudio 使用场景，**稀疏层的相机位姿就已经足够**，稠密层可选。

#### 3.2.1 MVS 关键技术原理（数学简述）

- **针孔模型与去畸变**：成像关系 \(\mathbf{p} \propto \mathbf{K} (\mathbf{R}\mathbf{X}+\mathbf{t})\)，\(\mathbf{K}\) 为内参（焦距 \(f_x,f_y\)、主点 \(c_x,c_y\) 及径向/切向畸变）。去畸变将观测统一到理想针孔，便于多视图几何一致计算。
- **PatchMatch Stereo**：对参考图每个像素估计一个**局部平面**（深度 \(d\) 与法向 \(\mathbf{n}\)）。通过随机初始化 + 迭代**传播**（邻域复制）与**随机扰动**，在光度一致性（如 NCC、绝对差）下优化 \(d,\mathbf{n}\)，得到每像素视差/深度图。
- **多视图融合**：对各视角的深度图进行**一致性检验**（同一 3D 点在不同视图的深度一致），通过加权平均或投票得到稠密 3D 点云，并写入 PLY 等格式。

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

#### 5.1 NeRF 与体渲染的关键技术原理（数学简述）

- **辐射场表示**：场景由连续场函数 \(f(\mathbf{x}, \mathbf{d})\) 描述，输入为 3D 点 \(\mathbf{x}\) 与视线方向 \(\mathbf{d}\)，输出为颜色 \(\mathbf{c}\) 与体密度 \(\sigma\)（通常用 MLP 拟合）。
- **体渲染**：沿射线 \(\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}\) 积分颜色。离散化后，像素颜色为
  \[
  \hat{C}(\mathbf{r}) = \sum_i T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i,\quad T_i = \exp\biggl(-\sum_{j<i} \sigma_j \delta_j\biggr),
  \]
  其中 \(\delta_i\) 为采样段长，\(T_i\) 为透射率。可微，便于对 \(\mathbf{c},\sigma\) 反向传播。
- **训练目标**：对每条射线计算 \(\hat{C}(\mathbf{r})\)，与真实像素颜色做 MSE（或其它损失），优化 MLP 参数；常用 positional encoding 对 \(\mathbf{x},\mathbf{d}\) 编码以提升高频细节。Nerfacto 等变体在 MLP 结构、采样策略与外观编码上有所改进。

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

#### 8.1.1 3D 高斯与 Splatting 的关键技术原理（数学简述）

- **高斯表示**：每个 3D 高斯由中心 \(\boldsymbol{\mu}\)、协方差 \(\boldsymbol{\Sigma}\)（正定，对应椭球形状与朝向）、不透明度 \(\alpha\) 与球谐系数（或 RGB）表示颜色。为保持正定，常用尺度 \(\mathbf{s}\) 与旋转 \(\mathbf{R}\) 参数化：\(\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top\)，\(\mathbf{S} = \operatorname{diag}(\mathbf{s})\)。
- **投影与 Splatting**：将 3D 高斯按相机投影到 2D，得到 2D 协方差与中心；在像平面上按高斯权重做 **alpha blending**（按深度排序），得到像素颜色。整个过程可微，便于梯度反传。
- **优化**：初始高斯常由 SfM 稀疏点或 NeRF 密度场得到；通过渲染损失（与输入图像 MSE）优化高斯参数（位置、尺度、旋转、颜色、不透明度），并可配合致密化/剪枝（增加高误差处高斯、移除低贡献高斯）提升质量与效率。

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

---

## 九、语义层：使场景可查询/可理解

在几何层与辐射场/3DGS 之上，增加**语义层**是“世界模型”中 ROI 很高的增量：让场景支持**按类别、按区域、按点**的查询，从而“可理解”。

### 9.1 语义层在整体架构中的位置

```text
多视角图像
  → COLMAP（几何层）
  → Nerfstudio / 3DGS（辐射场/显式表示）
  → 语义标注（2D 分割 + 可选 3D 点融合）→ 可查询接口
```

- **2D 语义**：对每张场景图像做语义分割（如 SegFormer + ADE20K 类别），得到每像素类别与每图类别统计。
- **3D 语义**（可选）：利用 COLMAP 稀疏点及其在多视图中的观测，将 2D 标签投票到 3D 点，得到“带语义的稀疏点云”，支持 3D 空间查询。

#### 9.1.1 语义层关键技术原理（数学简述）

- **2D 语义分割**：模型将图像映射到每像素类别 \(y_{uv} \in \{1,\ldots,K\}\)（如 ADE20K 的 \(K=150\)）。典型流程：编码器-解码器（如 SegFormer）输出 \(H\times W\times K\) 的 logits，逐像素 argmax 得到标签；训练时用交叉熵等损失监督。本仓库对每张图保存 label map 与每类像素数（class_counts），便于“该图包含哪些类”的快速查询。
- **3D 语义融合**：对 COLMAP 中每个 3D 点 \(\mathbf{X}\)，已知其在若干图像中的观测（\(\mathbf{p}_{ij}\) 与相机 \(\mathbf{P}_i\)）。将 \(\mathbf{X}\) 反投影到各视图得到像素 \((u,v)\)，在对应 2D label map 上读取 \(y_{uv}\)；对多视图的 \(y\) 做**多数投票**（或加权）得到该点的语义标签与置信度，从而得到“带语义的稀疏点云”。查询时可用 3D 包围盒过滤或最近邻检索（如 `get_semantic_at_point`）。

### 9.2 运行语义标注

在 `worldrecon` 环境中，使用仓库内脚本对已具备 `ns_processed` 的场景做 2D 语义分割，并可选用 COLMAP sparse 做 3D 融合：

```bash
conda activate worldrecon
REPO_ROOT=/home/wenhanxiao/code/3D_Reconstruction_ing
cd $REPO_ROOT

# 仅 2D 语义（只需 ns_processed）
python scripts/run_semantic_labeling.py \
  --processed-dir mipnerf360/db/playroom/ns_processed \
  --output-dir mipnerf360/db/playroom/semantic \
  --scene-id mipnerf360/db/playroom

# 2D + 3D 融合（需 COLMAP sparse 目录）
python scripts/run_semantic_labeling.py \
  --processed-dir mipnerf360/db/playroom/ns_processed \
  --sparse-dir mipnerf360/db/playroom/sparse/0 \
  --output-dir mipnerf360/db/playroom/semantic \
  --scene-id mipnerf360/db/playroom
```

输出目录（如 `mipnerf360/db/playroom/semantic/`）中将包含：

- `semantic_scene.json`：类别表、每张图的语义元数据、以及（若做 3D 融合）每个稀疏点的语义；
- `label_maps/*.npy`：每张图对应的像素级类别 id 图（可选用于可视化或后续分析）。

### 9.3 在 World State 中挂接语义层

在场景的 `world_state.*.json` 中增加 `representations.semantics`，指向上述语义结果：

```json
"semantics": {
  "semantic_scene_json": "semantic/semantic_scene.json"
}
```
（路径相对于场景根目录。）

这样，通过 `SceneState.load_semantic_scene()` 即可在同一场景状态中加载语义，并与几何/NeRF/3DGS 一起使用。

### 9.4 查询 API 使用示例

加载场景状态与语义后，可进行“按类别”“按 3D 区域”“按点”的查询，使场景可理解、可查询：

```python
from pathlib import Path
from src.world_model import load_scene_state
from src.world_model.semantics import load_semantic_scene

REPO_ROOT = Path("/home/wenhanxiao/code/3D_Reconstruction_ing")
state = load_scene_state(REPO_ROOT / "mipnerf360/db/playroom/world_state.playroom.json")
sem = state.load_semantic_scene()
if sem is None:
    sem = load_semantic_scene(REPO_ROOT / "mipnerf360/db/playroom/semantic/semantic_scene.json")

# 按类别查询：哪些图像/点包含“椅子”
result = sem.query_by_class("chair", include_2d=True, include_3d=True)
print("包含 chair 的图像数:", len(result["images"]))
print("包含 chair 的 3D 点数:", len(result["points"]))

# 列出场景中出现的所有类别
for cid, cname in sem.list_classes():
    print(cid, cname)

# 按 3D 包围盒查询
points_in_region = sem.query_region(bbox_min=(0, 0, 0), bbox_max=(2, 2, 2), class_filter="table")

# 查询某空间点的语义（最近邻带语义 3D 点）
ps = sem.get_semantic_at_point(0.5, 0.0, -1.0)
if ps:
    print("该点语义:", ps.class_name, ps.confidence)
```

### 9.5 在 Viewer 中按类别上色（3D 语义查看器）

完成 3D 语义融合后，可使用本仓库提供的 **3D 语义查看器** 在独立窗口中按类别着色查看点云（基于 Open3D）：

```bash
conda activate worldrecon
cd $REPO_ROOT

# 方式一：直接指定 semantic_scene.json
python scripts/view_semantic_3d.py --semantic-json mipnerf360/db/playroom/semantic/semantic_scene.json

# 方式二：通过 world_state 自动定位语义文件
python scripts/view_semantic_3d.py --world-state mipnerf360/db/playroom/world_state.playroom.json
```

要求：`semantic_scene.json` 中需包含 3D 点（即曾用 `run_semantic_labeling.py` 并传入 `--sparse-dir` 做过 3D 融合）。打开后每个语义类别一种颜色，图例在终端打印。

**无显示器（SSH/服务器）时**：无法弹窗，可先导出彩色 PLY 再在本地查看：

```bash
python scripts/view_semantic_3d.py --semantic-json mipnerf360/db/playroom/semantic/semantic_scene.json --export-ply semantic_colored.ply
```

将生成的 `semantic_colored.ply` 下载到本机，用 MeshLab、CloudCompare 或本地 Open3D 打开即可按类别颜色查看。

### 9.6 训练时语义 NeRF 与 3DGS

在已有 2D 语义标签（`run_semantic_labeling.py` 生成的 `label_maps/`）的基础上，本仓库支持**训练时**将语义纳入 NeRF 与 3DGS，使渲染结果可直接输出按类别着色的语义图。

**语义 NeRF（训练时监督）**

- 使用轻量体渲染 + 语义头（`src/semantic_nerf`），对每条射线同时渲染 RGB 与语义 logits，用 2D 标签做交叉熵监督。
- 运行前需已具备 `ns_processed` 与 `semantic/label_maps/*.npy`。

```bash
conda activate worldrecon
cd $REPO_ROOT

python scripts/train_semantic_nerf.py \
  --processed-dir mipnerf360/db/playroom/ns_processed \
  --semantic-dir mipnerf360/db/playroom/semantic \
  --output-dir mipnerf360/db/playroom/semantic_nerf_runs \
  --num-classes 150 \
  --steps 10000
```

- 输出：`semantic_nerf_runs/semantic_nerf.pt`（含 MLP 权重），可用于后续渲染 RGB 与语义图。

**语义 3DGS（在 splatfacto 稠密高斯上挂语义）**

- 先用 **splatfacto** 训好稠密 3DGS（`ns-train splatfacto`），再在其高斯上挂语义 logits，仅训语义头，用 2D 标签监督；几何与 RGB 冻结。
- 运行前需：已存在 splatfacto 的 config.yml（含 checkpoint 的 run 目录）、`ns_processed`、以及 `semantic/label_maps/*.npy`（`run_semantic_labeling.py` 产出）；环境需 `gsplat` 与 `nerfstudio`（worldrecon 已包含）。

```bash
conda activate worldrecon
cd $REPO_ROOT

# 1）先训 splatfacto（若尚未训练）
ns-train splatfacto --data mipnerf360/db/playroom/ns_processed \
  --output-dir mipnerf360/db/playroom/ns_runs/playroom_splatfacto \
  --max-num-iterations 30000

# 2）在 splatfacto 输出目录下找到 config.yml（通常在 ns_processed/splatfacto/<时间戳>/ 下），再训语义头
python scripts/train_semantic_3dgs.py \
  --splatfacto-config mipnerf360/db/playroom/ns_runs/playroom_splatfacto/ns_processed/splatfacto/YYYY-MM-DD_HHMMSS/config.yml \
  --processed-dir mipnerf360/db/playroom/ns_processed \
  --semantic-dir mipnerf360/db/playroom/semantic \
  --output-dir mipnerf360/db/playroom/semantic_3dgs_runs \
  --steps 3000
```

- 输出：`semantic_3dgs_runs/semantic_3dgs.pt`（稠密高斯参数 + 语义 logits），可用于按视角渲染 RGB 与按类别着色的语义图。  
- 说明：`--splatfacto-config` 中的 `YYYY-MM-DD_HHMMSS` 需替换为 `ns-train splatfacto` 实际生成的时间戳目录名（在 `ns_runs/playroom_splatfacto/ns_processed/splatfacto/` 下查看）。

**在 Viewer 里查看 NeRF/3DGS 语义上色**

训练完成后，用渲染脚本生成 RGB 与按类别着色的语义图，保存为图片后即可用任意图片查看器或浏览器查看上色效果：

```bash
# 语义 NeRF：渲染第一帧视角
python scripts/render_semantic_nerf.py \
  --checkpoint mipnerf360/db/playroom/semantic_nerf_runs/semantic_nerf.pt \
  --processed-dir mipnerf360/db/playroom/ns_processed \
  --output-dir mipnerf360/db/playroom/semantic_nerf_runs/renders \
  --frame-idx 0

# 语义 3DGS：渲染第一帧视角
python scripts/render_semantic_3dgs.py \
  --checkpoint mipnerf360/db/playroom/semantic_3dgs_runs/semantic_3dgs.pt \
  --processed-dir mipnerf360/db/playroom/ns_processed \
  --output-dir mipnerf360/db/playroom/semantic_3dgs_runs/renders \
  --frame-idx 0
```

输出目录中会生成 `rgb.png`、`semantic_colormap.png`（按类别着色）和 `rgb_semantic_sidebyside.png`（左右对比），用系统图片查看器或浏览器打开即可在 Viewer 中查看上色效果。更换 `--frame-idx` 可渲染不同视角。

以上能力共同构成“可查询/可理解”的语义层，与现有几何层、辐射场层一起，形成更完整的世界模型静态场景状态。

