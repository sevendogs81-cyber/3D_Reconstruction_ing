# 三维重建动手学习项目

这是一个用于教学与实践的三维重建（Structure-from-Motion / Multi-View Stereo）动手学习项目。项目演示如何使用 `pycolmap` 和常用 Python 工具链进行基于图像的三维重建实验。

主要内容：
- 使用 `pycolmap` 调用 COLMAP 功能进行特征提取、匹配与重建。
- 使用 `numpy`、`scipy`、`opencv-python` 等进行数据处理与可视化。

快速开始：

1. 在项目根目录创建并使用虚拟环境（已包含示例 `.venv`）：

```bash
python -m venv .venv
source .venv/bin/activate
```

2. 安装依赖：

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

3. 运行示例脚本：

```bash
.venv/bin/python colmap.py
```

注意事项：
- 某些 `pycolmap` 功能依赖系统层面的 COLMAP 可执行文件，若出现相关错误，请确保系统已安装 COLMAP。
- 避免将 `.venv` 提交到仓库；本项目已在 `.gitignore` 中忽略虚拟环境目录。

许可证 & 贡献：
- 本仓库为学习用途，可自由修改；如要贡献代码，请发起 Pull Request。

欢迎基于此项目进行实验与扩展。
