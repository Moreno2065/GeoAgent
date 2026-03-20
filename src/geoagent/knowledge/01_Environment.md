# 环境与依赖基座

## 依赖隔离原则

**核心逻辑**：隔离 C++ 依赖（GDAL/PROJ），绝对禁止在 REPL 环境中使用 `pip install` 动态安装核心地理库。

### 为什么要隔离？

GDAL、PROJ 等底层库依赖 C++ 编译，在 REPL 沙盒中动态安装会导致：
- 编译超时
- 路径不兼容
- 内存泄漏

### 依赖层次

| 层级 | 库 | 说明 |
|------|-----|------|
| 底层 | GDAL, PROJ, GEOS | C++ 编译，必须预装 |
| 中层 | shapely, pyproj, fiona | Python 绑定 |
| 上层 | geopandas, rasterio, xarray | 高级 API |
| 可选 | torch, torchgeo, segmentation_models | 深度学习 |

---

## Conda 环境配置

### Windows 部署规范

在 Windows 系统下，必须强制使用 Conda 预先构建环境，以避免底层 C++ 编译器路径报错。

### 标准 environment.yml

```yaml
name: gis_repl_env
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - gdal
  - geopandas>=0.14
  - rasterio
  - xarray
  - rioxarray
  - shapely>=2.0
  - pyproj>=3.6
  - fiona>=1.9
  - numpy>=1.24
  - pandas>=2.0
  - matplotlib>=3.7
  - folium>=0.14
  - whitebox
  - osmnx
  - networkx
  - pyarrow
  - jupyter_client
  # 深度学习可选
  - pytorch::pytorch
  - pytorch::torchvision
```

### 创建环境命令

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate gis_repl_env

# 验证核心库
python -c "import geopandas; import rasterio; print('OK')"
```

---

## 沙盒环境检测

Agent 在执行代码前，应自动检测环境是否就绪：

```python
def check_gis_environment():
    """检测 GIS 环境是否完整"""
    required = ['geopandas', 'rasterio', 'shapely', 'pyproj']
    missing = []
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    
    if missing:
        raise RuntimeError(f"缺少依赖库: {', '.join(missing)}")
    return True

# 沙盒代码第一行自动插入
check_gis_environment()
```

---

## 包管理禁令

### 禁止操作

- ❌ `pip install geopandas` 在运行时安装
- ❌ `pip install rasterio` 在沙盒中安装
- ❌ `conda install gdal` 在 REPL 中执行

### 正确做法

- ✅ 环境预先配置好所有依赖
- ✅ 如需额外库，在 environment.yml 中声明
- ✅ 使用 `importlib` 检查已安装的包

---

## 路径规范

### 工作目录结构

```
project/
├── workspace/          # 用户上传的 GIS 数据
│   ├── *.shp
│   ├── *.tif
│   └── *.geojson
├── outputs/            # 分析结果输出
│   ├── *.png
│   └── *.html
└── knowledge/          # 知识库（可选）
```

### 代码中的路径约定

```python
import os
from pathlib import Path

# 工作目录
WORKSPACE = Path('workspace')
OUTPUTS = Path('outputs')
OUTPUTS.mkdir(exist_ok=True)

# 文件读取
data_path = WORKSPACE / 'data.shp'
raster_path = WORKSPACE / 'dem.tif'

# 结果保存
output_path = OUTPUTS / 'result.png'
```
