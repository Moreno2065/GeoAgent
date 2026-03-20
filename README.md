# Geo-Agent 空间智能分析系统

基于 DeepSeek 的类 ArcMap 空间智能分析 Agentic Workflow 系统。

## 项目概述

Geo-Agent 能够将用户的自然语言指令转化为复杂的 GIS 空间分析代码并本地执行。采用**高频固化工具 + 动态代码沙盒(Code Interpreter)** 的双层架构。

## 目录结构

```
GeoAgent/
├── app.py                      # Streamlit 前端界面
├── agent_core.py               # Agent 核心逻辑（DeepSeek API 封装）
├── requirements.txt            # Python 依赖
├── workspace/                  # 用户数据、临时文件、输出地图
│   └── README.md
└── gis_tools/                 # GIS 工具模块
    ├── __init__.py
    ├── fixed_tools.py          # 固化高频工具
    └── code_sandbox.py         # 动态代码沙盒引擎
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

```bash
# 设置 DeepSeek API 密钥
export DEEPSEEK_API_KEY="your-api-key"

# Windows PowerShell
$env:DEEPSEEK_API_KEY="your-api-key"
```

### 3. 启动应用

```bash
streamlit run app.py
```

## 使用说明

1. **上传数据**：在侧边栏上传 GIS 数据文件（支持 Shapefile, GeoJSON, GeoPackage, TIF/DEM 等格式）
2. **输入分析需求**：在下方输入框描述您的空间分析需求
3. **Agent 自动执行**：Agent 会自动分析数据并生成代码
4. **查看结果**：生成的地图会自动显示在界面中

### 支持的分析类型

- **矢量分析**：缓冲区分析、空间连接、叠置分析、泰森多边形
- **栅格分析**：坡度/坡向计算、等高线提取、表面分析、DEM 处理
- **网络分析**：可达性分析、最短路径、服务区分析

## 核心功能

### 1. 数据探查 (`get_data_info`)

Agent 首先调用此工具了解数据的坐标系(EPSG)、边界(BBox)和属性结构。

### 2. 动态代码沙盒 (`execute_dynamic_python`)

对于复杂的长尾分析需求，Agent 动态生成 Python 代码并在本地执行：

- 投影转换（EPSG:4326 ↔ 局部投影）
- geopandas/shapely 矢量操作
- rasterio 栅格处理
- whitebox 高级地形分析
- folium HTML 地图生成

### 3. 自我纠错 (Self-Healing)

当代码执行失败时，Agent 自动分析错误信息，修改代码并重新执行。

## 技术栈

- **前端**：Streamlit
- **Agent**：DeepSeek API (OpenAI 兼容)
- **GIS 核心**：geopandas, shapely, rasterio, whitebox
- **可视化**：folium, leaflet

## 环境要求

- Python 3.9+
- Windows / macOS / Linux
- DeepSeek API Key
