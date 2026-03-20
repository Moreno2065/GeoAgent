"""
===================================================================
GeoAgent 第二层架构：全能 GIS Agent 终极四层架构
===================================================================
┌─────────────────────────────────────────────────────────────────┐
│  ① 大脑中枢层  (Orchestration Layer)  →  core.py (LangGraph)   │
│  ② 能力封装层  (Capability Layer)     →  gis_tools/             │
│  ③ 记忆与上下文层(Memory Layer)       →  knowledge_rag.py       │
│  ④ 交互控制层  (UI Layer)            →  app.py                 │
└─────────────────────────────────────────────────────────────────┘

本模块实现 LangGraph 三节点流水线：
  - Planner   : 理解自然语言 → 严格 JSON 任务清单
  - Executor  : 拿着清单 → 调用 GeoToolbox 生成并执行 Python 代码
  - Reviewer  : 检查 workspace/ 下是否生成真实 .shp/.tif 文件

每个 Agent 对话都会注入动态 Workspace State，根治"文件幻觉"。
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set
from openai import OpenAI
import threading

from geoagent.tools import execute_tool
from geoagent.tools.tool_rag import retrieve_gis_tools, format_retrieval_context
from geoagent.knowledge.knowledge_rag import get_workspace_state

# API Key 持久化存储
_API_KEY_FILE = Path(__file__).parent.parent.parent / ".api_key"


# ============================================================================
# 静态域提示词片段（按需动态注入）
# ============================================================================

_RASTER_DOMAIN_PROMPT = """
### 【栅格 / 遥感域】

**铁律：严禁 `dataset.read()` 全量读取！**

1. **元数据先行**：`get_raster_metadata(file_name)` 确认波段数、CRS、尺寸。
2. **超大影像处理**（宽或高 > 10000 px）：使用 `run_gdal_algorithm`。
3. **波段运算**：始终用 `calculate_raster_index`，禁止手写 `src.read()` 后做波段数学。
4. **NDVI 公式**：Sentinel-2 用 `(b2-b1)/(b2+b1)`（b1=NIR, b2=Red），Landsat 8 用 `(b5-b4)/(b5+b4)`。
"""

_VECTOR_DOMAIN_PROMPT = """
### 【矢量 / 空间分析域】

1. **元数据先行**：`get_data_info(file_name)` 确认 CRS、字段列表、几何类型。
2. **CRS 铁律**：矢量叠加前必须 `.to_crs()` 统一坐标系。
3. **空间连接**（`gpd.sjoin`）：用 `predicate="within/intersects"` 指定拓扑关系。
4. **PySAL 高级分析**：莫兰指数需先计算空间权重矩阵 `w = libpysal.weightsQueen.from_dataframe(gdf)`。
"""

_NETWORK_DOMAIN_PROMPT = """
### 【路网 / 网络分析域】

1. **专用路由工具**：`osmnx_routing` 零数据路径规划（实时拉 OSM，无需本地数据）。
2. **中文地名**：`osmnx_routing(city_name="芜湖市", ...)` 支持中文。
3. **可达范围**：`osmnx_routing` 计算后对边长阈值裁剪，再转 GeoDataFrame。
"""

_VIZ_DOMAIN_PROMPT = """
### 【地图可视化域】

1. **交互地图**：生成 `folium.Map` 后 `m.save("output.html")`，Streamlit 自动渲染。
2. **PyDeck 3D**：`render_3d_map(layer_type='column/hexagon/heatmap/scatterplot')`。
3. **注意**：`r.to_html()` 保存为 HTML，禁止 `r.show()`！
"""

_STAC_DOMAIN_PROMPT = """
### 【STAC 云原生遥感域】

1. **专用工具**：`search_stac_imagery` 覆盖 Planetary Computer、AWS 等主流端点。
2. **自动签名**：Planetary Computer 的数据 `search_stac_imagery` 自动处理。
3. **一体化管道**：`stac_to_visualization` — STAC 搜索 → COG 读取 → 3D 可视化，一步到位。
"""

_HOTSPOT_DOMAIN_PROMPT = """
### 【热点分析与智能选址域】

1. **高级热点分析**：`geospatial_hotspot_analysis` — Moran's I + LISA + Gi* 三剑客。
2. **MCDA 智能选址**：`multi_criteria_site_selection` — 多准则决策分析。
3. **可视化**：热点分析结果用 `render_3d_map(layer_type='hexagon')` 可视化。
"""

_GIS_EXPERT_SYSTEM_PROMPT = """你是一个高级 GIS/RS 空间数据科学家，能够代替 ArcMap/ArcGIS Pro 完成各种空间分析任务。

---

## 【知识库检索 - 理解问题时才查阅，执行任务时直接调用工具】

你有一个结构化的 **GIS/RS 知识库**（`01_Environment` / `02_GIS_Core` / `03_GeoAI_Compute` / `04_Agent_Protocols` / `05_GIS_Theory` / `06_Python_Ecosystem` / `07_Advanced_QA` / `08_SelfCorrecting_REPL` / `08_GIS_RS_Comprehensive`）。

**【决策规则 — 必须严格遵守】**

> **⚠️ 黄金法则：先执行，后理解。不要在理解中循环。**

| 任务类型 | 例子 | 正确做法 |
|---------|------|---------|
| **执行型** | "计算NDVI" / "下载数据" / "最短路径" | **立即调用对应工具** |
| **理解型** | "NDVI公式为什么这样设计" | **先检索知识库** |
| **混合型** | "用rasterio计算NDVI，代码怎么写" | **最多1次知识库检索获取模板，然后直接调用工具执行** |

---

## 【ReAct 任务循环】

1. **解析意图**：理解用户的空间分析需求
2. **制定计划**：确定分析步骤和所需工具
3. **调用工具**：通过 function calling 使用合适的工具
4. **获取结果**：分析工具返回的数据
5. **迭代优化**：根据结果调整策略
6. **返回结果**：向用户返回分析结论

---

## 【自修正代码执行循环】

当需要执行自定义 Python 代码时，必须使用 `run_python_code` 工具：

1. **写出代码**：根据任务写出完整 Python 代码
2. **执行**：`run_python_code(code="...")` → 如果 `success=True`，任务完成
3. **分析错误**：如果 `success=False`，仔细阅读 `stderr` 中的错误报告
4. **修复代码**：针对错误信息和 `hint` 提示修改代码
5. **重复**：再次 `run_python_code`，直到 `success=True`

**关键规则**：
- `session_id` 跨调用持久化：同一个任务使用相同的 `session_id`
- 收敛检测：如果 `is_converged=True` 或 `convergence_warning` 出现，说明同一错误重复多次，应停止重复尝试

---

**【GeoToolbox 使用铁律 — 沙盒代码必须遵守】**

> ⚠️ **在 `run_python_code` 沙盒中，全局静态类 `GeoToolbox` 已注入，可直接调用！**
> **必须优先调用 `GeoToolbox` 中的方法，绝对禁止自己手写底层 GIS 逻辑！**

### 可用方法清单

| 模块 | 方法 | 说明 |
|------|------|------|
| `GeoToolbox.Vector` | `.project(in, out, crs)` | 矢量投影转换 |
| `GeoToolbox.Vector` | `.buffer(in, out, dist, dissolve=True)` | 缓冲区分析 |
| `GeoToolbox.Vector` | `.overlay(in1, in2, out, how='intersection')` | 空间叠置 |
| `GeoToolbox.Vector` | `.spatial_join(target, join, out, how, predicate)` | 空间连接 |
| `GeoToolbox.Raster` | `.calculate_index(in, out, "(b4-b3)/(b4+b3)")` | 波段指数 |
| `GeoToolbox.Raster` | `.clip_by_mask(raster, mask, out)` | 栅格裁剪 |
| `GeoToolbox.Network` | `.isochrone(center_address, out, walk_time_mins)` | 等时圈生成 |
| `GeoToolbox.Network` | `.shortest_path(city, origin, dest, out, mode)` | 最短路径 |
| `GeoToolbox.Stats` | `.hotspot_analysis(in, value_col, out)` | 热点分析 |
| `GeoToolbox.Viz` | `.export_3d_map(in, elevation_col, out.html)` | PyDeck 3D |
| `GeoToolbox.CloudRS` | `.search_stac(bbox, start, end, output)` | STAC 搜索 |

### 正确 vs 错误写法对比

```python
# ✅ 正确写法（强制）：
GeoToolbox.Vector.buffer('a.shp', 'a_buf.shp', 100)
GeoToolbox.Raster.clip_by_mask('dem.tif', 'mask.shp', 'dem_clip.tif')
GeoToolbox.Stats.hotspot_analysis('district.shp', 'price', 'hotspots.shp')
GeoToolbox.Viz.export_3d_map('buildings.shp', 'height', '3d_map.html')

# ❌ 错误写法（严禁）：
# gdf = gpd.read_file('a.shp')
# gdf2 = gdf.to_crs('EPSG:4326')   ← 自己写 to_crs 容易出错
# buffered = gdf.buffer(100)         ← 自己写 buffer 容易忘融合
```

---

**【专用工具优先规则 — 必须严格遵守】**

> ⚠️ **当存在专用工具时，禁止用 `run_python_code` 替代！**

| 任务 | 正确工具 | 禁止 |
|------|---------|------|
| 计算 NDVI / NDWI | `calculate_raster_index` | ❌ 写 Python 代码 |
| 裁剪/重投影/坡度 | `run_gdal_algorithm` | ❌ 写 Rasterio 代码 |
| 探查栅格元数据 | `get_raster_metadata` | ❌ 写 Python 代码 |
| 探查矢量元数据 | `get_data_info` | ❌ 写 Python 代码 |
| OSM 路网最短路径 | `osmnx_routing` | ❌ 写 osmnx 代码 |
| STAC 遥感搜索 | `search_stac_imagery` | ❌ 写 pystac-client 代码 |

---

**【CRS 规范 — 叠加分析铁律】**
> - 任何叠置分析前，必须检查 CRS 是否一致
> - 用 `get_data_info` 探查两个图层的 CRS
> - 不一致时用 GeoPandas `.to_crs()` 转换后再叠加

**【中文字符与编码防御铁律 — 必须严格遵守】**
> - 在写文件（尤其是 `open()`, `.to_csv()`, `.to_json()`）时，**必须显式指定 `encoding='utf-8`**！
> - 在使用 `matplotlib` 绘图并包含中文时，代码开头必须添加：
>   `import matplotlib.pyplot as plt`
>   `plt.rcParams['font.sans-serif'] = ['SimHei']`
>   `plt.rcParams['axes.unicode_minus'] = False

---

**【可用工具（通过 function calling 调用）】**

- `get_data_info(file_name)` — 探查矢量文件元数据（CRS、字段、几何类型）
- `get_raster_metadata(file_name)` — 探查栅格文件元数据（CRS、波段数、尺寸）
- `search_online_data(search_query, item_type, max_items)` — 搜索 ArcGIS Online 公开数据
- `download_features(layer_url, where, out_file, max_records)` — 下载 ArcGIS 矢量数据
- `deepseek_search(query, recency_days)` — 联网搜索实时信息
- `amap(action, ...)` — 高德地图 API（地理编码/POI搜索/路径规划/天气）
- `osm(action, ...)` — OSMnx 海外地理分析
- `osmnx_routing(city_name, ...)` — 零数据动态路径规划
- `calculate_raster_index(input_file, band_math_expr, output_file)` — 波段指数计算（**专用工具，优先使用**）
- `run_gdal_algorithm(algo_name, params)` — GDAL/QGIS 算法（**专用工具，优先使用**）
- `run_python_code(code, session_id, reset_session)` — 自修正 Python 代码执行
- `search_gis_knowledge(query)` — 检索 GIS/RS 知识库
- `render_3d_map(...)` — PyDeck 3D 高性能可视化
- `search_stac_imagery(...)` — 增强型 STAC 遥感影像搜索
- `geospatial_hotspot_analysis(...)` — 高级空间热点分析
- `multi_criteria_site_selection(...)` — MCDA 智能选址
- `render_accessibility_map(...)` — 设施可达性 3D 可视化
- `stac_to_visualization(...)` — STAC→COG→3D 可视化一体化管道

**【band_math_expr 使用规则 — 必须严格遵守】**
> ⚠️ `band_math_expr` 必须使用 `b1, b2, ...` 引用波段，禁止使用 NIR/Red 等文字标签！
> 先用 `get_raster_metadata` 查看波段数量，然后用 b1=第1波段, b2=第2波段 ... 来写公式。
- `calculate_raster_index(input_file="sentinel.tif", band_math_expr="(b2-b1)/(b2+b1)", output_file="ndvi.tif")` — Sentinel-2 NDVI（b1=NIR, b2=Red）
- `calculate_raster_index(input_file="landsat8.tif", band_math_expr="(b5-b4)/(b5+b4)", output_file="ndvi.tif")` — Landsat 8 NDVI（b4=Red, b5=NIR）
"""


# ============================================================================
# 工具 Schema（与 registry.py 同步）
# ============================================================================

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_data_info", "description": "探查矢量文件元数据（CRS、字段、几何类型）", "parameters": {"type": "object", "properties": {"file_name": {"type": "string", "description": "文件路径或文件名（相对于 workspace 目录）"}}, "required": ["file_name"]}}},
    {"type": "function", "function": {"name": "get_raster_metadata", "description": "探查栅格文件元数据（CRS、波段数、尺寸、分辨率）", "parameters": {"type": "object", "properties": {"file_name": {"type": "string", "description": "文件路径或文件名"}}, "required": ["file_name"]}}},
    {"type": "function", "function": {"name": "calculate_raster_index", "description": "对栅格文件执行波段数学运算，计算植被/水体等指数。band_math_expr 用 b1, b2... 引用波段。", "parameters": {"type": "object", "properties": {"input_file": {"type": "string"}, "band_math_expr": {"type": "string", "description": "Python/NumPy 波段表达式，如 '(b4 - b3) / (b4 + b3)' (NDVI)"}, "output_file": {"type": "string"}}, "required": ["input_file", "band_math_expr", "output_file"]}}},
    {"type": "function", "function": {"name": "run_gdal_algorithm", "description": "调用 GDAL/QGIS 命令行算法", "parameters": {"type": "object", "properties": {"algo_name": {"type": "string", "description": "GDAL 算法名：warp（重投影）/ clip（裁剪）/ slope（坡度）/ aspect（坡向）/ hillshade（山体阴影）"}, "params": {"type": "object"}}, "required": ["algo_name", "params"]}}},
    {"type": "function", "function": {"name": "search_online_data", "description": "搜索 ArcGIS Online 公开数据", "parameters": {"type": "object", "properties": {"search_query": {"type": "string"}, "item_type": {"type": "string", "default": "Feature Layer"}, "max_items": {"type": "integer", "default": 10}}, "required": ["search_query"]}}},
    {"type": "function", "function": {"name": "download_features", "description": "从 ArcGIS Online 下载矢量数据", "parameters": {"type": "object", "properties": {"layer_url": {"type": "string"}, "where": {"type": "string", "default": "1=1"}, "out_file": {"type": "string"}, "max_records": {"type": "integer", "default": 1000}}, "required": ["layer_url"]}}},
    {"type": "function", "function": {"name": "deepseek_search", "description": "联网搜索实时信息", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "recency_days": {"type": "integer", "default": 30}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "amap", "description": "高德地图 API（国内）：地理编码/POI搜索/路径规划/天气", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["geocode", "regeocode", "poi_text_search", "poi_around_search", "direction_driving", "direction_walking", "direction_transit", "weather_query", "convert_coords", "district"]}, "address": {"type": "string"}, "city": {"type": "string"}, "location": {"type": "string"}, "keywords": {"type": "string"}, "origin": {"type": "string"}, "destination": {"type": "string"}, "coords": {"type": "string"}, "from_": {"type": "string"}, "to_": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "osm", "description": "OSMnx 海外地理分析（国内用 amap）", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["geocode", "poi_search", "network_analysis", "shortest_path", "reachable_area", "elevation_profile", "routing"]}, "location": {"type": "string"}, "keywords": {"type": "string"}, "dist": {"type": "integer", "default": 1000}, "network_type": {"type": "string", "enum": ["drive", "walk", "bike"], "default": "drive"}, "origin": {"type": "string"}, "destination": {"type": "string"}, "weight": {"type": "string", "enum": ["length", "travel_time"], "default": "length"}, "max_dist": {"type": "integer", "default": 5000}, "mode": {"type": "string", "enum": ["drive", "walk", "bike"], "default": "drive"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "osmnx_routing", "description": "零数据动态路径规划：实时拉取 OSM 路网，计算最短路径，生成交互式地图", "parameters": {"type": "object", "properties": {"city_name": {"type": "string", "default": "Wuhu, China"}, "origin_address": {"type": "string"}, "destination_address": {"type": "string"}, "mode": {"type": "string", "enum": ["drive", "walk", "bike"], "default": "drive"}, "output_map_file": {"type": "string"}, "plot_type": {"type": "string", "enum": ["folium", "matplotlib"], "default": "folium"}}, "required": ["city_name"]}}},
    {"type": "function", "function": {"name": "search_gis_knowledge", "description": "检索 GIS/RS 知识库，获取标准代码范例和领域知识", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "run_python_code", "description": "自修正 Python 代码执行工具。模型自己写代码→执行→出错后自我检查修复→循环直到正确", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "要执行的 Python 代码（多行字符串，\\n 换行）"}, "mode": {"type": "string", "enum": ["exec", "eval"], "default": "exec"}, "reset_session": {"type": "boolean", "default": False}, "session_id": {"type": "string"}, "workspace": {"type": "string"}, "get_state_only": {"type": "boolean", "default": False}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "render_3d_map", "description": "PyDeck 高性能 3D 交互式地图（支持百万级点）。layer_type: column/hexagon/heatmap/scatterplot", "parameters": {"type": "object", "properties": {"vector_file": {"type": "string"}, "output_html": {"type": "string"}, "height_column": {"type": "string"}, "color_column": {"type": "string"}, "map_style": {"type": "string", "enum": ["dark", "light", "road", "satellite"], "default": "dark"}, "layer_type": {"type": "string", "enum": ["column", "hexagon", "heatmap", "scatterplot"], "default": "column"}, "radius": {"type": "integer", "default": 100}, "elevation_scale": {"type": "integer", "default": 50}, "opacity": {"type": "number", "default": 0.8}}, "required": ["vector_file"]}}},
    {"type": "function", "function": {"name": "search_stac_imagery", "description": "增强型 STAC 遥感影像搜索，覆盖 Planetary Computer、AWS 等主流端点，支持自动数据签名", "parameters": {"type": "object", "properties": {"bbox": {"type": "array", "items": {"type": "number"}}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "collection": {"type": "string", "default": "sentinel-2-l2a"}, "cloud_cover_max": {"type": "integer", "default": 10}, "max_items": {"type": "integer", "default": 20}, "bands": {"type": "array", "items": {"type": "string"}}, "output_file": {"type": "string"}}, "required": ["bbox", "start_date", "end_date"]}}},
    {"type": "function", "function": {"name": "geospatial_hotspot_analysis", "description": "高级空间热点分析。analysis_type: auto/Moran's I/LISA/Gi*", "parameters": {"type": "object", "properties": {"vector_file": {"type": "string"}, "value_column": {"type": "string"}, "output_file": {"type": "string"}, "analysis_type": {"type": "string", "enum": ["auto", "lisa", "gstar"], "default": "auto"}, "neighbor_strategy": {"type": "string", "enum": ["queen", "rook", "knn"], "default": "queen"}, "k_neighbors": {"type": "integer", "default": 8}, "permutations": {"type": "integer", "default": 999}, "significance_level": {"type": "number", "default": 0.05}}, "required": ["vector_file", "value_column"]}}},
    {"type": "function", "function": {"name": "multi_criteria_site_selection", "description": "MCDA 智能选址 — Plan-and-Execute 核心节点。多准则决策分析（权重自动归一化）", "parameters": {"type": "object", "properties": {"city_name": {"type": "string"}, "criteria_weights": {"type": "object"}, "aoi_bbox": {"type": "array", "items": {"type": "number"}}, "candidate_count": {"type": "integer", "default": 10}, "output_file": {"type": "string"}, "use_amap": {"type": "boolean", "default": True}, "use_osm": {"type": "boolean", "default": True}, "use_stac": {"type": "boolean", "default": False}}, "required": ["city_name", "criteria_weights"]}}},
    {"type": "function", "function": {"name": "render_accessibility_map", "description": "设施可达性 3D 可视化 — PyDeck 蜂窝图展示服务覆盖范围", "parameters": {"type": "object", "properties": {"demand_file": {"type": "string"}, "facilities_file": {"type": "string"}, "output_html": {"type": "string"}, "max_travel_time": {"type": "number", "default": 30.0}, "travel_mode": {"type": "string", "enum": ["drive", "walk", "bike"], "default": "drive"}, "weight_column": {"type": "string"}, "bucket_count": {"type": "integer", "default": 8}}, "required": ["demand_file", "facilities_file"]}}},
    {"type": "function", "function": {"name": "stac_to_visualization", "description": "STAC搜索→COG直接读取→PyDeck 3D可视化 一体化管道", "parameters": {"type": "object", "properties": {"collection": {"type": "string", "default": "sentinel-2-l2a"}, "bbox": {"type": "array", "items": {"type": "number"}}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "cloud_cover_max": {"type": "integer", "default": 10}, "bands": {"type": "array", "items": {"type": "string"}}, "output_dir": {"type": "string"}, "render_type": {"type": "string", "enum": ["natural_color", "false_color", "ndvi", "ndwi"], "default": "natural_color"}, "output_html": {"type": "string"}}, "required": ["collection", "bbox", "start_date", "end_date"]}}},
]




# ============================================================================
# Intent Router（保留但简化，供 System Prompt 动态组装使用）
# ============================================================================

class IntentRouter:
    """
    轻量级 Query 意图分类器。
    根据用户输入中的关键词，识别当前任务涉及哪些 GIS 域。
    """

    DOMAIN_PATTERNS: Dict[str, Dict[str, Any]] = {
        "raster": {
            "keywords": ["遥感", "卫星", "影像", "波段", "NDVI", "NDWI", "EVI", "植被指数", "水体指数",
                         "sentinel", "landsat", "modis", "rasterio", "raster", "tif", "tiff", "geotiff",
                         "DEM", "坡度", "坡向", "山体阴影", "hillshade", "slope", "aspect",
                         "裁剪", "重投影", "镶嵌", "波段运算", "band", "地表温度"],
            "description": "栅格/遥感处理",
            "prompt": _RASTER_DOMAIN_PROMPT,
        },
        "vector": {
            "keywords": ["矢量", "shp", "geojson", "parquet", "geopandas", "空间连接", "叠加分析",
                         "缓冲区", "convex hull", "dissolve", "merge", "clip", "intersect", "union",
                         "泰森多边形", "voronoi", "核密度", "KDE", "空间自相关", "莫兰指数",
                         "LISA", "esda", "pysal", "POI", "兴趣点", "设施选址", "服务区",
                         "CRS", "投影", "坐标转换", "to_crs", "几何", "点线面",
                         "土地利用", "土地覆被", "LULC", "城市规划", "行政区划",
                         "人口密度", "热岛", "犯罪分析", "疾病制图"],
            "description": "矢量/空间分析",
            "prompt": _VECTOR_DOMAIN_PROMPT,
        },
        "network": {
            "keywords": ["路网", "道路", "最短路径", "shortest path", "routing", "导航", "步行",
                         "驾车", "骑行", "drive", "walk", "bike", "isochrone", "reachable area",
                         "OSM", "osmnx", "networkx", "图结构", "od矩阵", "出行", "通勤",
                         "公交", "transit", "地铁", "站点", "拓扑", "连通性"],
            "description": "路网/网络分析",
            "prompt": _NETWORK_DOMAIN_PROMPT,
        },
        "visualization": {
            "keywords": ["地图", "folium", "choropleth", "交互地图", "webmap", "html",
                         "静态图", "出图", "图例", "color", "colormap", "底图", "basemap",
                         "tile", "3D", "pydeck", "deck.gl", "3D地图", "3D柱状图", "蜂窝图",
                         "热力图", "scatterplot", "columnlayer", "hexagonlayer", "heatmaplayer",
                         "大屏", "数据大屏", "可达圈", "可达性", "accessibility"],
            "description": "3D 高性能可视化（PyDeck）+ 交互地图",
            "prompt": _VIZ_DOMAIN_PROMPT,
        },
        "stac": {
            "keywords": ["stac", "pystac", "pystac-client", "planetary computer",
                         "sentinel", "landsat", "cop-dem", "遥感影像", "云量", "cloud cover",
                         "COG", "cloud optimized", "哨兵", "Sentinel-2", "波段读取", "asset href"],
            "description": "STAC 云原生遥感搜索",
            "prompt": _STAC_DOMAIN_PROMPT,
        },
        "hotspot": {
            "keywords": ["热点分析", "hotspot", "空间自相关", "spatial autocorrelation",
                         "morans i", "LISA", "Gi*", "getis-ord", "高-高聚集", "低-低聚集",
                         "选址", "site selection", "MCDA", "多准则决策", "选址分析",
                         "蓝海", "竞争分析", "综合得分", "权重"],
            "description": "空间热点分析 + MCDA 智能选址",
            "prompt": _HOTSPOT_DOMAIN_PROMPT,
        },
    }

    @classmethod
    def classify(cls, query: str) -> Set[str]:
        """对用户 query 进行意图分类，返回涉及的域标签集合。"""
        q = query.lower()
        matched: Set[str] = set()
        for domain, cfg in cls.DOMAIN_PATTERNS.items():
            for kw in cfg["keywords"]:
                if kw.lower() in q:
                    matched.add(domain)
                    break
        if not matched:
            matched.add("general")
        return matched

    @classmethod
    def build_dynamic_system_prompt(cls, query: str, base_prompt: str) -> str:
        """动态组装 System Prompt，注入域专属片段和 Tool RAG 上下文。"""
        domains = cls.classify(query)
        fragments = []
        for domain in domains:
            if domain in cls.DOMAIN_PATTERNS:
                fragments.append(cls.DOMAIN_PATTERNS[domain]["prompt"])

        matched_descs = [cls.DOMAIN_PATTERNS[d]["description"] for d in domains if d in cls.DOMAIN_PATTERNS]
        header = f"\n\n[本次路由域: {' | '.join(matched_descs)}]\n"

        # Tool RAG 检索
        try:
            rag_context = format_retrieval_context(query, top_k=5)
        except Exception:
            rag_context = ""

        rag_section = f"""

---

## 【Tool RAG 动态工具集 — 本次任务专属】

{rag_context}

> 💡 **Tool RAG 使用说明**：上方列出的工具是本任务**最相关**的专项工具，优先使用它们而不是写 Python 代码。

"""

        return base_prompt + "".join(fragments) + header + rag_section


# ============================================================================
# GeoAgent 核心类（升级版：支持 LangGraph 流水线 + Workspace State 注入）
# ============================================================================
# GeoAgent 核心类（升级版：支持 LangGraph 流水线 + Workspace State 注入）
# ============================================================================

class GeoAgent:
    """
    GeoAgent 核心类 — 全能 GIS Agent

    升级点：
    1. LangGraph 三节点流水线（Planner / Executor / Reviewer）
    2. Workspace State 动态注入（根治文件幻觉）
    3. Intent Router 保留，Tool RAG 增强
    4. 事件回调（流式推送到 Streamlit UI）
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        temperature: float = 0.7,
        max_history: int = 20,
        history_file: str = "conversation_history.json",
        enable_search: bool = True,
    ):
        if not api_key:
            api_key = self._load_api_key()
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

        if not api_key.startswith("sk-"):
            raise ValueError(f"无效的 API Key 格式：应以为 'sk-' 开头，当前为：{api_key[:8]}***")

        self.api_key = api_key
        self._save_api_key(api_key)
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_history = max_history
        self.history_file = str(Path(__file__).parent.parent.parent / history_file)
        self.enable_search = enable_search

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.messages: List[Dict[str, Any]] = []
        self._init_system_prompt()
        self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}
        self._load_history()
        self._stop_event = threading.Event()

    # ── API Key 持久化 ──────────────────────────────────────────────────────

    @staticmethod
    def _save_api_key(api_key: str) -> None:
        try:
            with open(_API_KEY_FILE, 'w', encoding='utf-8') as f:
                f.write(api_key)
        except Exception:
            pass

    @staticmethod
    def _load_api_key() -> Optional[str]:
        try:
            if _API_KEY_FILE.exists():
                with open(_API_KEY_FILE, 'r', encoding='utf-8') as f:
                    key = f.read().strip()
                    if key:
                        return key
        except Exception:
            pass
        return None

    # ── 事件类型 ───────────────────────────────────────────────────────────

    class EventType:
        TURN_START = "turn_start"
        LLM_THINKING = "llm_thinking"
        TOOL_CALL_START = "tool_call_start"
        TOOL_CALL_END = "tool_call_end"
        TURN_END = "turn_end"
        FINAL_RESPONSE = "final_response"
        STEP_START = "step_start"
        STEP_END = "step_end"
        PLAN_GENERATED = "plan_generated"
        ERROR = "error"
        COMPLETE = "complete"
        STOPPED = "stopped"

    def _emit(self, callback: Callable, event_type: str, payload: dict):
        if callback:
            try:
                callback(event_type, payload)
            except Exception:
                pass

    # ── System Prompt 初始化（含 Workspace State 注入） ──────────────────────

    def _init_system_prompt(self, user_query: Optional[str] = None):
        """初始化系统提示，动态注入域知识 + Tool RAG + Workspace State。"""
        base = _GIS_EXPERT_SYSTEM_PROMPT.strip()

        if user_query:
            dynamic = IntentRouter.build_dynamic_system_prompt(user_query, base)
        else:
            dynamic = base

        # Workspace State 注入（动态记忆的核心）
        workspace_section = get_workspace_state()
        dynamic = dynamic + f"""

---

## 【工作区动态记忆 — Workspace State】

{workspace_section}
"""
        self.messages = [{"role": "system", "content": dynamic}]

    # ── 历史管理 ───────────────────────────────────────────────────────────

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        first_msg = data[0]
                        if (isinstance(first_msg, dict) and
                                first_msg.get("role") == "system" and
                                first_msg.get("content", "").strip() == _GIS_EXPERT_SYSTEM_PROMPT.strip()):
                            for msg in data:
                                if "id" not in msg:
                                    role = msg.get("role", "user")
                                    prefix = {"system": "sys", "user": "user", "assistant": "asst", "tool": "tool"}.get(role, "msg")
                                    msg["id"] = f"{prefix}_{uuid.uuid4().hex[:8]}"
                                if msg.get("role") == "assistant" and "tool_calls" in msg:
                                    for tc in msg["tool_calls"]:
                                        if "id" not in tc:
                                            tc["id"] = f"tc_{uuid.uuid4().hex[:8]}"
                                        if "type" not in tc:
                                            tc["type"] = "function"
                            self.messages = data
                        else:
                            self._init_system_prompt()
                    else:
                        self._init_system_prompt()
            except Exception:
                self._init_system_prompt()

    def _save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _check_history_limit(self) -> bool:
        non_system = [m for m in self.messages if m.get("role") != "system"]
        return len(non_system) >= self.max_history

    def clear_history(self):
        """清除历史对话"""
        self._init_system_prompt()
        self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}
        if os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
            except Exception:
                pass

    def reset_conversation(self):
        """重置对话历史"""
        self._init_system_prompt()
        self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}
        if os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
            except Exception:
                pass

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return self.messages.copy()

    def save_context(self) -> dict:
        return {"messages": self.messages.copy(), "stats": self.stats.copy()}

    def restore_context(self, context: dict):
        if context:
            self.messages = context.get("messages", [{"role": "system", "content": _GIS_EXPERT_SYSTEM_PROMPT}])
            self.stats = context.get("stats", {"total_turns": 0, "tool_calls": 0, "errors": 0})

    def reset_to_system_prompt(self, user_query: Optional[str] = None):
        self._init_system_prompt(user_query)
        self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()

    def stop(self):
        self._stop_event.set()

    # ── LangGraph Workflow ───────────────────────────────────────────────────

    def chat_langgraph(
        self,
        user_input: str,
        event_callback: Callable = None,
        max_steps: int = 8,
        max_retries: int = 2,
        thread_id: str = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        使用 LangGraph DAG 执行对话 — Planner/Executor/Reviewer 三角色分离。

        DAG 节点流转：
            START → planner → executor → reviewer
                                      ↑          ↓
                                      ← (retry) ← ↓
                                                → summarizer → END

        死循环防护：
        1. Reviewer 是唯一循环守卫，所有重试必须经过 Reviewer
        2. 同一工具连续失败 >= 3 次 → 强制跳过（executor 硬截止）
        3. 单步重试超过 max_retries → 标记 skipped，推进下一步
        4. 计划解析失败 → 自动降级为受限 ReAct 模式

        参数
        ----
        user_input    : 用户输入
        event_callback: 事件回调，签名为 (event_type: str, payload: dict) -> None
        max_steps     : 计划最多步数
        max_retries   : 单步最大重试次数
        thread_id     : 线程 ID（用于 checkpointer 多轮对话）

        Yields
        ------
        事件 dict：plan_start / plan_generated / step_start / step_end /
                   review_pass / review_retry / review_skip / final_response 等

        最终 yield 包含完整结果 dict。
        """
        try:
            from geoagent.workflow import GeoAgentWorkflow
            wf = GeoAgentWorkflow(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                max_steps=max_steps,
                max_retries=max_retries,
            )
            for event in wf.run_stream(user_input, event_callback, thread_id):
                yield event
        except ImportError:
            yield {"event": "error", "payload": {"error": "langgraph 未安装，请运行: pip install langgraph"}}
        except Exception as e:
            yield {"event": "error", "payload": {"error": str(e)}}

    # ── LangGraph 流水线对话 ───────────────────────────────────────────────

    def chat_with_langgraph(
        self,
        user_input: str,
        event_callback: Callable = None,
        max_turns: int = 15,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        LangGraph Plan-and-Execute 模式对话（现已委托给 GeoAgentWorkflow）。

        流水线：
        1. Planner: 生成严格 JSON 计划（含 Workspace State）
        2. Executor: 按计划顺序执行每步
        3. Reviewer: 检查文件是否生成
        4. 生成最终回复
        """
        yield from self.chat_langgraph(user_input, event_callback, max_steps=max_turns)

    # ── 核心聊天方法 ───────────────────────────────────────────────────────

    def _call_api(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用 DeepSeek Chat API（原生 function calling）"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
                choice = response.choices[0]
                message = choice.message
                return {
                    "content": message.content or "",
                    "tool_calls": getattr(message, 'tool_calls', None) or [],
                }
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if ("missing field" in error_str or "invalid_request_error" in error_str):
                    raise last_error
        raise last_error

    def _call_api_json(self, messages: List[Dict[str, Any]], max_tokens: int = 4096) -> str:
        """
        原生 JSON 模式 API 调用 — 根治 JSON 幻觉！

        适用场景：
        - 代码解析（代码块提取）
        - 摘要生成
        - 结构化数据生成
        - 非 function calling 的 JSON 输出场景

        注意：此方法不使用 function calling，因为 response_format 和 tools 互斥
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,  # 低温保证 JSON 稳定性
                    max_tokens=max_tokens,
                    # ✅ 原生客户端 JSON 模式，彻底根治 JSON 幻觉
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return content
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if ("missing field" in error_str or "invalid_request_error" in error_str):
                    raise last_error
        raise last_error

    def chat_stream(
        self,
        user_input: str,
        event_callback: Callable[[str, dict], None] = None,
        max_turns: Optional[int] = 10,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式对话入口 — 统一使用 LangGraph Plan-and-Execute 流水线。
        """
        yield from self.chat_with_langgraph(user_input, event_callback, max_turns)

    def chat(
        self,
        user_input: str,
        max_turns: Optional[int] = 10,
        reasoning_callback: callable = None,
        tool_result_callback: callable = None,
    ) -> Dict[str, Any]:
        """执行一次完整的对话交互（统一使用 LangGraph 流水线）。"""
        for event in self.chat_with_langgraph(user_input, max_turns=max_turns):
            if event.get("mode"):
                    return event
        return {"success": False, "error": "LangGraph 模式异常"}


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_agent(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    max_history: int = 20,
    history_file: str = "conversation_history.json",
    enable_search: bool = True,
) -> GeoAgent:
    """
    创建 GeoAgent 实例。
    """
    return GeoAgent(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_history=max_history,
        history_file=history_file,
        enable_search=enable_search,
    )


# 向后兼容导出
GIS_EXPERT_SYSTEM_PROMPT = _GIS_EXPERT_SYSTEM_PROMPT
IntentRouter = IntentRouter
