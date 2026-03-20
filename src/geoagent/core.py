"""
GeoAgent 核心模块
封装与 DeepSeek API 的交互，实现多轮对话和工具调用
"""

import itertools
import os
import json
import uuid
import re
from typing import List, Dict, Any, Optional, Generator, Callable
from pathlib import Path
from openai import OpenAI
import threading
import queue

from geoagent.tools import execute_tool

# API Key 持久化存储文件
_API_KEY_FILE = Path(__file__).parent.parent.parent / ".api_key"


def _save_api_key(api_key: str) -> None:
    """保存 API Key 到本地文件"""
    try:
        with open(_API_KEY_FILE, 'w', encoding='utf-8') as f:
            f.write(api_key)
    except Exception:
        pass


def _load_api_key() -> Optional[str]:
    """从本地文件加载 API Key"""
    try:
        if _API_KEY_FILE.exists():
            with open(_API_KEY_FILE, 'r', encoding='utf-8') as f:
                key = f.read().strip()
                if key:
                    return key
    except Exception:
        pass
    return None


# =============================================================================
# System Prompt - GIS 专家角色设定
# =============================================================================

GIS_EXPERT_SYSTEM_PROMPT = """你是一个高级 GIS/RS 空间数据科学家，能够代替 ArcMap/ArcGIS Pro 完成各种空间分析任务。

---

## 【知识库检索 - 理解问题时才查阅，执行任务时直接调用工具】

你有一个结构化的 **GIS/RS 知识库**（`01_Environment` / `02_GIS_Core` / `03_GeoAI_Compute` / `04_Agent_Protocols` / `05_GIS_Theory` / `06_Python_Ecosystem` / `07_Advanced_QA` / `08_SelfCorrecting_REPL` / `08_GIS_RS_Comprehensive`）。

**【决策规则 — 必须严格遵守】**

> **⚠️ 黄金法则：先执行，后理解。不要在理解中循环。**

**第一步：判断任务类型**

| 任务类型 | 例子 | 正确做法 |
|---------|------|---------|
| **执行型** | "计算NDVI" / "下载数据" / "最短路径" / "裁剪TIFF" | **立即调用对应工具**，不要先检索知识库 |
| **理解型** | "NDVI公式为什么这样设计" / "矢量栅格区别" | **先检索知识库**，理解后再回答 |
| **混合型** | "用rasterio计算NDVI，代码怎么写" | **最多1次**知识库检索获取模板，然后**直接调用工具执行** |

**第二步：执行型任务 — 禁止先检索再执行**

执行型任务必须：
1. **直接调用对应工具**（见下方工具表）
2. 工具返回后分析结果，给出回答
3. 工具执行失败后（如文件不存在），才检索知识库获取代码模板

**禁止行为：**
- ❌ 先调用 `search_gis_knowledge` → 再调用工具（多走一步弯路）
- ❌ 反复调用 `search_gis_knowledge`（同一任务最多1次）
- ❌ 用文字描述代替工具调用

**触发知识库检索的条件（仅理解型/混合型）：**
- 被问及 GIS/RS **理论**（如"矢量/栅格模型区别"）
- 被问及 **为什么**某种方法有效（如"NDVI 公式原理"）
- 不确定某种**方法**如何实现，需要代码模板参考
- 涉及 CRS 规范、OOM 防御、GPU 加速等**操作约束**问题

**检索示例：**
```python
# ✅ 理解型：需要知道原理
search_gis_knowledge("NDVI 公式 遥感物理原理")
# ✅ 方法型：不确定代码怎么写
search_gis_knowledge("PySAL 莫兰指数 代码示例")
# ❌ 执行型：直接调用工具，不要先检索
calculate_raster_index(input_file="sentinel.tif", band_math_expr="(b2-b1)/(b2+b1)", output_file="workspace/ndvi.tif")
```

---

## 【GIS 专家推理链 — 行动计划 → 立即执行】

**⚠️ 每次接收到任务时，严格按以下顺序执行：**

**步骤 1：分类任务**
- 执行型？（涉及文件处理/数据下载/路径规划）→ 跳到步骤 2
- 理解型？（问"是什么"或"为什么"）→ 先调用 `search_gis_knowledge`，再回答

**步骤 2：制定行动计划**
- 列出分析步骤，明确每个步骤需要调用的工具

**步骤 3：立即调用工具**
- 按顺序调用工具，不要先解释再调用——调用和解释可以并行

**【内存规范 — 栅格处理铁律】**
> - 面对 `.tif` / `.tiff` 文件，**严禁 `dataset.read()` 全量读取**！
> - 先用 `get_raster_metadata` 确认波段数和坐标系
> - 超大影像（宽或高 > 10000 像素），优先使用 `run_gdal_algorithm`
> - 波段计算（NDVI/NDWI）使用 `calculate_raster_index`

**【CRS 规范 — 叠加分析铁律】**
> - 任何叠置分析前，必须检查 CRS 是否一致
> - 用 `get_data_info` 探查两个图层的 CRS
> - 不一致时用 GeoPandas `.to_crs()` 转换后再叠加

---

## 【ReAct 任务循环】

当你接收到用户的 GIS 分析任务时，执行以下循环：

1. **解析意图**：理解用户的空间分析需求
2. **制定计划**：确定分析步骤和所需工具
3. **调用工具**：通过 function calling 使用合适的工具
4. **获取结果**：分析工具返回的数据
5. **迭代优化**：根据结果调整策略
6. **返回结果**：向用户返回分析结论

---

## 【自修正代码执行循环（Agent 引导）】

当需要执行自定义 Python 代码时，必须使用 `run_python_code` 工具，通过以下自修正循环完成代码调试：

### 标准流程

1. **写出代码**：根据任务写出完整 Python 代码（参考知识库中的标准范例）
2. **执行**：`run_python_code(code="...")` → 如果 `success=True`，任务完成
3. **分析错误**：如果 `success=False`，仔细阅读 `stderr` 中的错误报告
4. **修复代码**：针对错误信息和 `hint` 提示修改代码
5. **重复**：再次 `run_python_code`，直到 `success=True`

### 关键规则

- **session_id 跨调用持久化**：同一个任务使用相同的 `session_id`，变量状态自动保留
- **变量复用**：后续代码块可以直接访问之前代码块中创建的变量（如 gdf、ndvi 等）
- **收敛检测**：如果 `is_converged=True` 或 `convergence_warning` 出现，说明同一错误重复多次，应停止重复尝试，重新分析问题根源（检查数据质量、参考知识库范例、换算法思路）
- **新任务用 `reset_session=True`**：开始新任务时重置会话，避免旧变量干扰
- **执行状态查询**：`get_state_only=True` 可查看当前会话的所有变量、错误模式、执行历史

### run_python_code 调用示例

```python
# 第 1 次：写代码并执行
run_python_code(
    code="import geopandas as gpd\\ngdf = gpd.read_file('data.shp')\\nprint(gdf.head())\\nprint(gdf.crs)",
    session_id="task_001"
)

# 如果出错，查看 stderr 和 hint，修复后再次执行（session_id 相同，变量保留）
run_python_code(
    code="print('当前 gdf 的列名:', gdf.columns.tolist())\\nprint('当前 CRS:', gdf.crs)",
    session_id="task_001"
)

# 完成后，开始新任务（reset_session=True）
run_python_code(
    code="import rasterio\\nwith rasterio.open('dem.tif') as src:\\n    print(src.crs)",
    session_id="task_002",
    reset_session=True
)
```

### 常用快捷函数

在代码中可直接使用：
- `ls()` — 列出 workspace 目录文件
- `show('变量名')` — 查看变量摘要（类型、shape、长度）
- `WORKSPACE` — workspace 目录路径
- `OUTPUTS` — outputs 子目录（自动创建）

---

**【专用工具优先规则 — 必须严格遵守】**

> ⚠️ **当存在专用工具时，禁止用 `run_python_code` 替代！专用工具经过测试，更加可靠。**

| 任务 | 正确工具 | 禁止 |
|------|---------|------|
| 计算 NDVI / NDWI / EVI 等波段指数 | `calculate_raster_index` | ❌ 写 Python 代码 |
| 裁剪/重投影/坡度/等高线（栅格） | `run_gdal_algorithm` | ❌ 写 Rasterio 代码 |
| 探查栅格元数据（CRS/波段/尺寸） | `get_raster_metadata` | ❌ 写 Python 代码 |
| 探查矢量元数据（CRS/字段） | `get_data_info` | ❌ 写 Python 代码 |
| 下载矢量数据 | `download_features` | ❌ 写 Python 代码 |
| OSM 路网最短路径规划 | `osmnx_routing` | ❌ 写 osmnx 代码 |
| 搜索 ArcGIS Online 数据 | `search_online_data` | ❌ 写 arcgis 代码 |

**何时才用 `run_python_code`：**
- 专用工具无法覆盖的复杂自定义分析（如 PySAL 高级统计、多步骤流水线）
- 数据清洗、格式转换等通用 Python 操作
- 必须在专用工具失败后才考虑

**工具选择顺序：**
1. 检查是否存在专用工具 ✅
2. 如有 → 调用专用工具
3. 如无 → 才考虑 `run_python_code`

---

**【可用工具（通过 function calling 调用）】**

- `get_data_info(file_name)` — 探查矢量文件元数据（CRS、字段、几何类型）
- `get_raster_metadata(file_name)` — 探查栅格文件元数据（CRS、波段数、尺寸）
- `search_online_data(search_query, item_type, max_items)` — 搜索 ArcGIS Online 公开数据
- `access_layer_info(layer_url)` — 访问 ArcGIS 图层元数据
- `download_features(layer_url, where, out_file, max_records)` — 下载 ArcGIS 矢量数据
- `deepseek_search(query, recency_days)` — 联网搜索实时信息（天气/新闻/数据/知识/地理编码验证）
- `amap(action, ...)` — 高德地图 API（地理编码/POI搜索/路径规划/天气）
- `osm(action, ...)` — OSMnx 海外地理分析（POI/路网/最短路径/可达范围）
- `osmnx_routing(city_name, ...)` — 零数据动态路径规划（实时拉取 OSM 路网，生成交互式地图）
- `calculate_raster_index(input_file, band_math_expr, output_file)` — 波段指数计算（NDVI/NDWI 等）**【专用工具，优先使用】**
- `run_gdal_algorithm(algo_name, params)` — GDAL/QGIS 算法（裁剪/重投影/坡度/等高线等）**【专用工具，优先使用】**
- `run_python_code(code, session_id, reset_session)` — 自修正 Python 代码执行（仅在专用工具无法覆盖时才使用）

**【osmnx_routing 工具调用示例】**
- `osmnx_routing(city_name="Wuhu, China")` — 无本地数据，实时拉取芜湖市路网，随机演示最短路径
- `osmnx_routing(city_name="Paris, France", origin_address="Eiffel Tower", destination_address="Arc de Triomphe", mode="walk")` — 步行路径规划
- `osmnx_routing(city_name="Tokyo", origin_address="Shibuya Station", destination_address="Tokyo Tower", output_map_file="workspace/tokyo_route.html")` — 生成交互式地图
- `osmnx_routing(city_name="Wuhu", origin_address="芜湖南站", destination_address="方特主题公园", mode="drive")` — 支持中文地名

**【deepseek_search 工具调用示例】**
- `deepseek_search(query="北京今日天气", recency_days=1)` — 实时天气
- `deepseek_search(query="2024年中国人口统计", recency_days=90)` — 最新数据
- `deepseek_search(query="WGS84坐标系EPSG代码", recency_days=365)` — 地理知识
- `deepseek_search(query="how to get from Eiffel Tower to Louvre by walking", recency_days=7)` — 验证英文地点名称

**【amap 工具调用示例】**
- `amap(action="geocode", address="北京市朝阳区望京街道")` — 地址→坐标
- `amap(action="poi_text_search", keywords="餐厅", city="上海")` — POI搜索
- `amap(action="direction_driving", origin="天安门", destination="故宫")` — 驾车规划
- `amap(action="weather_query", location="116.4,39.9")` — 天气查询

**【osm 工具调用示例】**
- `osm(action="geocode", location="Eiffel Tower, Paris")` — 地名→坐标
- `osm(action="poi_search", location="Times Square, New York", keywords="restaurant")` — POI搜索
- `osm(action="routing", origin="Eiffel Tower, Paris", destination="Arc de Triomphe, Paris", mode="walk")` — 步行路径
- `osm(action="reachable_area", location="Times Square, New York", mode="walk", max_dist=3000)` — 可达范围

**【calculate_raster_index 工具调用示例 — 专用工具，禁止替代】**
> ⚠️ **band_math_expr 必须使用 `b1, b2, ...` 引用波段，禁止使用 NIR/Red 等文字标签！**
> 先用 `get_raster_metadata` 查看波段数量，然后用 b1=第1波段, b2=第2波段 ... 来写公式。
- `calculate_raster_index(input_file="sentinel.tif", band_math_expr="(b2-b1)/(b2+b1)", output_file="workspace/ndvi.tif")` — 计算 NDVI（Sentinel-2 双波段：b1=Red, b2=NIR）
- `calculate_raster_index(input_file="landsat8.tif", band_math_expr="(b5-b4)/(b5+b4)", output_file="workspace/ndvi.tif")` — 计算 NDVI（Lansat 8：b4=Red, b5=NIR）
- `calculate_raster_index(input_file="image.tif", band_math_expr="(b2-b4)/(b2+b4)", output_file="workspace/ndwi.tif")` — 计算 NDWI（b2=Green, b4=NIR）
> **正确流程：get_raster_metadata("sentinel.tif") → 确认波段数 → calculate_raster_index(...)**

**【run_gdal_algorithm 工具调用示例 — 专用工具，禁止替代】**
> **执行前必须先检查数据：用 `get_raster_metadata` 确认栅格 CRS/波段，用 `get_data_info` 确认矢量 CRS！**
- `get_raster_metadata(file_name="dem.tif")` → 确认坐标系
- `get_data_info(file_name="study_area.shp")` → 确认裁剪矢量 CRS
- `run_gdal_algorithm(algo_name="warp", params={"input_file":"dem.tif","output_file":"dem_reprojected.tif","target_crs":"EPSG:32650"})` — 栅格重投影
- `run_gdal_algorithm(algo_name="warp", params={"input_file":"dem.tif","cutline":"study_area.shp","output_file":"dem_clipped.tif"})` — 栅格裁剪到矢量边界
- `run_gdal_algorithm(algo_name="dem", params={"input_file":"dem.tif","output_file":"slope.tif","processing":"slope"})` — 计算坡度

**【栅格与遥感处理铁律】**
> - 面对 `.tif` 或 `.tiff` 文件，**严禁一次性将全部像素 read() 到内存**！大型影像 OOM 会导致整个进程崩溃。
> - 在执行栅格任务前，先用 `get_raster_metadata` 确认波段数量和坐标系。
> - 超大影像（宽或高 > 10000 像素），优先使用 GDAL 命令行工具（`run_gdal_algorithm`）。
> - 波段数学计算（NDVI、NDWI 等），使用 `calculate_raster_index` 工具。"""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_data_info",
            "description": "探查矢量文件（Shapefile/GeoJSON/Parquet 等）的元数据，包括 CRS、字段列表、几何类型、行数等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "文件路径或文件名（相对于 workspace 目录）"},
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_raster_metadata",
            "description": "探查栅格文件（GeoTIFF/COG 等）的元数据，包括 CRS、波段数、尺寸、分辨率、数据类型等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "文件路径或文件名（相对于 workspace 目录）"},
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_raster_index",
            "description": "对栅格文件执行波段数学运算，计算植被/水体等指数（NDVI、NDWI、EVI 等）。支持任意 BandMath 表达式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "输入栅格文件路径"},
                    "band_math_expr": {
                        "type": "string",
                        "description": "Python/NumPy 波段表达式，用 b1, b2, ... 引用第 1、2... 波段。如 'NDVI = (b4 - b3) / (b4 + b3)'（Landsat 8）或 'NDWI = (b2 - b5) / (b2 + b5)'（Sentinel-2）"
                    },
                    "output_file": {"type": "string", "description": "输出栅格文件路径"},
                },
                "required": ["input_file", "band_math_expr", "output_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_gdal_algorithm",
            "description": "调用 GDAL/QGIS 命令行算法（裁剪、重投影、坡度、等高线、栅格重采样等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "algo_name": {
                        "type": "string",
                        "description": "GDAL/QGIS 算法名称（使用 gdal 或 gdal_id 前缀）：\n"
                            "  warp          - 重投影（常用参数：src_srs, dst_srs, resampling）\n"
                            "  clip          - 裁剪（配合 cutline 参数使用矢量面裁剪）\n"
                            "  slope         - 坡度（需 DEM 数据，配合 z_factor）\n"
                            "  aspect        - 坡向\n"
                            "  hillshade     - 山体阴影\n"
                            "  gdal:cliprasterbymasklayer  - QGIS 掩膜裁剪栅格\n"
                            "  gdal:reprojectvector         - QGIS 矢量重投影\n"
                            "完整算法列表请调用 search_gis_knowledge 或查看 GDAL 文档",
                    },
                    "params": {
                        "type": "object",
                        "description": "算法参数字典，如 {'input': 'dem.tif', 'output': 'clipped.tif', 'cutline': 'study_area.shp'}",
                    },
                },
                "required": ["algo_name", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_online_data",
            "description": "搜索 ArcGIS Online 上的公开数据图层，支持按关键词搜索 Feature Layer、Map Service 等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "搜索关键词，如 '上海 绿地'、'urban green space'"},
                    "item_type": {"type": "string", "description": "数据类型，默认为 'Feature Layer'"},
                    "max_items": {"type": "integer", "description": "最大返回条数，默认为 10"},
                },
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "access_layer_info",
            "description": "访问 ArcGIS 图层的元数据信息（字段、几何类型、空间范围等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_url": {"type": "string", "description": "ArcGIS FeatureLayer 或 MapServer 的 REST URL"},
                },
                "required": ["layer_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_features",
            "description": "从 ArcGIS Online 下载矢量数据到本地 GeoJSON 文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_url": {"type": "string", "description": "ArcGIS FeatureLayer URL"},
                    "where": {
                        "type": "string",
                        "description": "ArcGIS REST API WHERE 子句过滤条件，使用标准 SQL 表达式。如 '1=1'（全部）、\"NAME='Beijing'\"、'POP > 100000'。字段名区分大小写",
                        "default": "1=1",
                    },
                    "out_file": {"type": "string", "description": "输出文件路径，默认为 'workspace/arcgis_download.geojson'"},
                    "max_records": {"type": "integer", "description": "最大下载记录数，默认为 1000"},
                },
                "required": ["layer_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deepseek_search",
            "description": "联网搜索实时信息（天气、新闻、地理编码验证等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "recency_days": {"type": "integer", "description": "时间范围（天），默认为 30"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "amap",
            "description": "高德地图 API（国内）：地理编码/逆地理编码、POI 搜索、路径规划、坐标转换、天气查询。不同 action 组合不同参数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "操作类型，必须为以下值之一：\n"
                            "  geocode      - 地址转坐标（address → lon/lat），需传 address\n"
                            "  regeocode    - 坐标转地址（lon,lat → address），需传 location\n"
                            "  poi_text_search   - POI 文本搜索，需传 keywords 和 city\n"
                            "  poi_around_search - 周边搜索，需传 location 和 keywords\n"
                            "  direction_driving - 驾车路径规划，需传 origin 和 destination\n"
                            "  direction_walking - 步行路径规划，需传 origin 和 destination\n"
                            "  direction_transit - 公交路径规划，需传 city、origin 和 destination\n"
                            "  weather_query      - 城市天气查询，需传 city\n"
                            "  convert_coords     - 坐标系转换，需传 coords 和 from/to\n"
                            "  district      - 行政区划查询，需传 keywords",
                        "enum": [
                            "geocode", "regeocode", "poi_text_search", "poi_around_search",
                            "direction_driving", "direction_walking", "direction_transit",
                            "weather_query", "convert_coords", "district",
                        ],
                    },
                    "address": {
                        "type": "string",
                        "description": "【geocode 专用】待解析的地址字符串，如 '北京市朝阳区望京街'",
                    },
                    "city": {
                        "type": "string",
                        "description": "【geocode/poi_text_search/direction_transit/weather_query 专用】城市名称，如 '北京'",
                    },
                    "location": {
                        "type": "string",
                        "description": "【regeocode/poi_around_search 专用】坐标点，格式 'lon,lat'，如 '116.481028,39.989643'",
                    },
                    "keywords": {
                        "type": "string",
                        "description": "【poi_text_search/poi_around_search/district 专用】搜索关键词，如 '餐厅'、'银行'",
                    },
                    "origin": {
                        "type": "string",
                        "description": "【direction_driving/walking/transit 专用】起点坐标，格式 'lon,lat'",
                    },
                    "destination": {
                        "type": "string",
                        "description": "【direction_driving/walking/transit 专用】终点坐标，格式 'lon,lat'",
                    },
                    "coords": {
                        "type": "string",
                        "description": "【convert_coords 专用】待转换的坐标串，多个用 ';' 分隔，格式 'lon,lat;lon,lat'",
                    },
                    "from_": {
                        "type": "string",
                        "description": "【convert_coords 专用】源坐标系类型：gps、mapbar、baidu、autonavi",
                    },
                    "to_": {
                        "type": "string",
                        "description": "【convert_coords 专用】目标坐标系类型：gps、mapbar、baidu、autonavi",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "osm",
            "description": "OSMnx 海外地理分析（国内请用 amap）：地名解析、POI 搜索、路网分析、最短路径、可达范围、高程剖面、路径规划。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "操作类型，必须为以下值之一：\n"
                            "  geocode          - 地名解析，需传 location\n"
                            "  poi_search       - POI 搜索，需传 location 和 keywords\n"
                            "  network_analysis - 路网统计，需传 location 和 dist\n"
                            "  shortest_path    - 最短路径分析，需传 origin 和 destination\n"
                            "  reachable_area   - 可达范围，需传 location\n"
                            "  elevation_profile - 高程剖面，需传 location\n"
                            "  routing          - 路径规划，需传 origin 和 destination",
                        "enum": [
                            "geocode", "poi_search", "network_analysis",
                            "shortest_path", "reachable_area",
                            "elevation_profile", "routing",
                        ],
                    },
                    "location": {
                        "type": "string",
                        "description": "【geocode/poi_search/network_analysis/reachable_area/elevation_profile 专用】地名或地址，如 'Central Park, New York'",
                    },
                    "keywords": {
                        "type": "string",
                        "description": "【poi_search 专用】POI 搜索关键词，如 'restaurant'、'park'、'hotel'",
                    },
                    "dist": {
                        "type": "integer",
                        "description": "【poi_search/network_analysis 专用】搜索半径（米），默认 1000",
                        "default": 1000,
                    },
                    "network_type": {
                        "type": "string",
                        "description": "【network_analysis/shortest_path/reachable_area/routing 专用】路网类型：drive（驾车，默认）、walk（步行）、bike（骑行）",
                        "enum": ["drive", "walk", "bike"],
                        "default": "drive",
                    },
                    "origin": {
                        "type": "string",
                        "description": "【shortest_path/routing 专用】起点地名或地址",
                    },
                    "destination": {
                        "type": "string",
                        "description": "【shortest_path/routing 专用】终点地名或地址",
                    },
                    "weight": {
                        "type": "string",
                        "description": "【shortest_path 专用】最短路径权重：length（距离，默认）、travel_time（时间）",
                        "enum": ["length", "travel_time"],
                        "default": "length",
                    },
                    "max_dist": {
                        "type": "integer",
                        "description": "【reachable_area 专用】最大可达距离（米），默认 5000",
                        "default": 5000,
                    },
                    "mode": {
                        "type": "string",
                        "description": "【reachable_area/routing 专用】出行方式：drive（驾车，默认）、walk（步行）、bike（骑行）",
                        "enum": ["drive", "walk", "bike"],
                        "default": "drive",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "osmnx_routing",
            "description": "零数据动态路径规划：实时拉取 OSM 路网，计算最短路径，生成交互式地图。支持中文地址。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {"type": "string", "description": "城市名称，默认为 'Wuhu, China'，支持中文如 '芜湖'"},
                    "origin_address": {"type": "string", "description": "起点地址，如 '芜湖南站'"},
                    "destination_address": {"type": "string", "description": "终点地址，如 '方特主题公园'"},
                    "mode": {
                        "type": "string",
                        "description": "出行方式（仅接受以下小写值）：drive（驾车）、walk（步行）、bike（骑行），默认 drive",
                        "enum": ["drive", "walk", "bike"],
                        "default": "drive",
                    },
                    "output_map_file": {"type": "string", "description": "输出地图文件路径，默认 'workspace/osmnx_route_map.html'"},
                    "plot_type": {"type": "string", "description": "绘图类型：folium（交互式）或 matplotlib（静态图），默认 folium", "enum": ["folium", "matplotlib"], "default": "folium"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gis_knowledge",
            "description": "检索 GIS/RS 知识库，获取标准代码范例和领域知识。当不确定 GIS 库用法、CRS 问题、OOM 问题、Python 生态工具用法时应主动调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询，如 'NDVI 计算 rasterio'、'CRS 投影转换 geopandas'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "【核心】自修正 Python 代码执行工具。模型自己写代码、执行、出错后自我检查修复、循环直到正确。支持：沙盒安全执行（禁止 os.system 等危险操作）、自动捕获 stdout/stderr、变量状态跨调用持久化、错误上下文完整报告、针对性修复提示、收敛检测防止死循环。使用方式：模型写出代码 → run_python_code 执行 → 若出错则根据错误上下文修复代码 → 再次执行，直到 success=True。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码（支持多行字符串，\\n 换行）。相对路径相对于 workspace/ 目录。"},
                    "mode": {"type": "string", "description": "执行模式：exec（语句块，默认）或 eval（单个表达式求值）", "enum": ["exec", "eval"], "default": "exec"},
                    "reset_session": {"type": "boolean", "description": "是否重置会话（清除所有变量和历史）。用于开始一个全新的任务。", "default": False},
                    "session_id": {"type": "string", "description": "会话 ID（用于跨调用保持状态）。默认自动分配。"},
                    "workspace": {"type": "string", "description": "工作空间路径，默认 workspace/"},
                    "get_state_only": {"type": "boolean", "description": "仅返回会话状态，不执行代码", "default": False},
                },
                "required": ["code"],
            },
        },
    },
]


# =============================================================================
# Agent 核心类
# =============================================================================

class GeoAgent:
    """
    Geo-Agent 核心类

    负责：
    - 与 DeepSeek API 通信
    - 管理多轮对话历史
    - 原生 function calling 工具执行
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
        enable_search: bool = True
    ):
        """
        初始化 Geo-Agent

        Args:
            api_key: DeepSeek API 密钥
            model: 模型名称，默认 deepseek-chat（普通对话模型）
            base_url: API 基础 URL
            max_retries: 最大自我纠错重试次数
            temperature: 生成温度
            max_history: 最大历史轮次
            history_file: 对话历史保存文件路径
        """
        if not api_key:
            api_key = _load_api_key()
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("必须提供 API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

        if not api_key.startswith("sk-"):
            raise ValueError(f"无效的 API Key 格式：应以为 'sk-' 开头，当前为：{api_key[:8]}***")

        self.api_key = api_key
        _save_api_key(api_key)

        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_history = max_history
        self.history_file = str(Path(__file__).parent.parent.parent / history_file)
        self.enable_search = enable_search

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.messages: List[Dict[str, Any]] = []
        self._init_system_prompt()
        self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}
        self._load_history()
        self._stop_event = threading.Event()

    # -------------------------------------------------------------------------
    # 停止控制
    # -------------------------------------------------------------------------
    def stop(self):
        """发送停止信号，终止当前正在进行的对话"""
        self._stop_event.set()
    class EventType:
        TURN_START = "turn_start"           # 新轮次开始  payload: {"turn": int}
        LLM_THINKING = "llm_thinking"       # LLM 思考中  payload: {"chunk": str}
        TOOL_CALL_START = "tool_call_start" # 工具调用开始 payload: {"tool": str, "arguments": dict}
        TOOL_CALL_END = "tool_call_end"    # 工具调用结束 payload: {"tool": str, "success": bool, "result": str}
        TURN_END = "turn_end"               # 轮次结束    payload: {"turn": int}
        FINAL_RESPONSE = "final_response"   # 最终回复   payload: {"content": str}
        ERROR = "error"                     # 错误       payload: {"error": str}
        COMPLETE = "complete"                # 全部完成   payload: {}
        STOPPED = "stopped"                 # 用户停止   payload: {}

    def _emit(
        self,
        callback: Callable,
        event_type: str,
        payload: dict,
    ):
        """通过回调安全发送事件"""
        if callback:
            try:
                callback(event_type, payload)
            except Exception:
                pass

    def _chat_stream_core(
        self,
        user_input: str,
        event_callback: Callable,
        max_turns: Optional[int],
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式 chat 核心实现（Generator 风格），每产生一个事件 yield 一个 dict，
        方便前端实时消费。
        """
        history_warning = None
        if self._check_history_limit():
            history_warning = f"已达到历史轮次上限 ({self.max_history})，请手动清空对话"

        self._stop_event.clear()

        self.messages.append({
            "role": "user",
            "id": f"user_{uuid.uuid4().hex[:8]}",
            "content": user_input
        })

        tool_results_summary = []

        for turn in itertools.count():
            if max_turns is not None and turn >= max_turns:
                if max_turns > 0:
                    self._emit(event_callback, self.EventType.ERROR,
                               {"error": f"达到最大轮次限制 ({max_turns})"})
                    self._emit(event_callback, self.EventType.COMPLETE, {})
                    yield {
                        "success": False,
                        "error": f"达到最大轮次限制 ({max_turns})",
                        "turns": max_turns,
                        "stats": self.stats.copy(),
                        "tool_results": tool_results_summary,
                    }
                return

            self.stats["total_turns"] += 1
            self._emit(event_callback, self.EventType.TURN_START, {"turn": turn + 1})

            try:
                # ------------------- API 调用（含增量输出） -------------------
                all_content_chunks = []
                api_response = {"content": "", "tool_calls": []}

                # 构建搜索参数（启用联网搜索）
                extra_body = {}
                if self.enable_search:
                    # DeepSeek 联网搜索通过 deepseek_search 工具实现
                    pass

                # 流式调用 LLM
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    stream=True,
                    **extra_body
                )

                # 收集完整响应（同时触发 LLM_THINKING 事件）
                tool_calls_raw = []
                current_text = ""

                for chunk in stream:
                    if self._stop_event.is_set():
                        self._emit(event_callback, self.EventType.STOPPED, {})
                        return
                    delta = chunk.choices[0].delta
                    # 处理增量文本
                    if delta.content:
                        chunk_text = delta.content
                        all_content_chunks.append(chunk_text)
                        current_text += chunk_text
                        self._emit(
                            event_callback,
                            self.EventType.LLM_THINKING,
                            {"chunk": chunk_text, "full_text": current_text},
                        )
                    # 处理 tool_calls 增量（最后一个 chunk 才完整）
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            while len(tool_calls_raw) <= idx:
                                tool_calls_raw.append({"id": "", "function": {"name": "", "arguments": ""}})
                            if tc_delta.id:
                                tool_calls_raw[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_raw[idx]["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_raw[idx]["function"]["arguments"] += tc_delta.function.arguments

                final_content = "".join(all_content_chunks)

                # 组装 tool_calls（统一 id/type 格式）
                tool_calls = []
                for i, tc_raw in enumerate(tool_calls_raw):
                    tool_calls.append({
                        "id": f"tc_{uuid.uuid4().hex[:8]}" if not tc_raw.get("id") else tc_raw["id"],
                        "type": "function",
                        "function": {
                            "name": tc_raw["function"]["name"],
                            "arguments": tc_raw["function"]["arguments"],
                        }
                    })

                api_response = {
                    "content": final_content,
                    "tool_calls": tool_calls,
                }

                content = api_response.get("content", "")
                tool_calls = api_response.get("tool_calls", [])

                # ------------------- 追加 assistant 消息到历史 -------------------
                assistant_msg = {
                    "role": "assistant",
                    "id": f"asst_{uuid.uuid4().hex[:8]}",
                    "content": content
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                self.messages.append(assistant_msg)

                self._emit(event_callback, self.EventType.TURN_END, {"turn": turn + 1})

                if tool_calls:
                    # ------------------- 执行工具调用 -------------------
                    for i, tc in enumerate(tool_calls):
                        tool_name = tc["function"]["name"]
                        args_raw = tc["function"]["arguments"]
                        try:
                            arguments = json.loads(args_raw)
                        except json.JSONDecodeError:
                            arguments = {}

                        self._emit(
                            event_callback,
                            self.EventType.TOOL_CALL_START,
                            {"tool": tool_name, "arguments": arguments},
                        )

                        tool_result = execute_tool(tool_name, arguments)
                        self.stats["tool_calls"] += 1

                        # 解析 success 状态
                        tool_success = True
                        tool_error_msg = None
                        try:
                            tr_data = json.loads(tool_result)
                            tool_success = tr_data.get("success", True)
                            tool_error_msg = tr_data.get("error")
                        except Exception:
                            pass

                        tool_results_summary.append({
                            "tool": tool_name,
                            "arguments": arguments,
                            "success": tool_success,
                            "error": tool_error_msg,
                            "result": tool_result,
                        })

                        tc_id = assistant_msg["tool_calls"][i]["id"]

                        # 添加工具结果消息
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "id": f"tool_{uuid.uuid4().hex[:8]}",
                            "content": tool_result
                        })

                        self._emit(
                            event_callback,
                            self.EventType.TOOL_CALL_END,
                            {
                                "tool": tool_name,
                                "success": tool_success,
                                "error": tool_error_msg,
                                "result": tool_result,
                            },
                        )

                    # 继续循环，让模型处理工具结果
                    continue

                else:
                    # 无工具调用，返回最终响应
                    self._save_history()
                    self._emit(event_callback, self.EventType.FINAL_RESPONSE, {"content": content})
                    self._emit(event_callback, self.EventType.COMPLETE, {})

                    result = {
                        "success": True,
                        "response": content,
                        "turns": turn + 1,
                        "stats": self.stats.copy(),
                        "tool_results": tool_results_summary,
                    }
                    if history_warning:
                        result["history_warning"] = history_warning
                    yield result
                    return

            except Exception as e:
                import traceback as _tb_module
                tb = _tb_module.format_exc()
                error_str = str(e)

                is_format_error = (
                    "missing field 'id'" in error_str or
                    ("missing field" in error_str and "id" in error_str and "tool_call" in error_str) or
                    ("messages" in error_str and "id" in error_str and ("required" in error_str or "must" in error_str))
                )

                if is_format_error:
                    self._init_system_prompt()
                    self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}
                    if os.path.exists(self.history_file):
                        try:
                            os.remove(self.history_file)
                        except Exception:
                            pass
                    self._emit(
                        event_callback, self.EventType.ERROR,
                        {"error": f"对话历史格式异常，已自动重置。请重新输入您的需求。\n详情: {tb}"},
                    )
                    self._emit(event_callback, self.EventType.COMPLETE, {})
                    yield {
                        "success": False,
                        "error": "对话历史格式异常，已自动重置。请重新输入您的需求。",
                        "auto_reset": True,
                        "turns": turn + 1,
                        "stats": self.stats.copy(),
                        "tool_results": tool_results_summary,
                    }
                    return

                self.stats["errors"] += 1
                self._emit(event_callback, self.EventType.ERROR, {"error": f"{error_str}\n{tb}"})
                self._emit(event_callback, self.EventType.COMPLETE, {})
                yield {
                    "success": False,
                    "error": f"{error_str} (详见终端日志)",
                    "turns": turn + 1,
                    "stats": self.stats.copy(),
                    "tool_results": tool_results_summary,
                }
                return

    def chat_stream(
        self,
        user_input: str,
        event_callback: Callable[[str, dict], None] = None,
        max_turns: Optional[int] = 10,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式对话，通过 event_callback 实时推送事件。

        事件类型（EventType.*）：
            - TURN_START      : 新轮次开始
            - LLM_THINKING    : LLM token 增量（显示打字效果）
            - TOOL_CALL_START : 工具调用开始
            - TOOL_CALL_END   : 工具调用结束
            - TURN_END        : 轮次结束
            - FINAL_RESPONSE  : 最终回复完成
            - ERROR           : 出错
            - COMPLETE        : 全部完成

        用法示例：
            for event in agent.chat_stream("南京医院有哪些", my_callback):
                print(event)

        Returns:
            Generator，每 yield 一次是一个 event dict：
                {"event": EventType.XXX, "payload": {...}, ...result}
        """
        for item in self._chat_stream_core(user_input, event_callback, max_turns):
            yield item

    def _init_system_prompt(self):
        """初始化系统提示"""
        self.messages = [
            {"role": "system", "content": GIS_EXPERT_SYSTEM_PROMPT}
        ]

    def _load_history(self):
        """加载历史对话（仅在系统提示词未变更时恢复，避免旧格式/旧版本数据污染）"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        first_msg = data[0]
                        if (isinstance(first_msg, dict) and
                                first_msg.get("role") == "system" and
                                first_msg.get("content", "").strip() == GIS_EXPERT_SYSTEM_PROMPT.strip()):
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
                            # 系统提示词不匹配（版本变更或外部修改），清空历史
                            self._init_system_prompt()
                    else:
                        self._init_system_prompt()
            except Exception:
                self._init_system_prompt()

    def _save_history(self):
        """保存历史对话"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _check_history_limit(self) -> bool:
        """检查是否达到历史轮次上限"""
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
        # 删除本地对话缓存文件
        if os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
            except Exception:
                pass

    def _call_api(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        调用 DeepSeek Chat API（原生 function calling），支持自动重试和联网搜索

        Args:
            messages: 消息列表

        Returns:
            包含 content 和 tool_calls 的字典
        """
        last_error = None

        # 构建搜索参数（启用联网搜索）
        extra_body = {}
        if self.enable_search:
            # DeepSeek 联网搜索通过 deepseek_search 工具实现
            # 不再使用 extra_body search_params，避免 SDK 版本兼容性问题
            pass

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    **extra_body
                )

                choice = response.choices[0]
                message = choice.message

                return {
                    "content": message.content or "",
                    "tool_calls": getattr(message, 'tool_calls', None) or []
                }
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # API 格式错误（如 missing field 'id'）不重试，直接抛出
                if ("missing field" in error_str or
                    "invalid_request_error" in error_str):
                    raise last_error

        # 所有重试均失败
        raise last_error

    def chat(self, user_input: str, max_turns: Optional[int] = 10,
             reasoning_callback: callable = None,
             tool_result_callback: callable = None) -> Dict[str, Any]:
        """
        执行一次完整的对话交互（支持 function calling）

        Args:
            user_input: 用户输入
            max_turns: 最大轮次数
            reasoning_callback: 推理过程回调，接收 (turn, reasoning) 参数
            tool_result_callback: 工具执行结果回调，接收 (tool_name, result) 参数

        Returns:
            包含最终响应的字典
        """
        history_warning = None
        if self._check_history_limit():
            history_warning = f"已达到历史轮次上限 ({self.max_history})，请手动清空对话"

        # 添加用户消息
        self.messages.append({
            "role": "user",
            "id": f"user_{uuid.uuid4().hex[:8]}",
            "content": user_input
        })

        tool_results_summary = []
        for turn in itertools.count():
            if max_turns is not None and turn >= max_turns:
                return {
                    "success": False,
                    "error": f"达到最大轮次限制 ({max_turns})",
                    "turns": max_turns,
                    "stats": self.stats.copy(),
                    "tool_results": tool_results_summary,
                }

            self.stats["total_turns"] += 1

            try:
                api_response = self._call_api(self.messages)
                content = api_response.get("content", "")
                tool_calls = api_response.get("tool_calls", [])

                # 添加助手消息到历史
                assistant_msg = {
                    "role": "assistant",
                    "id": f"asst_{uuid.uuid4().hex[:8]}",
                    "content": content
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": f"tc_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                self.messages.append(assistant_msg)

                if tool_calls:
                    # 执行所有工具调用
                    for i, tc in enumerate(tool_calls):
                        tool_name = tc.function.name
                        try:
                            arguments = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        tool_result = execute_tool(tool_name, arguments)
                        self.stats["tool_calls"] += 1

                        # 解析 success 状态
                        tool_success = True
                        tool_error_msg = None
                        try:
                            tr_data = json.loads(tool_result)
                            tool_success = tr_data.get("success", True)
                            tool_error_msg = tr_data.get("error")
                        except Exception:
                            pass

                        tool_results_summary.append({
                            "tool": tool_name,
                            "arguments": arguments,
                            "success": tool_success,
                            "error": tool_error_msg,
                            "result": tool_result,
                        })
                        tc_id = assistant_msg["tool_calls"][i]["id"]

                        # 添加工具结果消息
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "id": f"tool_{uuid.uuid4().hex[:8]}",
                            "content": tool_result
                        })

                        # 触发回调
                        if tool_result_callback:
                            try:
                                tool_result_callback(tool_name, arguments, tool_result)
                            except Exception:
                                pass

                    # 继续循环，让模型处理工具结果
                    continue

                else:
                    # 无工具调用，返回最终响应
                    self._save_history()

                    result = {
                        "success": True,
                        "response": content,
                        "turns": turn + 1,
                        "stats": self.stats.copy(),
                        "tool_results": tool_results_summary,
                    }
                    if history_warning:
                        result["history_warning"] = history_warning
                    return result

            except Exception as e:
                import traceback as _tb_module
                tb = _tb_module.format_exc()
                error_str = str(e)
                error_lower = error_str.lower()

                is_format_error = (
                    "missing field 'id'" in error_lower or
                    ("missing field" in error_lower and "id" in error_lower and "tool_call" in error_lower) or
                    ("messages" in error_lower and "id" in error_lower and ("required" in error_lower or "must" in error_lower))
                )

                if is_format_error:
                    self._init_system_prompt()
                    self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}
                    if os.path.exists(self.history_file):
                        try:
                            os.remove(self.history_file)
                        except Exception:
                            pass
                    return {
                        "success": False,
                        "error": f"对话历史格式异常，已自动重置。请重新输入您的需求。\n详情: {tb}",
                        "auto_reset": True,
                        "turns": turn + 1,
                        "stats": self.stats.copy(),
                        "tool_results": tool_results_summary,
                    }

                self.stats["errors"] += 1
                return {
                    "success": False,
                    "error": f"{error_str} (详见终端日志)\n{tb}",
                    "turns": turn + 1,
                    "stats": self.stats.copy(),
                    "tool_results": tool_results_summary,
                }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.messages.copy()

    # -------------------------------------------------------------------------
    # 对话上下文管理（支持多对话）
    # -------------------------------------------------------------------------
    def save_context(self) -> dict:
        """
        保存当前 agent 上下文状态（切换对话时调用）。
        返回一个 dict，包含 messages 列表和 stats。
        """
        return {
            "messages": self.messages.copy(),
            "stats": self.stats.copy(),
        }

    def restore_context(self, context: dict):
        """
        恢复 agent 上下文状态（切换回某对话时调用）。
        """
        if context:
            self.messages = context.get("messages", [{"role": "system", "content": GIS_EXPERT_SYSTEM_PROMPT}])
            self.stats = context.get("stats", {"total_turns": 0, "tool_calls": 0, "errors": 0})

    def reset_to_system_prompt(self):
        """仅重置 messages 为仅含 system prompt，不影响文件历史"""
        self.messages = [
            {"role": "system", "content": GIS_EXPERT_SYSTEM_PROMPT}
        ]
        self.stats = {"total_turns": 0, "tool_calls": 0, "errors": 0}

    def get_stats(self) -> Dict[str, int]:
        """获取执行统计"""
        return self.stats.copy()


# =============================================================================
# 便捷函数
# =============================================================================

def create_agent(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    max_history: int = 20,
    history_file: str = "conversation_history.json",
    enable_search: bool = True
) -> GeoAgent:
    """
    创建 Geo-Agent 实例

    Args:
        api_key: DeepSeek API 密钥
        model: 模型名称，默认 deepseek-chat
        base_url: API 基础 URL
        max_history: 最大历史轮次
        history_file: 对话历史保存文件

    Returns:
        GeoAgent 实例
    """
    return GeoAgent(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_history=max_history,
        history_file=history_file,
        enable_search=enable_search
    )
