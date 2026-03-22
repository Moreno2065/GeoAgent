# GeoAgent — 空间智能分析 Agent 系统

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Version](https://img.shields.io/badge/Version-0.3.0-2ecc71)
![License](https://img.shields.io/badge/License-MIT-orange.svg)

**GeoAgent** 是一个基于 LLM 的空间智能 GIS 分析 Agent 系统。用户通过自然语言（中英文皆可）描述分析需求，系统将其转化为结构化任务并调用确定性后端引擎执行。覆盖路径规划、缓冲区分析、叠加分析、插值计算、阴影分析、NDVI 计算、热点分析、可视化等八大场景。

> 核心理念：**LLM 仅负责翻译（自然语言 → GeoDSL），后端代码负责路由和执行，无 ReAct 推理循环**。

---

## 目录

- [项目架构](#项目架构)
- [快速开始](#快速开始)
- [五层架构详解](#五层架构详解)
- [使用示例](#使用示例)
- [核心功能](#核心功能)
- [插件系统](#插件系统)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 项目架构

### 设计原则

GeoAgent 采用**七层收敛（Seven-Layer Convergence）**架构，确保分析过程可预测、可测试、可解释：

```
┌─────────────────────────────────────────────────────────┐
│                    用户自然语言输入                       │
│            "从芜湖南站到方特欢乐世界的步行路线"             │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1 · 输入层   InputParser                         │
│  输入验证 · 标准化 · 多模态路由（文字/语音/地图点击）       │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2 · 意图层   IntentClassifier（无 LLM）          │
│  关键词匹配 → ROUTE（置信度 0.95）                       │
│  覆盖 8 大场景：路径/缓冲/叠加/插值/阴影/统计/栅格/可视化  │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3 · 编排层   ScenarioOrchestrator                │
│  正则提取参数 → 缺失时主动澄清 → 参数校验                  │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 4 · Reasoner 层   GeoAgentReasoner              │
│  LLM 翻译 NL → GeoDSL JSON（DeepSeek 模型支持） │
│  支持单步任务和多步骤工作流（workflow）                    │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 5 · DSL 层   DSLBuilder / SchemaValidator        │
│  GeoDSL 结构化协议（Pydantic 模型验证）                   │
│  scenario / inputs / parameters / outputs / steps       │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 6 · 执行层   TaskRouter + Executors              │
│  确定性执行：RouteExecutor / BufferExecutor 等 10 个执行器  │
│  自动引擎回退：GeoPandas → ArcPy，Amap → OSMnx           │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 7 · 渲染层   ResultRenderer                      │
│  业务友好输出：ExplanationCard + BusinessConclusion      │
└─────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 说明 |
|------|------|
| **意图分类无 LLM** | Layer 2 使用中英文关键词字典，零 token 消耗，响应稳定 |
| **Reasoner 翻译层** | Layer 4 使用 LLM 将 NL 翻译为 GeoDSL JSON，支持 DeepSeek 模型 |
| **无 ReAct 循环** | LLM 仅做 NL → GeoDSL 翻译，后端代码负责路由，不做工具选择 |
| **多步骤工作流** | Reasoner 支持 `translate_workflow()` 输出 steps 数组的复杂任务 |
| **参数澄清机制** | 缺失必填参数时主动询问用户，而非猜测 |
| **自动引擎回退** | 每个 Executor 优先使用开源引擎（GeoPandas 等），自动回退到商业引擎（ArcPy） |
| **国内外自动路由** | 路径规划自动检测地址：国内 → 高德地图，海外 → OSMnx |

---

## 快速开始

### 1. 安装

```bash
# 方式一：从源码安装（推荐）
pip install -e .

# 方式二：选择性安装
pip install geoagent[core]          # 核心功能
pip install geoagent[all]          # 所有功能（推荐）
pip install geoagent[full]         # 完整依赖（含 ArcGIS/LangChain/RAG）

# 方式三：开发模式
pip install -e ".[dev]"
```

### 2. 配置 API 密钥

GeoAgent 使用 LLM 进行自然语言理解，支持 DeepSeek 模型：

```bash
# DeepSeek API（推荐）
echo DEEPSEEK_API_KEY="your-deepseek-api-key" > .env

# 国内路径规划（可选，高德地图）
echo AMAP_API_KEY="your-amap-api-key" > .env
```

> 无需手动指定 `.env` 文件路径，GeoAgent 会自动从当前工作目录或用户目录 `~/.geoagent/` 读取。

### 3. 启动

```bash
# Web 界面（推荐）
streamlit run app.py

# 命令行
geoagent

# Python API
python -c "from geoagent import GeoAgent; print(GeoAgent().compile('从芜湖南站到方特欢乐世界的步行路线'))"
```

---

## 七层架构详解

### Layer 1 — 输入层 (`layers/layer1_input.py`)

接收并标准化用户输入：

- **文本输入**：自然语言查询（中英文）
- **输入验证**：长度限制、危险字符过滤
- **多模态预留**（MVP 仅支持文本）：语音、地图点击

```python
from geoagent.layers.layer1_input import InputParser

parser = InputParser()
user_input = parser.parse("从芜湖南站到方特欢乐世界的步行路线")
# → UserInput(source='text', raw="...", lang='zh', validated=True)
```

### Layer 2 — 意图层 (`layers/layer2_intent.py`)

基于关键词的零-shot 意图分类，无需调用 LLM：

| 场景 | 关键词（中文） | 关键词（英文） |
|------|---------------|---------------|
| `ROUTE` | 路径、路线、导航、步行、开车、公交、最近 | route, navigation, walk, drive |
| `BUFFER` | 缓冲、周边、附近、多少米 | buffer, nearby, within |
| `OVERLAY` | 叠加、相交、裁剪、合并、叠置 | overlay, intersect, clip, union |
| `INTERPOLATION` | 插值、IDW、克里金、空间插值、预测 | interpolation, IDW, kriging |
| `VIEWSHED` / `SHADOW` | 视域、阴影、日照、采光、通视 | viewshed, shadow, sunlight, sightline |
| `STATISTICS` | 统计、汇总、面积、周长 | statistics, summary, area |
| `RASTER` | 栅格、NDVI、遥感指数、植被、水体 | raster, NDVI, vegetation, water |
| `VISUALIZATION` | 可视化、地图、渲染、显示 | visualization, map, render |

### Layer 3 — 编排层 (`layers/layer3_orchestrate.py`)

确定性参数提取三步走：

1. **ParameterExtractor**：正则表达式从文本中提取关键参数
   - 距离：`500米`、`3公里`
   - 地点：`芜湖南站`、`方特欢乐世界`
   - 模式：`步行`、`开车`、`骑行`
   - 文件路径：`data/buildings.shp`
2. **ClarificationEngine**：检查必填参数是否完整，缺失时生成 `ClarificationQuestion`
3. **ParameterFiller**：自动填充默认值

```python
from geoagent.layers.layer3_orchestrate import ScenarioOrchestrator

orchestrator = ScenarioOrchestrator()
result = orchestrator.orchestrate(IntentResult(scenario="route", confidence=0.95), user_input)
# → OrchestrationResult(status=PIPELINE_STATUS.ORCHESTRATED, params={...})
#   若参数缺失 → status=PIPELINE_STATUS.NEED_CLARIFICATION, questions=[...])
```

### Layer 4 — Reasoner 层 (`layers/reasoner.py`)

基于 LLM 的自然语言 → GeoDSL 翻译器，支持 DeepSeek 模型：

```python
from geoagent.layers.reasoner import GeoAgentReasoner, get_reasoner

# DeepSeek Reasoner
reasoner = get_reasoner(api_key="sk-...")
dsl = reasoner.translate("对安徽师范大学周边500米做缓冲区分析")
# → {"scenario": "buffer", "task": "buffer", "inputs": {"input_layer": "安徽师范大学"}, ...}

# 多步骤工作流翻译
workflow_dsl = reasoner.translate_workflow("分析成都IFS周边1公里步行道路总长度")
# → {"is_workflow": true, "steps": [{"task": "geocode", ...}, {"task": "fetch_osm", ...}, ...]}
```

**支持的模型：**
| 模型 | base_url | 说明 |
|------|----------|------|
| `deepseek-reasoner` | api.deepseek.com | DeepSeek 推理模型（推荐） |
| `deepseek-chat` | api.deepseek.com | DeepSeek 通用对话 |
| `deepseek-v3` | api.deepseek.com | DeepSeek V3 |

### Layer 5 — DSL 层 (`layers/layer4_dsl.py`)

将编排结果构建为 `GeoDSL` 结构化协议（Pydantic 模型）：

```python
from geoagent.layers.layer4_dsl import DSLBuilder, SchemaValidator

builder = DSLBuilder()
dsl = builder.build(orchestration_result)
# → GeoDSL(version="1.0", scenario=ScenarioType.ROUTE,
#          task="route", inputs={...}, parameters={...}, outputs={...})

validator = SchemaValidator()
validator.validate(dsl)  # 每个场景有独立的必填字段校验规则
```

### Layer 6 — 执行层 (`layers/layer5_executor.py`)

确定性任务路由和执行，无任何 LLM 推理：

```python
from geoagent.layers.layer5_executor import execute_task

result = execute_task(dsl.model_dump())
# → ExecutorResult(success=True, data={route_geojson, distance, duration})
```

### Layer 7 — 渲染层 (`layers/layer6_render.py`)

将技术执行结果转换为业务友好输出：

```python
from geoagent.layers.layer6_render import ResultRenderer

renderer = ResultRenderer()
render_result = renderer.render(executor_result, dsl)
# → RenderResult(
#       summary="从芜湖南站到方特欢乐世界的步行路线，全长约2.3公里，预计步行28分钟",
#       key_findings=["路线途经赭山公园，景色优美", "周边有多个公交站点可换乘"],
#       recommendations=["推荐早高峰时段使用公交出行"],
#       metrics={"distance_km": 2.3, "duration_min": 28, "elevation_gain_m": 15},
#       output_files=["route.html", "route.geojson"]
#   )
```

---

## 使用示例

### 示例 1：路径规划（自动国内外路由）

```python
from geoagent import GeoAgent

agent = GeoAgent()
result = agent.compile("从芜湖南站到方特欢乐世界的步行路线")
print(result["summary"])
# → "从芜湖南站到方特欢乐世界的步行路线，全长约2.3公里，预计步行28分钟"
```

国内地址自动调用高德地图 API；英文/海外地址自动切换到 OSMnx + NetworkX。

### 示例 2：缓冲区分析

```python
result = agent.compile("以合肥南站为中心，半径1公里范围内有哪些地铁站？")
print(result["geojson"])  # GeoJSON 格式结果
```

### 示例 3：插值分析（缺失参数澄清）

```python
# 缺失分辨率参数 → 系统主动询问
result = agent.compile("根据CSV中的散点数据进行空间插值")
# → {"status": "NEED_CLARIFICATION", "questions": [
#      {"param": "resolution", "text": "请指定插值网格分辨率（米）",
#       "options": ["100m", "200m", "500m"]}
#   ]}
```

### 示例 4：NDVI 计算

```python
result = agent.compile("计算 landsat8 数据中2024年6月的NDVI并导出GeoTIFF")
```

支持 Sentinel-2、Landsat-8/9 自动波段映射；也支持自定义波段表达式。

### 示例 5：热点分析

```python
result = agent.compile("对城市POI数据进行Getis-Ord Gi*热点分析，权重矩阵用K近邻")
```

使用 PySAL（esda + libpysal）本地计算，输出 HH/LL/HL/LH 四象限分类。

### 示例 6：阴影分析

```python
result = agent.compile("计算2024年6月21日夏至中午12:00时建筑物的阴影范围")
```

自动从数据中检测高度字段，使用 Shapely 3D 几何计算太阳位置和阴影投射。

---

## 核心功能

### 十大执行器

| 执行器 | 场景类型 | 主引擎 | 备用引擎 | 特色功能 |
|--------|---------|--------|---------|---------|
| `RouteExecutor` | 路径规划 | 高德地图 API | OSMnx + NetworkX | 国内外自动路由；步行/驾车/公交 |
| `BufferExecutor` | 缓冲区分析 | GeoPandas + Shapely | ArcPy | CRS 自动检测；支持 dissolve |
| `OverlayExecutor` | 叠加分析 | GeoPandas | ArcPy | intersect/union/clip/difference |
| `IDWExecutor` | 空间插值 | NumPy/SciPy | GDAL gdal_grid, PyKrige | IDW（可调幂次）/克里金/最近邻 |
| `NdviExecutor` | 遥感指数 | rasterio + NumPy | ArcPy | 30+ 遥感指数；云端 COG 直接读取 |
| `ShadowExecutor` | 阴影分析 | Shapely 3D | ArcGIS 3D Analyst | 自动高度检测；任意时间/地点 |
| `HotspotExecutor` | 热点分析 | PySAL | ArcGIS | Gi*/Moran's I/LISA；Queen/Rook/KNN 权重 |
| `VizExecutor` | 可视化 | Folium | PyDeck, Matplotlib | 多图层地图；PyDeck 3D 可视化 |
| `PostGISExecutor` | 数据库 | psycopg2 | SQLAlchemy | 空间查询；读写 PostGIS |
| `GeneralExecutor` | 通用任务 | py_repl 沙盒 | advanced_tools | 复杂多步任务；高级工具调用 |

### 参数澄清机制

当用户输入不足以完成分析时，系统会主动暂停并请求澄清：

```
用户: "帮我做一个分析"
     ↓
系统: "请问您需要哪种类型的分析？
      ① 路径规划 ② 缓冲区分析 ③ 叠加分析 ④ 插值计算
      或者请直接描述您的需求，例如：'附近500米有哪些设施'"
```

---

## 插件系统

### 高德地图插件 (`plugins/amap_plugin.py`)

| 功能 | 方法 | 用途 |
|------|------|------|
| 地理编码 | `geocode()` / `reverse_geocode()` | 地址 → 坐标 / 坐标 → 地址 |
| POI 搜索 | `poi_text_search()` / `poi_around_search()` | 关键词查询 / 周边查询 |
| 路径规划 | `direction_walking/driving/transit()` | 步行/驾车/公交路线 |
| 天气查询 | `weather_query()` | 实时天气 |
| 坐标转换 | `convert()` | GCJ-02 ↔ WGS-84 |

### OpenStreetMap 插件 (`plugins/osm_plugin.py`)

| 功能 | 方法 | 用途 |
|------|------|------|
| 地理编码 | `geocode()` | Nominatim 地理编码 |
| POI 搜索 | `poi_search()` | OSM 数据 POI 查询 |
| 网络分析 | `get_network()` | 下载 OSM 道路网络 |
| 最短路径 | `shortest_path()` | NetworkX 最短路径 |
| 可达范围 | `reachable_area()` | 等时圈分析 |
| 高程剖面 | `elevation_profile()` | 路径高程变化 |

---

## 技术栈

### 核心运行时
`streamlit` · `openai` · `python-dotenv` · `requests` · `duckduckgo-search`

### GIS 核心
`geopandas` · `shapely` · `rasterio` · `whitebox` · `fiona` · `pyproj` · `rioxarray` · `xarray`

### 高级 GIS
`osmnx` · `networkx` · `libpysal` · `esda` · `pysal` · `laspy` · `earthpy` · `spyndex`（30+ 遥感指数） · `pystac-client` · `planetary-computer`

### 可视化
`folium` · `leafmap` · `matplotlib` · `plotly` · `streamlit-folium` · `pydeck` · `contextily`

### AI / Agent
`langchain-openai` · `langchain-core` · `langchain-experimental` · `langchain-community` · `langgraph`

### RAG 检索
`faiss-cpu` · `pypdf` · `tiktoken` · `unstructured`

### 可选：商业 GIS
`arcgis` · `arcgis-mapping`

---

## 项目结构

```
src/geoagent/
├── __init__.py            # 包导出：GeoAgent / GISCompiler / 各 Executor / GeoEngine
├── core.py                # GeoAgent 类（LLM 编译器门面）
├── workflow.py            # SimpleCompilerWorkflow（统计跟踪封装）
├── version.py             # __version__ = "0.2.0"
├── cli.py                 # CLI 入口：geoagent 命令
├── streamlit_app.py       # Streamlit 启动器
├── system_prompts.py      # GIS 专家系统提示词（含七大矩阵）
│
├── layers/                # ═══ 七层架构 ═══
│   ├── architecture.py    # ScenarioType / EngineType / PipelineStatus / MVP 映射
│   ├── layer1_input.py    # UserInput · InputValidator · InputParser
│   ├── layer2_intent.py   # IntentClassifier（关键词分类，无 LLM）
│   ├── layer3_orchestrate.py  # ScenarioOrchestrator · ParameterExtractor · ClarificationEngine
│   ├── reasoner.py        # GeoAgentReasoner（LLM NL→GeoDSL 翻译，支持 DeepSeek）
│   ├── layer4_dsl.py      # GeoDSL · SchemaValidator · DSLBuilder
│   ├── layer5_executor.py # TaskRouter · execute_task() · ExecutorResult
│   └── layer6_render.py   # ResultRenderer · ExplanationCard · BusinessConclusion
│
├── compiler/              # 三层编译器（LLM 集成版）
│   ├── compiler.py        # GISCompiler（LLM + Orchestrator + Executor 流水线）
│   ├── intent_classifier.py   # IntentClassifier（同 layers/layer2）
│   ├── orchestrator.py        # ScenarioOrchestrator（同 layers/layer3）
│   ├── task_executor.py       # execute_task() 封装（同 layers/layer5）
│   └── task_schema.py         # 任务 Pydantic 模型
│
├── executors/             # ═══ 十大执行器 ═══
│   ├── base.py           # BaseExecutor 基类 · ExecutorResult
│   ├── router.py         # TaskRouter · SCENARIO_EXECUTOR_KEY · execute_task()
│   ├── route_executor.py # 路径规划（高德地图 + OSMnx）
│   ├── buffer_executor.py    # 缓冲区分析（GeoPandas + ArcPy）
│   ├── overlay_executor.py   # 叠加分析（GeoPandas + ArcPy）
│   ├── idw_executor.py       # 空间插值（IDW / Kriging / Nearest）
│   ├── ndvi_executor.py      # NDVI 计算（rasterio + ArcPy）
│   ├── shadow_executor.py     # 阴影分析（3D Shapely + ArcGIS 3D）
│   ├── hotspot_executor.py    # 热点分析（PySAL + ArcGIS）
│   ├── viz_executor.py       # 可视化（Folium + PyDeck + Matplotlib）
│   ├── postgis_executor.py   # PostGIS 数据库（psycopg2 + SQLAlchemy）
│   ├── general_executor.py   # 通用任务（py_repl 沙盒 + advanced_tools）
│   └── scenario.py           # ScenarioConfig · SCENARIO_CONFIGS
│
├── plugins/               # 外部 API 插件
│   ├── base.py           # BasePlugin 抽象基类
│   ├── amap_plugin.py    # 高德地图插件（地理编码 / POI / 路径 / 天气）
│   └── osm_plugin.py     # OSMnx 插件（地理编码 / POI / 网络 / 路由）
│
├── geo_engine/           # GeoEngine 统一执行系统（遗留）
│   ├── geo_engine.py    # GeoEngine 主类
│   ├── router.py        # 任务路由器
│   ├── vector_engine.py # 矢量引擎
│   ├── raster_engine.py # 栅格引擎
│   ├── network_engine.py    # 网络引擎
│   ├── analysis_engine.py   # 分析引擎
│   ├── io_engine.py         # I/O 引擎
│   ├── executor.py          # 执行器
│   └── data_utils.py        # 数据工具
│
├── gis_tools/            # GIS 工具箱
│   ├── geo_toolbox.py    # GeoToolbox（矢量 / 栅格 / 网络 / 统计 / 可视化 / LiDAR / 云端）
│   ├── fixed_tools.py    # 固化工具（get_data_info / list_workspace_files）
│   ├── gis_task_tools.py # GIS 任务工具
│   ├── raster_ops.py     # 栅格操作
│   └── advanced_tools.py # 高级工具（STAC / MCDA / 可达性 / 热点 / 3D 渲染）
│
├── renderer/
│   └── result_renderer.py    # 结果渲染器（重导出自 layers/layer6_render）
│
├── knowledge/            # RAG 知识库
│   └── knowledge_rag.py
│
├── tools/                # 工具注册与 RAG
│   ├── registry.py      # 工具注册表
│   ├── arcgis_tools.py  # ArcGIS 工具
│   └── tool_rag.py       # 工具 RAG
│
├── sandbox/             # Python 沙盒（代码执行环境）
│   ├── protocol.py
│   ├── server.py
│   └── client.py
│
└── py_repl.py           # Python REPL（沙盒执行器）
```

---

## 配置说明

### 环境变量

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API 密钥（自然语言理解用） |
| `AMAP_API_KEY` | 否 | 高德地图 API 密钥（国内路径规划用） |

### 依赖安装组

| 组名 | 说明 | 包含功能 |
|------|------|---------|
| `core` | 最小安装 | Streamlit 前端 · OpenAI · 请求库 |
| `gis` | GIS 核心 | GeoPandas · Shapely · Rasterio · Whitebox |
| `viz` | 可视化 | Folium · Leafmap · Matplotlib · Plotly |
| `arcgis` | ArcGIS | ArcGIS Python API（在线数据） |
| `osm` | 海外分析 | OSMnx · NetworkX |
| `data` | 数据处理 | NumPy · Pandas · SciPy |
| `langchain` | Agent 集成 | LangChain · LangGraph |
| `rag` | 检索增强 | FAISS · PyPDF · Tiktoken |
| `all` | 所有功能（推荐） | 以上所有 |
| `full` | 完整依赖 | 含 ArcGIS / LangChain / RAG / 遥感 |

---

## 常见问题

**Q: 为什么意图分类不使用 LLM？**

A: 关键词字典速度快（零延迟、零 token 消耗）、结果稳定（无随机性）、可精确测试。GIS 分析场景的意图类型有限且明确，关键词匹配完全满足需求。

**Q: Reasoner 和 Orchestrator 的区别是什么？**

A: `Orchestrator` 使用正则表达式提取参数，适合简单任务；`Reasoner` 使用 LLM 翻译，适合复杂多步骤任务。当用户需求包含"定位 + 下载数据 + 自定义计算"等多步骤时，Reasoner 可以生成包含 `steps` 数组的工作流 JSON。

**Q: 如何添加新的分析场景？**

A: 三步完成：
1. 在 `layers/architecture.py` 的 `ScenarioType` 枚举中添加新场景
2. 在 `layers/layer2_intent.py` 的 `KEYWORD_MAP` 中添加关键词映射
3. 创建新的 Executor（如 `new_scenario_executor.py`），在 `executors/router.py` 的 `SCENARIO_EXECUTOR_MAP` 中注册

**Q: 是否支持离线使用？**

A: 是的。大多数 Executor（Buffer、Overlay、IDW、NDVI、Shadow、Hotspot）使用纯 Python 开源库（GeoPandas、Shapely、rasterio、PySAL），完全离线可用。仅路径规划（需要高德/OSM API）和 PostGIS（需要数据库连接）需要网络。

**Q: 如何输出非交互式地图？**

A: `VizExecutor` 支持 Folium（交互式 HTML）、PyDeck（WebGL 3D）、Matplotlib（静态图片）三种渲染模式，自动根据数据量选择。
