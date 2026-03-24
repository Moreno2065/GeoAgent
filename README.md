# GeoAgent — 空间智能分析 Agent 系统

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Version](https://img.shields.io/badge/Version-2.0.0-2ecc71)
![License](https://img.shields.io/badge/License-MIT-orange.svg)

**GeoAgent** 是一个基于 LLM 的空间智能 GIS 分析 Agent 系统。用户通过自然语言（中英文皆可）描述分析需求，系统将其转化为结构化任务并调用确定性后端引擎执行。覆盖路径规划、缓冲区分析、叠加分析、插值计算、阴影分析、NDVI 计算、热点分析、三维分析、遥感智能等核心场景。

> 核心理念：**LLM 仅负责翻译（自然语言 → GeoDSL），后端代码负责路由和执行，无 ReAct 推理循环**。

---

## 目录

- [项目架构](#项目架构)
- [快速开始](#快速开始)
- [六层架构详解](#六层架构详解)
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

GeoAgent 采用**六层收敛（Six-Layer Convergence）**架构，确保分析过程可预测、可测试、可解释：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户自然语言输入                            │
│         "从芜湖南站到方特欢乐世界的步行路线"                    │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 · 输入层   InputParser                              │
│  输入验证 · 标准化 · 多模态路由（文字/语音/地图点击/文件）     │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 · 意图层   IntentClassifier（无 LLM）              │
│  关键词匹配 → 30+ 场景分类                                   │
│  覆盖：矢量/栅格/遥感/三维/网络/统计                          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 · 编排层   ScenarioOrchestrator                    │
│  正则提取参数 → 缺失时主动澄清 → 参数校验                      │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 · DSL 层   DSLBuilder / SchemaValidator            │
│  GeoDSL 结构化协议（Pydantic 模型验证）                       │
│  scenario / inputs / parameters / outputs / steps            │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 5 · 执行层   TaskRouter + Executors                  │
│  确定性执行：矢量/栅格/遥感/三维/网络/统计                     │
│  自动引擎回退：GeoPandas → ArcPy，Amap → OSMnx               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 6 · 渲染层   ResultRenderer                           │
│  业务友好输出：ExplanationCard + BusinessConclusion          │
│  防幻觉验证：输出文件真实性检查                                │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 说明 |
|------|------|
| **意图分类无 LLM** | Layer 2 使用中英文关键词字典，零 token 消耗，响应稳定 |
| **LLM 仅做翻译** | NL → GeoDSL 翻译，支持 DeepSeek 模型 |
| **无 ReAct 循环** | LLM 仅做 NL → GeoDSL 翻译，后端代码负责路由，不做工具选择 |
| **多步骤工作流** | 支持 `steps` 数组的复杂任务编排 |
| **参数澄清机制** | 缺失必填参数时主动询问用户，而非猜测 |
| **自动引擎回退** | 每个 Executor 优先使用开源引擎（GeoPandas 等），自动回退到商业引擎（ArcPy） |
| **防幻觉验证** | 严格验证输出文件真实性，防止捏造数据 |
| **代码沙盒** | 受限 Python 执行环境，复杂自定义任务安全执行 |
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
python -c "from geoagent import GeoAgent; print(GeoAgent().run('从芜湖南站到方特欢乐世界的步行路线').to_user_text())"
```

---

## 六层架构详解

### Layer 1 — 输入层 (`layers/layer1_input.py`)

接收并标准化用户输入：

- **文本输入**：自然语言查询（中英文）
- **文件上传**：CSV、GeoJSON、Shapefile、TIFF 等
- **多模态支持**：文本、文件、上下文信息
- **输入验证**：长度限制、危险字符过滤

```python
from geoagent.layers.layer1_input import InputParser

parser = InputParser()
user_input = parser.parse_text("从芜湖南站到方特欢乐世界的步行路线")
# → UserInput(source='text', raw="...", lang='zh', validated=True)

# 带文件输入
user_input = parser.parse_file_with_content(
    "分析这个数据",
    file_paths=["/path/to/data.csv"],
    session_id="session-123"
)
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
| `STATISTICS` / `HOTSPOT` | 统计、汇总、热点、聚类 | statistics, hotspot, cluster, Gi* |
| `RASTER` / `NDVI` | 栅格、NDVI、遥感指数、植被、水体 | raster, NDVI, vegetation, water |
| `VISUALIZATION` | 可视化、地图、渲染、显示 | visualization, map, render |
| `GEOCODE` | 地理编码、地址转坐标 | geocode, address to coordinates |
| `MULTI_CRITERIA_SEARCH` | 多条件、附近、距离 | multi-criteria, nearby |

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
result = orchestrator.orchestrate(
    text="以合肥南站为中心，半径1公里范围内有哪些地铁站？",
    context=None,
    intent_result=intent_result,
)
# → OrchestrationResult(status=PIPELINE_STATUS.ORCHESTRATED, params={...})
#   若参数缺失 → status=PIPELINE_STATUS.NEED_CLARIFICATION, questions=[...])
```

### Layer 4 — DSL 层 (`layers/layer4_dsl.py`)

将编排结果构建为 `GeoDSL` 结构化协议（Pydantic 模型）：

```python
from geoagent.layers.layer4_dsl import DSLBuilder, SchemaValidator

builder = DSLBuilder()
dsl = builder.build_from_orchestration(orchestration_result)
# → GeoDSL(version="1.0", scenario=Scenario.ROUTE,
#          task="route", inputs={...}, parameters={...}, outputs={...})

validator = SchemaValidator()
validator.validate(dsl)  # 每个场景有独立的必填字段校验规则
```

### Layer 5 — 执行层 (`layers/layer5_executor.py`)

确定性任务路由和执行，无任何 LLM 推理：

```python
from geoagent.executors.router import execute_task

result = execute_task(dsl.model_dump())
# → ExecutorResult(success=True, data={route_geojson, distance, duration})
```

### Layer 6 — 渲染层 (`layers/layer6_render.py`)

将技术执行结果转换为业务友好输出：

```python
from geoagent.layers.layer6_render import ResultRenderer

renderer = ResultRenderer()
render_result = renderer.render(executor_result)
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
result = agent.run("从芜湖南站到方特欢乐世界的步行路线")
print(result.to_user_text())
# → "从芜湖南站到方特欢乐世界的步行路线，全长约2.3公里，预计步行28分钟"
```

国内地址自动调用高德地图 API；英文/海外地址自动切换到 OSMnx + NetworkX。

### 示例 2：缓冲区分析

```python
result = agent.run("以合肥南站为中心，半径1公里范围内有哪些地铁站？")
print(result.to_user_text())  # 自然语言输出
print(result.metrics)         # 结构化指标
```

### 示例 3：插值分析（缺失参数澄清）

```python
# 缺失分辨率参数 → 系统主动询问
result = agent.run("根据CSV中的散点数据进行空间插值")
# → {"status": "NEED_CLARIFICATION", "clarification_questions": [
#      {"field": "resolution", "question": "请指定插值网格分辨率（米）",
#       "options": ["100m", "200m", "500m"]}
#   ]}
```

### 示例 4：NDVI 计算

```python
result = agent.run("计算 landsat8 数据中2024年6月的NDVI并导出GeoTIFF")
```

支持 Sentinel-2、Landsat-8/9 自动波段映射；也支持自定义波段表达式。

### 示例 5：热点分析

```python
result = agent.run("对城市POI数据进行Getis-Ord Gi*热点分析，权重矩阵用K近邻")
```

使用 PySAL（esda + libpysal）本地计算，输出 HH/LL/HL/LH 四象限分类。

### 示例 6：阴影分析

```python
result = agent.run("计算2024年6月21日夏至中午12:00时建筑物的阴影范围")
```

自动从数据中检测高度字段，使用 Shapely 3D 几何计算太阳位置和阴影投射。

### 示例 7：遥感变化检测

```python
result = agent.run("对比2019年和2024年的影像，检测土地利用变化区域")
```

支持 STAC 搜索、云端 COG 读取、30+ 遥感指数计算。

### 示例 8：代码沙盒（复杂自定义任务）

```python
result = agent.run("用Python计算这些点的 convex hull 并可视化")
```

受限 Python 执行环境，安全处理复杂自定义分析需求。

---

## 核心功能

### 十大执行器

#### Vector 矢量分析域

| 执行器 | 场景类型 | 主引擎 | 备用引擎 | 特色功能 |
|--------|---------|--------|---------|---------|
| `RouteExecutor` | 路径规划 | 高德地图 API | OSMnx + NetworkX | 国内外自动路由；步行/驾车/公交/骑行 |
| `BufferExecutor` | 缓冲区分析 | GeoPandas + Shapely | ArcPy | CRS 自动检测；支持 dissolve |
| `OverlayExecutor` | 叠加分析 | GeoPandas | ArcPy | intersect/union/clip/difference |
| `IDWExecutor` | 空间插值 | NumPy/SciPy | GDAL gdal_grid, PyKrige | IDW（可调幂次）/克里金/最近邻 |
| `HotspotExecutor` | 热点分析 | PySAL | ArcGIS | Gi*/Moran's I/LISA；Queen/Rook/KNN 权重 |
| `SuitabilityExecutor` | 适宜性分析 | NumPy | ArcGIS | MCDA 多准则决策分析 |

#### Terrain 地形分析域

| 执行器 | 场景类型 | 主引擎 | 备用引擎 | 特色功能 |
|--------|---------|--------|---------|---------|
| `ShadowExecutor` | 阴影分析 | Shapely 3D | ArcGIS 3D Analyst | 自动高度检测；任意时间/地点 |
| `LiDAR3DExecutor` | 三维分析 | Shapely 3D | ArcGIS 3D Analyst | 视域/体积/流域/填挖方 |

#### Web 网络服务域

| 执行器 | 场景类型 | 主引擎 | 备用引擎 | 特色功能 |
|--------|---------|--------|---------|---------|
| `AmapExecutor` | 高德服务 | 高德地图 API | - | 地理编码/POI搜索/路径规划/天气 |
| `OverpassExecutor` | Overpass API | requests | - | OSM 数据直接查询 |
| `OSMExecutor` | OSMnx | osmnx | - | 路网下载/最短路径/可达范围 |
| `STACSearchExecutor` | STAC 搜索 | pystac-client | Planetary Computer | 云端遥感数据搜索 |

#### Remote 遥感分析域

| 执行器 | 场景类型 | 主引擎 | 备用引擎 | 特色功能 |
|--------|---------|--------|---------|---------|
| `NdviExecutor` | 植被指数 | rasterio + NumPy | ArcPy | 30+ 遥感指数；云端 COG 直接读取 |
| `RemoteSensingExecutor` | 遥感分析 | rasterio | ArcPy | 变化检测/波段组合/云掩膜 |

#### Viz 可视化域

| 执行器 | 场景类型 | 主引擎 | 备用引擎 | 特色功能 |
|--------|---------|--------|---------|---------|
| `VisualizationExecutor` | 可视化 | Folium | PyDeck, Matplotlib | 多图层地图；PyDeck 3D 可视化 |

#### Core 核心域

| 执行器 | 场景类型 | 主引擎 | 备用引擎 | 特色功能 |
|--------|---------|--------|---------|---------|
| `GDALExecutor` | GDAL 工具 | GDAL CLI | - | 100+ GIS 工具白名单 |
| `PostGISExecutor` | 数据库 | psycopg2 | SQLAlchemy | 空间查询；读写 PostGIS |
| `CodeSandboxExecutor` | 代码沙盒 | py_repl 沙盒 | - | 受限 Python 执行 |
| `ArcGISExecutor` | ArcGIS | arcgis Python API | - | ArcGIS Online 数据 |
| `WorkflowEngine` | 工作流 | 自定义 | - | 多步骤任务编排 |

### 参数澄清机制

当用户输入不足以完成分析时，系统会主动暂停并请求澄清：

```
用户: "帮我做一个分析"
     ↓
系统: "为了完成分析，我需要确认以下几点：
      1. 请选择分析类型：
         ① 路径规划 ② 缓冲区分析 ③ 叠加分析 ④ 插值计算
         或者请直接描述您的需求，例如：'附近500米有哪些设施'"
```

### 防幻觉验证

系统内置严格的输出文件验证机制：

1. 所有执行产生的文件必须真实存在
2. LLM 回复中的文件引用必须通过验证
3. 未经确认的文件路径不会出现在最终输出中

### 黄金规则

| 规则 | 说明 |
|------|------|
| **CRS 铁律** | 任何叠置分析前必须检查坐标系，不一致自动转换 |
| **OOM 防御** | 大 TIFF 必须使用 Window 分块读取 |
| **防幻觉** | 不捏造文件/数据/坐标，所有输出可验证 |

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
`streamlit` · `openai` · `python-dotenv` · `requests` · `ddgs`

### GIS 核心
`geopandas` · `shapely` · `rasterio` · `whitebox` · `fiona` · `pyproj` · `rioxarray` · `xarray`

### 高级 GIS
`osmnx` · `networkx` · `libpysal` · `esda` · `pysal` · `laspy` · `earthpy` · `spyndex`（30+ 遥感指数） · `pystac-client` · `planetary-computer`

### 可视化
`folium` · `leafmap` · `matplotlib` · `plotly` · `streamlit-folium` · `pydeck` · `contextily`

### AI / Agent
`langchain-openai` · `langchain-core` · `langgraph`

### RAG 检索
`faiss-cpu` · `pypdf` · `tiktoken` · `unstructured`

### 可选：商业 GIS
`arcgis` · `arcgis-mapping`

---

## 项目结构

```
src/geoagent/
├── __init__.py            # 包导出：GeoAgent / 执行器 / 架构常量
├── core.py                # GeoAgent 核心类（六层架构入口）
├── version.py             # __version__ = "2.0.0"
├── cli.py                 # CLI 入口：geoagent 命令
├── streamlit_app.py       # Streamlit 启动器
├── system_prompts.py      # GIS 专家系统提示词
├── llm_config.py          # LLM 配置管理
│
├── layers/                # ═══ 六层架构 ═══
│   ├── architecture.py    # Scenario / PipelineStatus / SpatialOperation / Engine
│   ├── layer1_input.py   # UserInput · InputParser
│   ├── layer2_intent.py  # IntentClassifier（关键词分类，无 LLM）
│   ├── layer3_orchestrate.py  # ScenarioOrchestrator · ClarificationEngine
│   ├── layer4_dsl.py     # GeoDSL · SchemaValidator · DSLBuilder
│   └── layer6_render.py  # ResultRenderer · ExplanationCard · BusinessConclusion
│
├── pipeline/              # ═══ 统一流水线 ═══
│   ├── __init__.py       # GeoAgentPipeline · run_pipeline() · run_pipeline_mvp()
│   ├── multi_round.py    # 多轮对话管理
│   ├── step_planner.py   # 步骤解析
│   ├── multi_round_executor.py  # 多轮执行器
│   └── tool_call_validator.py   # 防幻觉验证器
│
├── executors/             # ═══ 执行器层 ═══
│   ├── __init__.py       # 统一导出 + TaskRouter
│   ├── base.py          # BaseExecutor 抽象基类 · ExecutorResult
│   ├── router.py        # TaskRouter · execute_task()
│   ├── scenario.py      # ScenarioConfig · SCENARIO_CONFIGS
│   ├── gdal_engine.py   # GDAL 工具引擎
│   ├── gdal_tool_caller.py  # GDAL 工具调用器
│   ├── gdal_schema.py   # GDAL 任务 Schema 验证
│   │
│   └── domains/         # 按功能域组织
│       ├── vector/      # 矢量分析
│       │   ├── route_executor.py
│       │   ├── buffer_executor.py
│       │   ├── overlay_executor.py
│       │   ├── idw_executor.py
│       │   ├── hotspot_executor.py
│       │   └── suitability_executor.py
│       │
│       ├── terrain/     # 地形分析
│       │   ├── shadow_executor.py
│       │   ├── lidar_3d_executor.py
│       │   └── sun_position.py
│       │
│       ├── web/        # Web 服务
│       │   ├── amap_executor.py
│       │   ├── overpass_executor.py
│       │   ├── osm_executor.py
│       │   └── stac_search_executor.py
│       │
│       ├── remote/     # 遥感分析
│       │   ├── ndvi_executor.py
│       │   └── remote_sensing_executor.py
│       │
│       ├── viz/        # 可视化
│       │   └── viz_executor.py
│       │
│       └── core/       # 核心/通用
│           ├── general_executor.py
│           ├── gdal_executor.py
│           ├── postgis_executor.py
│           ├── code_sandbox_executor.py
│           ├── arcgis_executor.py
│           └── workflow_engine.py
│
├── plugins/              # 外部 API 插件
│   ├── amap_plugin.py   # 高德地图插件
│   └── osm_plugin.py    # OSMnx 插件
│
├── gis_tools/            # GIS 工具箱
│   ├── geotoolbox.py    # GeoToolbox（矢量/栅格/网络/统计/可视化）
│   ├── fixed_tools.py   # 固化工具
│   ├── gis_task_tools.py  # GIS 任务工具
│   └── data_profiler.py # 数据分析器
│
├── sandbox/              # Python 沙盒
│   ├── server.py
│   └── client.py
│
└── py_repl.py           # Python REPL
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
| `search` | 联网搜索 | DuckDuckGo Search |
| `langchain` | Agent 集成 | LangChain · LangGraph |
| `langchain-gis` | Agent + GIS | LangChain + GIS 增强（推荐） |
| `rag` | 检索增强 | FAISS · PyPDF · Tiktoken |
| `all` | 所有功能（推荐） | 以上所有 |
| `full` | 完整依赖 | 含 ArcGIS / LangChain / RAG / 遥感 |

---

## 常见问题

**Q: 为什么意图分类不使用 LLM？**

A: 关键词字典速度快（零延迟、零 token 消耗）、结果稳定（无随机性）、可精确测试。GIS 分析场景的意图类型有限且明确，关键词匹配完全满足需求。

**Q: 如何处理复杂多步骤任务？**

A: GeoAgent 支持两种方式：
1. **Reasoner 模式**：启用 LLM 翻译多步骤工作流，输出 `steps` 数组
2. **WorkflowEngine**：确定性工作流编排，支持步骤依赖管理

**Q: 如何添加新的分析场景？**

A: 四步完成：
1. 在 `layers/architecture.py` 的 `Scenario` 枚举中添加新场景
2. 在 `layers/layer2_intent.py` 的 `KEYWORD_MAP` 中添加关键词映射
3. 创建新的 Executor（如 `executors/domains/vector/my_executor.py`）
4. 在 `executors/router.py` 的 `SCENARIO_EXECUTOR_MAP` 中注册

**Q: 是否支持离线使用？**

A: 是的。大多数 Executor（Buffer、Overlay、IDW、NDVI、Shadow、Hotspot）使用纯 Python 开源库（GeoPandas、Shapely、rasterio、PySAL），完全离线可用。仅路径规划（需要高德/OSM API）和 PostGIS（需要数据库连接）需要网络。

**Q: 如何输出非交互式地图？**

A: `VisualizationExecutor` 支持 Folium（交互式 HTML）、PyDeck（WebGL 3D）、Matplotlib（静态图片）三种渲染模式，自动根据数据量选择。

**Q: 什么是代码沙盒？**

A: `CodeSandboxExecutor` 提供受限的 Python 执行环境，用户可以编写自定义分析代码，系统在隔离环境中执行，避免危险操作。适用于标准场景无法覆盖的复杂需求。

**Q: 如何防止 LLM 生成虚假信息？**

A: GeoAgent 内置三层防幻觉机制：
1. **输入层**：Schema 验证 LLM 输出的 DSL
2. **执行层**：所有文件操作验证路径真实性
3. **输出层**：验证输出文件存在性，拒绝未确认的引用
