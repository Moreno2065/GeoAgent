"""
GeoAgent 六层架构定义
====================
严格按照用户架构指导重构的核心架构：

┌─────────────────────────────────────────────────────────────┐
│  第1层：用户输入层（Input Layer）                              │
│  - 文本、语音、文件上传、地图框选、图层点击                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第2层：意图识别层（Intent Classifier）                       │
│  - 判断用户属于哪个大场景（route/buffer/overlay等）              │
│  - 使用关键词匹配，稳定高效                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第3层：场景编排层（Scenario Orchestrator）                   │
│  - 决定请求属于哪类任务                                       │
│  - 需不需要追问                                              │
│  - 走哪个分析模板                                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第4层：任务编译层（Task DSL Builder）                        │
│  - 把自然语言变成结构化 GeoDSL                                 │
│  - Schema 校验                                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第5层：执行引擎层（Task Router + Executors）                │
│  - 确定性执行 ArcPy/GeoPandas/NetworkX/Amap/PostGIS          │
│  - 读 DSL → 查路由表 → 调用固定函数 → 生成结果                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第6层：结果呈现层（Result Renderer）                        │
│  - 地图、图层、表格、结论摘要                                  │
│  - 解释卡片                                                  │
└─────────────────────────────────────────────────────────────┘

核心设计原则：
1. LLM 只做"翻译"：NL → DSL，不做决策
2. 所有执行确定性：后端代码路由，无 ReAct 循环
3. 有限标准场景：route/buffer/overlay/interpolation/viewshed/statistics/raster
4. 参数补全+澄清机制：参数不完整必须追问
5. 结果面向业务结论：不是技术结果，是解释卡片

MVP 阶段只做三种核心场景：
- route（路径/可达性）
- buffer（缓冲/邻近）
- overlay（叠置/选址）
"""

from __future__ import annotations
from enum import Enum


# =============================================================================
# 场景类型枚举（严格限定 7 类标准场景）
# =============================================================================

class Scenario(str, Enum):
    """
    标准 GIS 场景枚举。

    不暴露 ArcToolbox 全家桶，只提供有限的标准场景。
    """
    # ── 核心 GIS 分析场景 ──────────────────────────────────────
    ROUTE = "route"            # 路径/可达性分析
    BUFFER = "buffer"          # 缓冲/邻近分析
    OVERLAY = "overlay"        # 叠置/裁剪分析
    INTERPOLATION = "interpolation"  # 插值/表面分析
    VIEWSHED = "viewshed"      # 视域/阴影
    STATISTICS = "statistics"   # 统计/聚合
    SUITABILITY = "suitability"  # 适宜性分析/选址（MCDA）
    RASTER = "raster"          # 栅格分析
    ACCESSIBILITY = "accessibility"   # 可达性分析
    SHADOW_ANALYSIS = "shadow_analysis"  # 阴影分析
    HOTSPOT = "hotspot"        # 热点分析
    VISUALIZATION = "visualization"  # 可视化

    # ── 🟣 代码沙盒场景（受限代码执行）──────────────────────────────
    CODE_SANDBOX = "code_sandbox"  # 受限代码执行（非标准任务的自定义 Python 代码）

    # ── 🟣 OSM 地图下载场景 ────────────────────────────────────────
    FETCH_OSM = "fetch_osm"  # 从 OpenStreetMap 下载路网/建筑物数据

    # ── 🟢 高德基础 Web 服务场景 ─────────────────────────────────
    GEOCODE = "geocode"        # 地理编码（地址 → 坐标）
    REGEOCODE = "regeocode"     # 逆地理编码（坐标 → 地址）
    DISTRICT = "district"       # 行政区域查询
    STATIC_MAP = "static_map"  # 静态地图
    COORD_CONVERT = "coord_convert"  # 坐标转换
    GRASP_ROAD = "grasp_road"  # 轨迹纠偏

    # ── 🔵 高德高级 Web 服务场景 ─────────────────────────────────
    POI_SEARCH = "poi_search"  # POI 搜索
    INPUT_TIPS = "input_tips"  # 输入提示
    TRAFFIC_STATUS = "traffic_status"  # 交通态势
    TRAFFIC_EVENTS = "traffic_events"   # 交通事件
    TRANSIT_INFO = "transit_info"  # 公交信息
    IP_LOCATION = "ip_location"  # IP 定位
    WEATHER = "weather"         # 天气查询

    # ── 可视化 Pipeline 扩展场景 ─────────────────────────────────
    POI_QUERY = "poi_query"  # POI 查询（Overpass API 封装，标准化 POI 搜索流程）
    HEATMAP = "heatmap"       # 热力图（POI/密度数据 → 热力图渲染）
    CHOROPLETH = "choropleth"  # 分级设色（面数据 + 数值字段 → 分类着色）
    DATA_SOURCE = "data_source"  # 数据源加载（从 OSM/API/文件加载数据）

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, s: str) -> "Scenario":
        """从字符串安全转换"""
        s = s.lower().strip()
        try:
            return cls(s)
        except ValueError:
            return cls.ROUTE  # 默认


# =============================================================================
# 场景子类型
# =============================================================================

class RouteSubType(str, Enum):
    WALKING = "walking"
    DRIVING = "driving"
    CYCLING = "cycling"
    TRANSIT = "transit"
    ISOCHRONE = "isochrone"  # 等时圈
    SERVICE_AREA = "service_area"


class BufferSubType(str, Enum):
    POINT = "point"          # 点缓冲（周边）
    LINE = "line"           # 线缓冲（沿线）
    POLYGON = "polygon"      # 面缓冲


class OverlaySubType(str, Enum):
    INTERSECT = "intersect"    # 交集
    UNION = "union"            # 并集
    CLIP = "clip"             # 裁剪
    DIFFERENCE = "difference"  # 差集


class InterpolationSubType(str, Enum):
    IDW = "idw"               # 反距离加权
    KRIGING = "kriging"       # 克里金
    NEAREST = "nearest"       # 最近邻


class ViewshedSubType(str, Enum):
    VIEWSHED = "viewshed"     # 视域分析
    SHADOW = "shadow"         # 阴影分析


class StatisticsSubType(str, Enum):
    HOTSPOT = "hotspot"       # 热点分析
    DENSITY = "density"       # 密度分析
    ZONAL = "zonal"           # 分区统计
    NDVI = "ndvi"             # 植被指数


# =============================================================================
# 🟢 高德基础服务子类型
# =============================================================================

class GeocodeSubType(str, Enum):
    ADDRESS = "address"       # 标准地址
    BATCH = "batch"          # 批量地址


class RegeocodeSubType(str, Enum):
    BASE = "base"            # 基本地址
    ALL = "all"              # 包含周边 POI


class DistrictLevel(str, Enum):
    COUNTRY = "country"      # 国家
    PROVINCE = "province"     # 省
    CITY = "city"            # 市
    DISTRICT = "district"    # 区/县
    STREET = "street"        # 街道


class CoordinateSystem(str, Enum):
    GPS = "gps"              # WGS84
    BAIDU = "baidu"          # 百度坐标
    MAPBAR = "mapbar"         # 图吧坐标
    AUTO = "auto"            # 自动检测


# =============================================================================
# 🔵 高德高级服务子类型
# =============================================================================

class PoiSortRule(str, Enum):
    DISTANCE = "distance"    # 距离优先
    WEIGHT = "weight"        # 综合权重


class TrafficLevel(str, Enum):
    HIGHWAY = 1              # 高速
    URBAN_EXPRESS = 2        # 城市快速路
    MAIN_ROAD = 3            # 主干道
    SECONDARY_ROAD = 4       # 次干道
    BRANCH_ROAD = 5          # 支路
    COUNTRY_ROAD = 6         # 乡道


class TrafficEventType(str, Enum):
    ALL = 0                  # 所有
    CONSTRUCTION = 1         # 施工
    ACCIDENT = 2             # 事故
    CONTROL = 3               # 管制


class WeatherExtensions(str, Enum):
    BASE = "base"            # 实时天气
    ALL = "all"              # 包含预报


# =============================================================================
# 标准空间操作枚举（与 Scenario 解耦，用于 Transform 层）
# =============================================================================

class SpatialOperation(str, Enum):
    """
    标准空间操作枚举。

    与 Scenario 解耦——一个 Scenario 可以包含多个操作。
    例如：叠置分析 = BUFFER + OVERLAY 两个操作组合。

    用于：
    - WorkflowEngine 的 step.task 字段
    - TransformEngine 的操作路由
    - Executor 的任务类型映射
    """
    BUFFER = "buffer"
    INTERSECT = "intersect"
    UNION = "union"
    CLIP = "clip"
    DIFFERENCE = "difference"
    DISSOLVE = "dissolve"
    ROUTE = "route"
    NEAREST = "nearest"
    HEATMAP = "heatmap"
    INTERPOLATE = "interpolate"
    VIEWSHED = "viewshed"
    SHADOW = "shadow"
    OVERLAY = "overlay"  # 通用叠置（可拆解为 intersect/union/difference）


# =============================================================================
# 执行状态枚举
# =============================================================================

class PipelineStatus(str, Enum):
    """流水线执行状态"""
    PENDING = "pending"                # 待处理
    INPUT_RECEIVED = "input_received"  # 输入已接收
    INTENT_CLASSIFIED = "intent_classified"  # 意图已分类
    ORCHESTRATED = "orchestrated"      # 已编排
    DSL_BUILT = "dsl_built"           # DSL 已构建
    SCHEMA_VALIDATED = "schema_validated"  # Schema 已校验
    ROUTED = "routed"                 # 已路由
    EXECUTING = "executing"            # 执行中
    COMPLETED = "completed"             # 完成
    FAILED = "failed"                  # 失败
    CLARIFICATION_NEEDED = "clarification_needed"  # 需要追问


# =============================================================================
# 架构元信息
# =============================================================================

ARCHITECTURE_VERSION = "2.0"
ARCHITECTURE_NAME = "Six-Layer GIS Agent Architecture"

# MVP 支持的场景
MVP_SCENARIOS = [Scenario.ROUTE, Scenario.BUFFER, Scenario.OVERLAY]

# MVP 执行器映射
MVP_SCENARIO_EXECUTOR_MAP = {
    Scenario.ROUTE: "route_executor",
    Scenario.BUFFER: "buffer_executor",
    Scenario.OVERLAY: "overlay_executor",
    Scenario.INTERPOLATION: "idw_executor",
    Scenario.VIEWSHED: "shadow_executor",
    Scenario.STATISTICS: "hotspot_executor",
    Scenario.RASTER: "ndvi_executor",
    Scenario.SUITABILITY: "suitability_executor",  # 适宜性选址（MCDA）
    Scenario.GEOCODE: "amap_executor",
    Scenario.REGEOCODE: "amap_executor",
    Scenario.DISTRICT: "amap_executor",
    Scenario.STATIC_MAP: "amap_executor",
    Scenario.COORD_CONVERT: "amap_executor",
    Scenario.GRASP_ROAD: "amap_executor",
    Scenario.POI_SEARCH: "amap_executor",
    Scenario.INPUT_TIPS: "amap_executor",
    Scenario.TRAFFIC_STATUS: "amap_executor",
    Scenario.TRAFFIC_EVENTS: "amap_executor",
    Scenario.TRANSIT_INFO: "amap_executor",
    Scenario.IP_LOCATION: "amap_executor",
    Scenario.WEATHER: "amap_executor",
    Scenario.CODE_SANDBOX: "code_sandbox_executor",  # 受限代码执行（补丁层）
}

# 引擎名称
class Engine(str, Enum):
    GEOPANDAS = "geopandas"
    ARCPY = "arcpy"
    NETWORKX = "networkx"
    AMAP = "amap"
    RASTERIO = "rasterio"
    SCIENTIFIC = "scientific"  # SciPy
    PYSAL = "pysal"
    POSTGIS = "postgis"
    SANDBOX = "sandbox"  # 代码沙盒（受限 Python 执行）


__all__ = [
    "Scenario",
    "RouteSubType",
    "BufferSubType",
    "OverlaySubType",
    "InterpolationSubType",
    "ViewshedSubType",
    "StatisticsSubType",
    # 高德基础服务子类型
    "GeocodeSubType",
    "RegeocodeSubType",
    "DistrictLevel",
    "CoordinateSystem",
    # 高德高级服务子类型
    "PoiSortRule",
    "TrafficLevel",
    "TrafficEventType",
    "WeatherExtensions",
    "PipelineStatus",
    "Engine",
    "SpatialOperation",
    "ARCHITECTURE_VERSION",
    "ARCHITECTURE_NAME",
    "MVP_SCENARIOS",
    "MVP_SCENARIO_EXECUTOR_MAP",
]
