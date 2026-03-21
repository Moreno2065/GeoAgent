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
    严格限定的 7 类标准 GIS 场景。

    不暴露 ArcToolbox 全家桶，只提供有限的标准场景。
    """
    ROUTE = "route"            # 路径/可达性分析
    BUFFER = "buffer"          # 缓冲/邻近分析
    OVERLAY = "overlay"        # 叠置/裁剪分析
    INTERPOLATION = "interpolation"  # 插值/表面分析
    VIEWSHED = "viewshed"      # 视域/阴影
    STATISTICS = "statistics"  # 统计/聚合
    RASTER = "raster"          # 栅格分析

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


__all__ = [
    "Scenario",
    "RouteSubType",
    "BufferSubType",
    "OverlaySubType",
    "InterpolationSubType",
    "ViewshedSubType",
    "StatisticsSubType",
    "PipelineStatus",
    "Engine",
    "ARCHITECTURE_VERSION",
    "ARCHITECTURE_NAME",
    "MVP_SCENARIOS",
    "MVP_SCENARIO_EXECUTOR_MAP",
]
