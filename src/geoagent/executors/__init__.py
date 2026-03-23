"""
Executor Layer - 统一执行层
=============================
核心设计原则：
- 所有库（ArcPy, GeoPandas, NetworkX, Amap, PostGIS, GDAL）都是"被调用者"，不是"决策者"
- 每个 Executor 封装一个特定能力，内部决定用哪个库
- 统一数据格式：输入 → GeoJSON/GeoDataFrame → 输出
- TaskRouter 统一调度，不让库之间互相调用

文件结构：
  executors/
    __init__.py       # 统一导出 + TaskRouter
    base.py           # BaseExecutor 抽象基类
    router.py         # 任务路由
    scenario.py       # 场景配置
    domains/          # 按功能域组织的执行器
      vector/         # 矢量分析 (route, buffer, overlay, idw, hotspot, suitability)
      terrain/        # 地形分析 (shadow, lidar_3d, sun_position)
      web/            # Web 服务 (amap, overpass, osm, stac)
      remote/         # 遥感分析 (ndvi, remote_sensing)
      viz/            # 可视化 (visualization)
      core/           # 核心/通用 (general, gdal, postgis, sandbox, arcgis)
"""

from geoagent.executors.base import BaseExecutor, ExecutorResult
from geoagent.executors.router import (
    TaskRouter,
    execute_task,
    execute_task_by_dict,
    execute_scenario,
    get_router,
    SCENARIO_EXECUTOR_KEY,
)
from geoagent.executors.scenario import (
    ScenarioConfig,
    SCENARIO_CONFIGS,
    get_scenario_config,
    get_all_scenarios,
    get_executor_key,
    resolve_engine,
)

# 各 Executor 类（从功能域导入）
from geoagent.executors.domains.vector import (
    RouteExecutor,
    BufferExecutor,
    OverlayExecutor,
    IDWExecutor,
    HotspotExecutor,
    SuitabilityExecutor,
)
from geoagent.executors.domains.terrain import (
    ShadowExecutor,
    LiDAR3DExecutor,
    calculate_sun_position,
)
from geoagent.executors.domains.web import (
    AmapExecutor,
    OverpassExecutor,
    OSMExecutor,
    STACSearchExecutor,
)
from geoagent.executors.domains.remote import (
    NdviExecutor,
    RemoteSensingExecutor,
    RemoteSensingIndex,
    BandMapping,
)
from geoagent.executors.domains.viz import VisualizationExecutor
from geoagent.executors.domains.core import (
    GeneralExecutor,
    GDALExecutor,
    PostGISExecutor,
    CodeSandboxExecutor,
    ArcGISExecutor,
    WorkflowEngine,
)

# GDAL 工具相关
from geoagent.executors.gdal_engine import (
    GDAL_TOOL_WHITELIST,
    GDAL_TOOL_DEFINITIONS,
    GDALResult,
    GDALEngine,
    get_gdal_engine,
)
from geoagent.executors.gdal_tool_caller import (
    ToolCallResult,
    GDALToolCaller,
    get_tool_caller,
    call_gdal_tool,
)
from geoagent.executors.gdal_schema import (
    TASK_SCHEMA_MAP,
    GDALSchemaValidator,
    get_schema_validator,
    validate_gdal_task,
)

# 其他执行器（暂未归类）
from geoagent.executors.multi_criteria_executor import MultiCriteriaSearchExecutor
from geoagent.executors.hybrid_retriever_executor import HybridRetrieverExecutor

__all__ = [
    # ---- 基础 ----
    "BaseExecutor",
    "ExecutorResult",
    # ---- Vector Domain ----
    "RouteExecutor",
    "BufferExecutor",
    "OverlayExecutor",
    "IDWExecutor",
    "HotspotExecutor",
    "SuitabilityExecutor",
    # ---- Terrain Domain ----
    "ShadowExecutor",
    "LiDAR3DExecutor",
    "calculate_sun_position",
    # ---- Web Domain ----
    "AmapExecutor",
    "OverpassExecutor",
    "OSMExecutor",
    "STACSearchExecutor",
    # ---- Remote Domain ----
    "NdviExecutor",
    "RemoteSensingExecutor",
    "RemoteSensingIndex",
    "BandMapping",
    # ---- Viz Domain ----
    "VisualizationExecutor",
    # ---- Core Domain ----
    "GeneralExecutor",
    "GDALExecutor",
    "PostGISExecutor",
    "CodeSandboxExecutor",
    "ArcGISExecutor",
    "WorkflowEngine",
    # ---- GDAL 工具 ----
    "GDAL_TOOL_WHITELIST",
    "GDAL_TOOL_DEFINITIONS",
    "GDALResult",
    "GDALEngine",
    "get_gdal_engine",
    "ToolCallResult",
    "GDALToolCaller",
    "get_tool_caller",
    "call_gdal_tool",
    "TASK_SCHEMA_MAP",
    "GDALSchemaValidator",
    "get_schema_validator",
    "validate_gdal_task",
    # ---- 路由 ----
    "TaskRouter",
    "execute_task",
    "execute_task_by_dict",
    "execute_scenario",
    "get_router",
    "SCENARIO_EXECUTOR_KEY",
    # ---- 场景配置 ----
    "ScenarioConfig",
    "SCENARIO_CONFIGS",
    "get_scenario_config",
    "get_all_scenarios",
    "get_executor_key",
    "resolve_engine",
    # ---- 其他 ----
    "MultiCriteriaSearchExecutor",
    "HybridRetrieverExecutor",
]
