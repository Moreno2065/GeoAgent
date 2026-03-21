"""
Executor Layer - 统一执行层
=============================
核心设计原则：
- 所有库（ArcPy, GeoPandas, NetworkX, Amap, PostGIS）都是"被调用者"，不是"决策者"
- 每个 Executor 封装一个特定能力，内部决定用哪个库
- 统一数据格式：输入 → GeoJSON/GeoDataFrame → 输出
- TaskRouter 统一调度，不让库之间互相调用

文件结构：
  executors/
    __init__.py       # 统一导出 + TaskRouter
    base.py           # BaseExecutor 抽象基类
    route_executor.py # 路径分析（Amap + NetworkX）
    buffer_executor.py # 缓冲区分析（ArcPy + GeoPandas）
    overlay_executor.py # 叠置分析（GeoPandas）
    idw_executor.py  # IDW插值（SciPy + ArcPy）
    shadow_executor.py # 阴影分析（3D geometry）
    ndvi_executor.py # NDVI植被指数（rasterio）
    hotspot_executor.py # 热点分析（PySAL）
    viz_executor.py   # 可视化（Folium + PyDeck）
    postgis_executor.py # PostGIS查询（GeoPandas + psycopg2）
    general_executor.py # 通用任务（sandboxed Python）
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

# 各 Executor 类（延迟导入，避免可选依赖未安装时报错）
from geoagent.executors.route_executor import RouteExecutor
from geoagent.executors.buffer_executor import BufferExecutor
from geoagent.executors.overlay_executor import OverlayExecutor
from geoagent.executors.idw_executor import IDWExecutor
from geoagent.executors.shadow_executor import ShadowExecutor
from geoagent.executors.ndvi_executor import NdviExecutor
from geoagent.executors.hotspot_executor import HotspotExecutor
from geoagent.executors.viz_executor import VisualizationExecutor
from geoagent.executors.postgis_executor import PostGISExecutor
from geoagent.executors.general_executor import GeneralExecutor

__all__ = [
    # ---- 基础 ----
    "BaseExecutor",
    "ExecutorResult",
    # ---- 执行器 ----
    "RouteExecutor",
    "BufferExecutor",
    "OverlayExecutor",
    "IDWExecutor",
    "ShadowExecutor",
    "NdviExecutor",
    "HotspotExecutor",
    "VisualizationExecutor",
    "PostGISExecutor",
    "GeneralExecutor",
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
]
