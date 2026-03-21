"""
GeoEngine - 统一地理空间分析执行引擎
======================================
为 GeoAgent 提供统一的 GIS 分析执行能力。

架构：
    GeoEngine
    ├── vector_engine   (GeoPandas)
    ├── raster_engine   (Rasterio)
    ├── network_engine  (NetworkX / OSMnx)
    ├── analysis_engine (SciPy / PySAL)
    └── io_engine       (Fiona / STAC)

核心设计原则：
    1. LLM 不直接调用 geopandas，只调用 GeoEngine
    2. 所有模块间统一格式：矢量→GeoDataFrame，栅格→xarray/rasterio，输出→GeoJSON
    3. 每个 task = 一个 executor，无 ReAct 循环
    4. 确定性执行，后端代码路由
"""

from geoagent.geo_engine.geo_engine import GeoEngine, create_geo_engine, get_geo_engine
from geoagent.geo_engine.router import (
    route_task, ENGINE_MAP, TASK_EXECUTOR_KEY, route_to_executor
)
from geoagent.geo_engine.executor import (
    execute_task as geo_execute_task,
    execute_task_via_executor_layer,
)

__all__ = [
    "GeoEngine",
    "create_geo_engine",
    "get_geo_engine",
    "route_task",
    "ENGINE_MAP",
    "TASK_EXECUTOR_KEY",
    "route_to_executor",
    "geo_execute_task",
    "execute_task_via_executor_layer",
]
