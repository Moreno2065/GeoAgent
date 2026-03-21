"""
GeoEngine - 统一地理空间分析执行引擎
====================================
GeoAgent 的核心执行系统。

结构：
    GeoEngine
    ├── vector_engine   (GeoPandas)
    ├── raster_engine   (Rasterio)
    ├── network_engine  (NetworkX / OSMnx)
    ├── analysis_engine (SciPy / PySAL)
    └── io_engine       (Fiona / STAC / Geopy)

核心设计原则：
    1. LLM 不直接调用 geopandas，只调用 GeoEngine
    2. 所有模块间统一格式：矢量→GeoDataFrame，栅格→xarray/rasterio，输出→GeoJSON
    3. 每个 task = 一个 executor，无 ReAct 循环
    4. 确定性执行，后端代码路由

使用方式：
    from geoagent.geo_engine import GeoEngine

    engine = GeoEngine()
    result = engine.execute({
        "task": "route",
        "type": "shortest_path",
        "inputs": {"start": "芜湖南站", "end": "方特"},
        "params": {"mode": "walking", "city": "芜湖"},
    })

    result = engine.execute({
        "task": "proximity",
        "type": "buffer",
        "inputs": {"layer": "roads.shp"},
        "params": {"distance": 500, "unit": "meters"},
    })
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from geoagent.geo_engine.router import (
    ENGINE_MAP, EngineName, route_task,
    validate_task_structure, TASK_EXAMPLES,
    TASK_EXECUTOR_KEY, route_to_executor,
)
from geoagent.geo_engine.executor import (
    execute_task,
    execute_task_via_executor_layer,
)
# 直接从子模块导入，避免循环依赖
from geoagent.geo_engine.vector_engine import VectorEngine
from geoagent.geo_engine.raster_engine import RasterEngine
from geoagent.geo_engine.network_engine import NetworkEngine
from geoagent.geo_engine.analysis_engine import AnalysisEngine
from geoagent.geo_engine.io_engine import IOEngine
from geoagent.geo_engine.data_utils import format_result


# =============================================================================
# GeoEngine 核心类
# =============================================================================

class GeoEngine:
    """
    统一地理空间分析执行引擎

    LLM 唯一的 GIS 执行接口。

    使用方式：
        engine = GeoEngine()

        # 方式 1：直接调用
        result = engine.execute({"task": "route", "type": "shortest_path", ...})

        # 方式 2：通过 sub-engine 调用
        engine.vector.buffer("roads.shp", 500, output_file="buf.shp")
        engine.raster.clip("dem.tif", "mask.shp", "dem_clip.tif")

        # 方式 3：通过 Engine.run(task) 调用
        task = {
            "task": "proximity",
            "type": "buffer",
            "inputs": {"layer": "roads.shp"},
            "params": {"distance": 500},
        }
        result = engine.execute(task)
    """

    def __init__(self):
        """初始化 GeoEngine"""
        self.vector = VectorEngine
        self.raster = RasterEngine
        self.network = NetworkEngine
        self.analysis = AnalysisEngine
        self.io = IOEngine

        self._stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
        }

    # ── 主执行入口 ───────────────────────────────────────────────────

    def execute(self, task: Dict[str, Any]) -> str:
        """
        执行 Task DSL 任务

        Args:
            task: Task DSL 字典

        Returns:
            JSON 格式的执行结果

        示例：
            task = {
                "task": "route",
                "type": "shortest_path",
                "inputs": {"start": "芜湖南站", "end": "方特欢乐世界"},
                "params": {"mode": "walking", "city": "芜湖"},
                "outputs": {"file": "route.shp"},
            }
            result = engine.execute(task)
        """
        self._stats["total"] += 1

        try:
            result = execute_task(task)

            # 检查是否成功
            try:
                result_data = json.loads(result)
                if result_data.get("success", False):
                    self._stats["successful"] += 1
                else:
                    self._stats["failed"] += 1
            except Exception:
                self._stats["successful"] += 1

            return result

        except Exception as e:
            self._stats["failed"] += 1
            return json.dumps({
                "success": False,
                "error": str(e),
            }, ensure_ascii=False, indent=2)

    # ── 统计 ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        """获取执行统计"""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = {"total": 0, "successful": 0, "failed": 0}

    # ── 工具方法 ───────────────────────────────────────────────────

    def get_engine_map(self) -> Dict[str, str]:
        """获取 Engine 映射表"""
        return ENGINE_MAP.copy()

    def get_task_examples(self) -> Dict[str, Dict]:
        """获取 Task DSL 示例"""
        return TASK_EXAMPLES.copy()

    def info(self) -> str:
        """显示 GeoEngine 信息"""
        lines = [
            "GeoEngine - Unified GeoSpatial Analysis Execution Engine",
            "=" * 55,
            "[Engine 1] VectorEngine -- Vector Analysis (GeoPandas)",
            "  .buffer()           Buffer analysis",
            "  .overlay()          Spatial overlay",
            "  .spatial_join()     Spatial join",
            "  .clip()             Vector clip",
            "  .project()          Projection transform",
            "  .dissolve()         Dissolve",
            "  .centroid()         Centroid",
            "  .simplify()         Simplify",
            "  .voronoi()          Voronoi diagram",
            "  .convert_format()   Format conversion",
            "  .geocode()          Geocoding",
            "[Engine 2] RasterEngine -- Raster Analysis (Rasterio)",
            "  .clip()             Raster clip",
            "  .reproject()       Reproject",
            "  .resample()        Resample",
            "  .calculate_index() Band index calculation",
            "  .calculate_spyndex() Remote sensing index (spyndex)",
            "  .slope_aspect()    Slope & aspect",
            "  .zonal_statistics() Zonal statistics",
            "  .reclassify()       Reclassify",
            "  .viewshed()        Viewshed analysis",
            "[Engine 3] NetworkEngine -- Network Analysis (OSMnx)",
            "  .shortest_path()    Shortest path",
            "  .isochrone()        Isochrone",
            "  .reachable_area()  Reachable area",
            "[Engine 4] AnalysisEngine -- Spatial Statistics (SciPy/PySAL)",
            "  .idw()              IDW interpolation",
            "  .kde()              Kernel density estimation",
            "  .hotspot()          Hotspot analysis (Gi*)",
            "  .morans_i()         Global Moran's I",
            "[Engine 5] IOEngine -- Data IO",
            "  .geocode()          Geocoding",
            "  .reverse_geocode()  Reverse geocoding",
            "  .search_stac()      STAC search",
            "  .read_cog_preview() COG preview",
            "=" * 55,
            "[Task DSL Format]",
            "  engine.execute({",
            '      "task": "route",',
            '      "type": "shortest_path",',
            '      "inputs": {"start": "...", "end": "..."},',
            '      "params": {"mode": "walking"},',
            '  })',
            "=" * 55,
        ]
        return "\n".join(lines)


# =============================================================================
# 便捷工厂函数
# =============================================================================

_geo_engine_instance: Optional[GeoEngine] = None


def create_geo_engine() -> GeoEngine:
    """
    创建 GeoEngine 实例（全局单例）

    Returns:
        GeoEngine 实例
    """
    global _geo_engine_instance
    if _geo_engine_instance is None:
        _geo_engine_instance = GeoEngine()
    return _geo_engine_instance


def get_geo_engine() -> GeoEngine:
    """
    获取 GeoEngine 全局实例

    Returns:
        GeoEngine 实例
    """
    return create_geo_engine()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "GeoEngine",
    "create_geo_engine",
    "get_geo_engine",
]
