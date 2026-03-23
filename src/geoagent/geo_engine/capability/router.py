"""
Capability Router - 能力路由适配器
================================
将 Task DSL 映射到 Capability Registry 的路由层。

核心职责：
1. 将 task type 映射到 capability 名称
2. 通过 CapabilityRegistry 执行能力
3. 统一返回 ExecutorResult 格式

设计原则：
- LLM → task DSL → capability_name → CapabilityRegistry.execute()
- 全部通过注册表，不直接调用函数
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


# =============================================================================
# Task Type → Capability Name 映射表
# =============================================================================

# 标准化映射：task type → capability 名称
TASK_CAPABILITY_MAP: Dict[str, str] = {
    # ── Vector ───────────────────────────────────────────────────────────────
    "buffer": "vector_buffer",
    "vector_buffer": "vector_buffer",
    "dissolve": "vector_dissolve",
    "vector_dissolve": "vector_dissolve",
    "union": "vector_union",
    "vector_union": "vector_union",
    "intersect": "vector_intersect",
    "vector_intersect": "vector_intersect",
    "intersection": "vector_intersect",
    "clip": "vector_clip",
    "vector_clip": "vector_clip",
    "erase": "vector_erase",
    "vector_erase": "vector_erase",
    "split": "vector_split",
    "vector_split": "vector_split",
    "merge": "vector_merge",
    "vector_merge": "vector_merge",
    "simplify": "vector_simplify",
    "vector_simplify": "vector_simplify",
    "reproject": "vector_reproject",
    "vector_reproject": "vector_reproject",
    "centroid": "vector_centroid",
    "vector_centroid": "vector_centroid",
    "convex_hull": "vector_convex_hull",
    "vector_convex_hull": "vector_convex_hull",
    "spatial_join": "vector_spatial_join",
    "vector_spatial_join": "vector_spatial_join",
    "sjoin": "vector_spatial_join",
    "nearest_join": "vector_nearest_join",
    "vector_nearest_join": "vector_nearest_join",
    "calculate_area": "vector_calculate_area",
    "vector_calculate_area": "vector_calculate_area",
    "calculate_length": "vector_calculate_length",
    "vector_calculate_length": "vector_calculate_length",

    # ── Raster ────────────────────────────────────────────────────────────
    "raster_clip": "raster_clip",
    "raster_mask": "raster_mask",
    "raster_merge": "raster_merge",
    "merge_raster": "raster_merge",
    "resample": "raster_resample",
    "raster_resample": "raster_resample",
    "raster_reproject": "raster_reproject",
    "warp": "raster_warp",
    "raster_warp": "raster_warp",
    "calculator": "raster_calculator",
    "raster_calculator": "raster_calculator",
    "band_math": "raster_calculator",
    "slope": "raster_slope",
    "raster_slope": "raster_slope",
    "aspect": "raster_aspect",
    "raster_aspect": "raster_aspect",
    "hillshade": "raster_hillshade",
    "raster_hillshade": "raster_hillshade",
    "ndvi": "raster_ndvi",
    "raster_ndvi": "raster_ndvi",
    "zonal": "raster_zonal_stats",
    "zonal_stats": "raster_zonal_stats",
    "raster_zonal_stats": "raster_zonal_stats",
    "contour": "raster_contour",
    "raster_contour": "raster_contour",
    "reclassify": "raster_reclassify",
    "raster_reclassify": "raster_reclassify",
    "fill_nodata": "raster_fill_nodata",
    "raster_fill_nodata": "raster_fill_nodata",

    # ── Network ───────────────────────────────────────────────────────────
    "route": "network_shortest_path",
    "shortest_path": "network_shortest_path",
    "network_shortest_path": "network_shortest_path",
    "routing": "network_shortest_path",
    "k_shortest_paths": "network_k_shortest_paths",
    "network_k_shortest_paths": "network_k_shortest_paths",
    "isochrone": "network_isochrone",
    "network_isochrone": "network_isochrone",
    "reachable_area": "network_service_area",
    "service_area": "network_service_area",
    "network_service_area": "network_service_area",
    "closest_facility": "network_closest_facility",
    "network_closest_facility": "network_closest_facility",
    "location_allocation": "network_location_allocation",
    "network_location_allocation": "network_location_allocation",
    "flow": "network_flow_analysis",
    "network_flow": "network_flow_analysis",
    "network_flow_analysis": "network_flow_analysis",
    "accessibility": "network_accessibility_score",
    "network_accessibility_score": "network_accessibility_score",

    # ── Analysis ────────────────────────────────────────────────────────
    "idw": "analysis_idw",
    "analysis_idw": "analysis_idw",
    "kriging": "analysis_kriging",
    "analysis_kriging": "analysis_kriging",
    "kde": "analysis_kde",
    "analysis_kde": "analysis_kde",
    "kernel_density": "analysis_kde",
    "hotspot": "analysis_hotspot",
    "analysis_hotspot": "analysis_hotspot",
    "cluster": "analysis_cluster_kmeans",
    "kmeans": "analysis_cluster_kmeans",
    "analysis_cluster_kmeans": "analysis_cluster_kmeans",
    "spatial_autocorrelation": "analysis_spatial_autocorrelation",
    "morans_i": "analysis_spatial_autocorrelation",
    "analysis_spatial_autocorrelation": "analysis_spatial_autocorrelation",
    "distance_matrix": "analysis_distance_matrix",
    "analysis_distance_matrix": "analysis_distance_matrix",
    "weighted_overlay": "analysis_weighted_overlay",
    "analysis_weighted_overlay": "analysis_weighted_overlay",
    "suitability": "analysis_weighted_overlay",

    # ── IO ───────────────────────────────────────────────────────────────
    "read_vector": "io_read_vector",
    "io_read_vector": "io_read_vector",
    "read_raster": "io_read_raster",
    "io_read_raster": "io_read_raster",
    "write_vector": "io_write_vector",
    "io_write_vector": "io_write_vector",
    "write_raster": "io_write_raster",
    "io_write_raster": "io_write_raster",
    "geocode": "io_geocode",
    "io_geocode": "io_geocode",
    "reverse_geocode": "io_reverse_geocode",
    "io_reverse_geocode": "io_reverse_geocode",
    "fetch_osm": "io_fetch_osm",
    "io_fetch_osm": "io_fetch_osm",
    "overpass": "io_overpass",
    "io_overpass": "io_overpass",
    "fetch_stac": "io_fetch_stac",
    "io_fetch_stac": "io_fetch_stac",
    "stac": "io_fetch_stac",
}


def _task_to_capability(task_type: str) -> Optional[str]:
    """
    将 task type 映射为 capability 名称

    Args:
        task_type: task 类型字符串

    Returns:
        capability 名称，如果未找到返回 None
    """
    t = task_type.lower().strip()
    return TASK_CAPABILITY_MAP.get(t)


# =============================================================================
# Capability Executor
# =============================================================================

class CapabilityExecutor(BaseExecutor):
    """
    能力执行器

    通过 CapabilityRegistry 执行 GIS 能力的统一执行器。

    路由策略：
    1. 从 task["task"] 获取 task type
    2. 通过 TASK_CAPABILITY_MAP 映射到 capability 名称
    3. 通过 CapabilityRegistry.execute() 执行能力
    4. 统一返回 ExecutorResult 格式

    使用方式：
        executor = CapabilityExecutor()
        result = executor.run({
            "task": "buffer",
            "inputs": {"layer": "roads.shp"},
            "params": {"distance": 500}
        })
    """

    task_type = "capability"
    supported_engines = {"capability"}

    def __init__(self):
        self._registry = None

    @property
    def registry(self):
        """懒加载能力注册表"""
        if self._registry is None:
            from geoagent.geo_engine.capability import get_capability_registry
            self._registry = get_capability_registry()
        return self._registry

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行能力

        Args:
            task: 任务字典，支持以下格式：
                - {"task": "buffer", "inputs": {...}, "params": {...}}
                - {"task_type": "buffer", "inputs": {...}, "params": {...}}
                - {"capability": "vector_buffer", "inputs": {...}, "params": {...}}

        Returns:
            ExecutorResult 统一结果格式
        """
        # 1. 提取 task type
        task_type = task.get("task") or task.get("task_type") or ""
        capability_name = task.get("capability") or _task_to_capability(task_type)

        if not capability_name:
            return ExecutorResult.err(
                task_type="capability",
                error=f"无法识别 task type: '{task_type}'",
                engine="capability",
            )

        # 2. 提取 inputs 和 params
        inputs = task.get("inputs", {})
        params = task.get("params", {})

        # 兼容旧格式：直接从 task 中提取
        if not inputs:
            for key in ["layer", "file", "raster", "dem", "points", "center",
                        "start", "end", "address", "bbox", "layers"]:
                if key in task and key not in params:
                    if key == "layer" or key == "file":
                        inputs["layer"] = task[key]
                    elif key == "raster" or key == "dem":
                        inputs["raster"] = task[key]
                    elif key == "points":
                        inputs["points"] = task[key]
                    elif key == "center":
                        inputs["center"] = task[key]
                    elif key == "start":
                        inputs["start"] = task[key]
                    elif key == "end":
                        inputs["end"] = task[key]
                    elif key == "address":
                        inputs["address"] = task[key]
                    elif key == "bbox":
                        inputs["bbox"] = task[key]
                    elif key == "layers":
                        inputs["layers"] = task[key]

        # 合并剩余的 task 字段到 params
        for key, value in task.items():
            if key not in ["task", "task_type", "capability", "inputs", "params"] and key not in params:
                params[key] = value

        # 3. 检查 capability 是否存在
        if not self.registry.get(capability_name):
            return ExecutorResult.err(
                task_type=task_type,
                error=f"能力 '{capability_name}' 不存在",
                engine="capability",
            )

        # 4. 执行能力
        try:
            result = self.registry.execute(capability_name, inputs, params)

            # 5. 转换为 ExecutorResult
            if result.get("success"):
                return ExecutorResult.ok(
                    task_type=task_type,
                    engine="capability",
                    data=result,
                    meta={
                        "capability": capability_name,
                        "category": self.registry.get_metadata(capability_name).category.value
                        if self.registry.get_metadata(capability_name) else None,
                    },
                )
            else:
                return ExecutorResult.err(
                    task_type=task_type,
                    error=result.get("error", "执行失败"),
                    engine="capability",
                    error_detail=result.get("error_detail"),
                )

        except Exception as e:
            return ExecutorResult.err(
                task_type=task_type,
                error=f"能力执行异常: {str(e)}",
                engine="capability",
            )

    def get_capabilities(self) -> list[str]:
        """获取所有可用的 capability 名称"""
        return self.registry.list_all()

    def list_by_category(self, category: str) -> list[str]:
        """按类别列出能力"""
        return self.registry.list_by_category(category)

    def search_capabilities(self, query: str) -> list[str]:
        """搜索能力"""
        return self.registry.search(query)

    def get_capability_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取能力信息"""
        return self.registry.info(name)


# =============================================================================
# 全局便捷函数
# =============================================================================

_capability_executor: Optional[CapabilityExecutor] = None


def get_capability_executor() -> CapabilityExecutor:
    """获取 Capability Executor 单例"""
    global _capability_executor
    if _capability_executor is None:
        _capability_executor = CapabilityExecutor()
    return _capability_executor


def execute_capability_task(task: Dict[str, Any]) -> ExecutorResult:
    """
    便捷函数：执行 capability 任务

    Args:
        task: 任务字典

    Returns:
        ExecutorResult
    """
    return get_capability_executor().run(task)


def task_to_capability(task_type: str) -> Optional[str]:
    """将 task type 映射到 capability 名称"""
    return _task_to_capability(task_type)


__all__ = [
    "TASK_CAPABILITY_MAP",
    "CapabilityExecutor",
    "get_capability_executor",
    "execute_capability_task",
    "task_to_capability",
]
