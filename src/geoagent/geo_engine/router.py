"""
Task DSL Router - 统一任务分发器
================================
将 Task DSL 映射到对应的 Engine。

核心映射表（12类分析）：
    route        → network
    proximity    → vector
    overlay      → vector
    vector       → vector
    surface      → analysis
    raster       → raster
    terrain      → raster
    network      → network
    analysis     → analysis
    geocode      → io
    hotspot      → analysis
    idw          → analysis
"""

from __future__ import annotations

from typing import Dict, Literal


# =============================================================================
# 核心映射表：task → engine
# =============================================================================

ENGINE_MAP: Dict[str, str] = {
    # ── 路径分析 ──────────────────────────────────────────────────────
    "route": "network",       # 路径规划 → 路网引擎
    "shortest_path": "network",
    "shortest route": "network",

    # ── 通用矢量任务 ────────────────────────────────────────────────────
    "vector": "vector",        # 通用矢量任务 → 矢量引擎
    "spatial join": "vector",
    "spatial_join": "vector",
    "sjoin": "vector",

    # ── 邻近分析 ──────────────────────────────────────────────────────
    "proximity": "vector",    # 缓冲区 → 矢量引擎
    "buffer": "vector",
    "buffer zone": "vector",
    "buffering": "vector",
    "within": "vector",

    # ── OSM 在线下载 ─────────────────────────────────────────────────
    "fetch_osm": "vector",    # OSM 在线下载 → 矢量引擎（返回 GeoDataFrame）
    "around": "vector",
    "nearby": "vector",

    # ── 空间叠置 ──────────────────────────────────────────────────────
    "overlay": "vector",       # 空间叠置 → 矢量引擎
    "intersect": "vector",
    "intersection": "vector",
    "union": "vector",
    "difference": "vector",
    "symmetric_difference": "vector",
    "clip": "vector",
    "spatial join": "vector",
    "spatial_join": "vector",
    "sjoin": "vector",

    # ── 插值 ─────────────────────────────────────────────────────────
    "surface": "analysis",    # 空间插值 → 分析引擎
    "interpolation": "analysis",
    "idw": "analysis",
    "kriging": "analysis",
    "inverse distance": "analysis",
    "spatial interpolation": "analysis",

    # ── 栅格分析 ─────────────────────────────────────────────────────
    "raster": "raster",        # 栅格操作 → 栅格引擎
    "clip raster": "raster",
    "mask raster": "raster",
    "reproject raster": "raster",
    "resample": "raster",
    "calculator": "raster",
    "band math": "raster",
    "map algebra": "raster",

    # ── 地形分析 ─────────────────────────────────────────────────────
    "terrain": "raster",       # 地形分析 → 栅格引擎
    "viewshed": "raster",
    "visibility": "raster",
    "slope": "raster",
    "aspect": "raster",
    "dem": "raster",
    "hillshade": "raster",

    # ── 路网分析 ──────────────────────────────────────────────────────
    "network": "network",      # 路网分析 → 路网引擎
    "isochrone": "network",
    "reachable": "network",
    "accessibility": "network",
    "service area": "network",
    "drive time": "network",

    # ── 空间统计 ──────────────────────────────────────────────────────
    "analysis": "analysis",    # 空间统计 → 分析引擎
    "hotspot": "analysis",
    "kde": "analysis",
    "kernel density": "analysis",
    "morans i": "analysis",
    "spatial autocorrelation": "analysis",
    "lisa": "analysis",
    "cluster": "analysis",
    "getis": "analysis",

    # ── 地理编码 ──────────────────────────────────────────────────────
    "geocode": "io",           # 地理编码 → IO 引擎
    "reverse geocode": "io",
    "stac": "io",
    "search satellite": "io",
    "cloud": "io",

    # ── 可视化 ─────────────────────────────────────────────────────────
    "visualization": "vector",  # 可视化（底层用矢量引擎）
    "visualise": "vector",
    "choropleth": "vector",
    "heatmap": "vector",

    # ── 植被指数 ──────────────────────────────────────────────────────
    "ndvi": "raster",          # NDVI → 栅格引擎
    "ndwi": "raster",
    "evi": "raster",
    "vegetation index": "raster",
    "remote sensing": "raster",

    # ── 通用任务 ──────────────────────────────────────────────────────
    "general": "general",
    "unknown": "general",
}


# =============================================================================
# Engine 名称常量
# =============================================================================

class EngineName:
    """Engine 名称枚举"""
    VECTOR = "vector"
    RASTER = "raster"
    NETWORK = "network"
    ANALYSIS = "analysis"
    IO = "io"
    GENERAL = "general"


# =============================================================================
# 路由函数
# =============================================================================

def route_task(task: Dict[str, str]) -> str:
    """
    根据 task DSL 中的 task 字段路由到对应 Engine

    Args:
        task: Task DSL 字典，至少包含 "task" 字段

    Returns:
        Engine 名称字符串 ("vector" | "raster" | "network" | "analysis" | "io" | "general")

    示例：
        task = {"task": "route", "type": "shortest_path", "inputs": {...}}
        engine_name = route_task(task)  # 返回 "network"
    """
    task_type = task.get("task", "unknown").lower().strip()
    engine_name = ENGINE_MAP.get(task_type, EngineName.GENERAL)
    return engine_name


def route_task_by_name(task_name: str) -> str:
    """
    根据 task 名称路由到对应 Engine

    Args:
        task_name: task 类型名称

    Returns:
        Engine 名称字符串
    """
    return ENGINE_MAP.get(task_name.lower().strip(), EngineName.GENERAL)


# =============================================================================
# 验证函数
# =============================================================================

def validate_task_structure(task: Dict) -> tuple[bool, str]:
    """
    验证 Task DSL 结构是否符合规范

    Args:
        task: Task DSL 字典

    Returns:
        (是否有效, 错误消息)
    """
    if not isinstance(task, dict):
        return False, "task 必须是字典类型"

    if "task" not in task:
        return False, "task 字典必须包含 'task' 字段"

    task_type = task.get("task", "")
    if not task_type:
        return False, "'task' 字段不能为空"

    if task_type not in ENGINE_MAP:
        return False, f"不支持的 task 类型: {task_type}，可选值: {list(ENGINE_MAP.keys())}"

    # 验证 inputs 字段（如果存在）
    if "inputs" in task and not isinstance(task["inputs"], dict):
        return False, "'inputs' 字段必须是字典类型"

    # 验证 params 字段（如果存在）
    if "params" in task and not isinstance(task["params"], dict):
        return False, "'params' 字段必须是字典类型"

    return True, ""


# =============================================================================
# 标准 Task DSL 示例
# =============================================================================

TASK_EXAMPLES = {
    "route": {
        "task": "route",
        "type": "shortest_path",
        "inputs": {"start": "芜湖南站", "end": "方特欢乐世界"},
        "params": {"mode": "walking", "city": "芜湖"},
        "outputs": {"file": "route.shp"},
    },
    "buffer": {
        "task": "proximity",
        "type": "buffer",
        "inputs": {"layer": "roads.shp"},
        "params": {"distance": 500, "unit": "meters"},
        "outputs": {"file": "roads_buf.shp"},
    },
    "overlay": {
        "task": "overlay",
        "type": "intersect",
        "inputs": {"layer1": "landuse.shp", "layer2": "flood.shp"},
        "params": {},
        "outputs": {"file": "intersect.shp"},
    },
    "spatial_join": {
        "task": "vector",
        "type": "spatial_join",
        "inputs": {"target": "pois.shp", "join": "districts.shp"},
        "params": {"predicate": "intersects", "how": "left"},
        "outputs": {"file": "joined.shp"},
    },
    "idw": {
        "task": "surface",
        "type": "IDW",
        "inputs": {"points": "stations.shp"},
        "params": {"field": "PM25", "power": 2.0, "cell_size": 0.01},
        "outputs": {"file": "idw.tif"},
    },
    "clip": {
        "task": "raster",
        "type": "clip",
        "inputs": {"raster": "dem.tif", "geometry": "study_area.shp"},
        "params": {},
        "outputs": {"file": "dem_clip.tif"},
    },
    "reproject": {
        "task": "raster",
        "type": "reproject",
        "inputs": {"raster": "dem.tif"},
        "params": {"crs": "EPSG:3857"},
        "outputs": {"file": "dem_3857.tif"},
    },
    "calculator": {
        "task": "raster",
        "type": "calculator",
        "inputs": {"raster1": "S2.tif"},
        "params": {"expression": "(b2-b1)/(b2+b1)"},
        "outputs": {"file": "ndvi.tif"},
    },
    "viewshed": {
        "task": "terrain",
        "type": "viewshed",
        "inputs": {"dem": "dem.tif", "observer": [118.38, 31.33]},
        "params": {"height": 1.7},
        "outputs": {"file": "viewshed.tif"},
    },
    "isochrone": {
        "task": "network",
        "type": "isochrone",
        "inputs": {"center": "北京天安门"},
        "params": {"time": 15, "mode": "walk"},
        "outputs": {"file": "isochrone.shp"},
    },
    "kde": {
        "task": "analysis",
        "type": "kde",
        "inputs": {"points": "pois.shp"},
        "params": {"bandwidth": 1.0, "cell_size": 0.01},
        "outputs": {"file": "kde.tif"},
    },
    "geocode": {
        "task": "geocode",
        "type": "geocode",
        "inputs": {"address": "芜湖南站"},
        "params": {"provider": "nominatim"},
        "outputs": {"file": "station.shp"},
    },
}


# =============================================================================
# 新架构：Task → Executor Layer 映射（优先使用）
# 注意：这是对 executors/router.py SCENARIO_EXECUTOR_KEY 的镜像
# 保留用于 geo_engine 内部路由，最终将被 executors 层统一替代
# =============================================================================

TASK_EXECUTOR_KEY: Dict[str, str] = {
    "route":            "route",
    "buffer":           "buffer",
    "overlay":          "overlay",
    "interpolation":     "interpolation",
    "shadow_analysis":   "shadow_analysis",
    "viewshed":         "viewshed",
    "ndvi":             "ndvi",
    "hotspot":          "hotspot",
    "visualization":     "visualization",
    "accessibility":     "accessibility",
    "suitability":      "suitability",
    "general":          "general",
    # 别名兼容
    "proximity":        "buffer",
    "surface":          "interpolation",
    "spatial join":     "overlay",
    "spatial_join":     "overlay",
    # OSM 在线下载
    "fetch_osm":       "fetch_osm",
}


# =============================================================================
# 新架构便捷方法（委托给 executors/router.py）
# =============================================================================

def route_to_executor(task: Dict[str, str]) -> str:
    """
    路由到 Executor Layer（推荐）

    将 task 映射到具体的 Executor key（route, buffer, overlay 等）。
    这是新架构的路由方式，替代旧的 engine-based 路由。
    """
    task_type = task.get("task", "general").lower().strip()
    return TASK_EXECUTOR_KEY.get(task_type, "general")


def route_to_executor_by_name(task_name: str) -> str:
    """根据 task 名称路由到 Executor Layer"""
    return TASK_EXECUTOR_KEY.get(task_name.lower().strip(), "general")


__all__ = [
    "ENGINE_MAP",
    "EngineName",
    "route_task",
    "route_task_by_name",
    "validate_task_structure",
    "TASK_EXAMPLES",
    # 新架构
    "TASK_EXECUTOR_KEY",
    "route_to_executor",
    "route_to_executor_by_name",
]
