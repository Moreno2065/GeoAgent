"""
Scenario-to-Executor 映射配置
=================================
定义每个 Scenario 对应的 Executor、可用引擎、参数规范。

核心原则：
- Scenario 是 DSL 层对任务的抽象描述
- Executor 是对底层库的统一封装
- 一个 Scenario 可能对应多个可用引擎（engine_routing）
- 所有引擎返回统一格式：ExecutorResult

文件结构映射：
  protocol.py      (DSL)      → scenario 定义
  scenario.py      (配置)     → scenario → executor + engines
  router.py        (调度)     → 路由到对应 executor
  [executor.py]    (实现)     → 调用底层库
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# 引擎选择策略
# =============================================================================

class EngineStrategy(Enum):
    """引擎选择策略"""

    PREFER_LIGHT = "prefer_light"
    PREFER_HEAVY = "prefer_heavy"
    FORCE = "force"
    AUTO = "auto"


# =============================================================================
# 场景配置
# =============================================================================

@dataclass
class ScenarioConfig:
    """
    场景配置：定义一个 scenario 的元信息

    Attributes:
        scenario_name: 场景名称（与 protocol.py SCENARIO_EXECUTOR_MAP 对应）
        executor_key: 对应的 Executor key（与 router.py SCENARIO_EXECUTOR_KEY 对应）
        display_name: 展示名称（中文）
        description: 场景描述
        available_engines: 可用的引擎列表（按优先级排序）
        default_engine: 默认引擎
        engine_routing: 引擎路由规则
        supported_params: 支持的参数列表
        output_format: 输出格式（geojson, dataframe, image 等）
        notes: 备注信息
    """

    scenario_name: str
    executor_key: str
    display_name: str
    description: str
    available_engines: List[str] = field(default_factory=list)
    default_engine: Optional[str] = None
    engine_routing: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    supported_params: List[str] = field(default_factory=list)
    output_format: str = "geojson"
    notes: str = ""

    def get_engine(self, task_params: Dict[str, Any]) -> str:
        """
        根据任务参数确定使用哪个引擎

        策略优先级：
        1. 任务参数中显式指定 engine → 使用指定引擎
        2. 按 engine_routing 规则匹配 → 使用匹配引擎
        3. 使用 default_engine
        4. 使用 available_engines[0]

        Args:
            task_params: 任务参数字典

        Returns:
            引擎名称字符串
        """
        # 1. 显式指定
        if "engine" in task_params:
            return task_params["engine"]

        # 2. 按路由规则匹配
        for rule_key, rule_config in self.engine_routing.items():
            if self._match_rule(rule_key, task_params):
                return rule_config.get("engine", self.default_engine or "")

        # 3. 默认引擎
        return self.default_engine or (self.available_engines[0] if self.available_engines else "")

    def _match_rule(self, rule_key: str, params: Dict[str, Any]) -> bool:
        """判断任务参数是否匹配路由规则"""
        # 规则格式示例："mode in [walking, driving]" 或 "data_type == raster"
        if " " not in rule_key and rule_key in params:
            return bool(params.get(rule_key))
        # 简单相等匹配：field == value
        if " == " in rule_key:
            field, value = rule_key.split(" == ", 1)
            return str(params.get(field, "")) == value.strip()
        # IN 匹配：field in [a, b]
        if " in " in rule_key:
            field_part, values_part = rule_key.split(" in ", 1)
            field = field_part.strip()
            values_str = values_part.strip().strip("[]")
            values = [v.strip().strip("'\"") for v in values_str.split(",")]
            return str(params.get(field, "")) in values
        return False


# =============================================================================
# 全局 Scenario 配置表
# =============================================================================

SCENARIO_CONFIGS: Dict[str, ScenarioConfig] = {

    # -------------------------------------------------------------------------
    # 路径分析 (route)
    # -------------------------------------------------------------------------
    "route": ScenarioConfig(
        scenario_name="route",
        executor_key="route",
        display_name="路径分析",
        description="计算两点或多点之间的最优路径，支持步行、驾车等模式",
        available_engines=["amap", "networkx"],
        default_engine="amap",
        engine_routing={
            "use_local_network == True": {
                "engine": "networkx",
                "description": "使用本地路网计算",
            },
            "mode in [walking, driving]": {
                "engine": "amap",
                "description": "使用高德地图 API 导航",
            },
        },
        supported_params=[
            "start_point", "end_point", "waypoints", "mode",
            "avoid_polygons", "use_local_network", "network_data",
            "engine", "output_format",
        ],
        output_format="geojson",
        notes="【高德限制令】amap 仅用于国内真实导航。几何计算（缓冲区/面积）→ GeoPandas/Shapely → Folium 渲染。",
    ),

    # -------------------------------------------------------------------------
    # 缓冲区分析 (buffer)
    # -------------------------------------------------------------------------
    "buffer": ScenarioConfig(
        scenario_name="buffer",
        executor_key="buffer",
        display_name="缓冲区分析",
        description="对点、线、面要素生成指定距离的缓冲区",
        available_engines=["geopandas", "arcpy"],
        default_engine="geopandas",
        engine_routing={
            "engine == arcpy": {
                "engine": "arcpy",
                "description": "使用 ArcPy 缓冲区分析",
            },
            "engine == geopandas": {
                "engine": "geopandas",
                "description": "使用 GeoPandas 缓冲区分析",
            },
        },
        supported_params=[
            "input_layer", "distance", "unit", "side_type",
            "end_style", "dissolve_option", "engine",
        ],
        output_format="geojson",
        notes="GeoPandas 轻量免费；ArcPy 功能全但部署复杂",
    ),

    # -------------------------------------------------------------------------
    # 叠置分析 (overlay)
    # -------------------------------------------------------------------------
    "overlay": ScenarioConfig(
        scenario_name="overlay",
        executor_key="overlay",
        display_name="叠置分析",
        description="对多个图层进行空间叠置分析（交集、联合、裁剪等）",
        available_engines=["geopandas"],
        default_engine="geopandas",
        engine_routing={
            "engine == geopandas": {
                "engine": "geopandas",
                "description": "使用 GeoPandas overlay",
            },
        },
        supported_params=[
            "input_layer", "overlay_layer", "operation",
            "engine", "output_format",
        ],
        output_format="geojson",
        notes="GeoPandas overlay 足够满足大部分需求",
    ),

    # -------------------------------------------------------------------------
    # 空间插值 (interpolation)
    # -------------------------------------------------------------------------
    "interpolation": ScenarioConfig(
        scenario_name="interpolation",
        executor_key="interpolation",
        display_name="空间插值",
        description="根据离散点数据生成连续表面（IDW、Kriging、最近邻等）",
        available_engines=["scipy", "gdal", "arcpy"],
        default_engine="scipy",
        engine_routing={
            "method == kriging": {
                "engine": "gdal",
                "description": "Kriging 使用 GDAL/PyKrige",
            },
            "engine == arcpy": {
                "engine": "arcpy",
                "description": "使用 ArcPy 地统计分析",
            },
        },
        supported_params=[
            "input_points", "output_extent", "cell_size", "method",
            "power", "search_radius", "engine", "output_format",
        ],
        output_format="raster",
        notes="IDW 推荐 scipy/numpy 自实现；Kriging 使用 GDAL",
    ),

    # -------------------------------------------------------------------------
    # 植被指数 (ndvi)
    # -------------------------------------------------------------------------
    "ndvi": ScenarioConfig(
        scenario_name="ndvi",
        executor_key="ndvi",
        display_name="植被指数计算",
        description="计算 NDVI、NDWI 等遥感植被/水体指数",
        available_engines=["rasterio", "arcpy"],
        default_engine="rasterio",
        engine_routing={
            "engine == arcpy": {
                "engine": "arcpy",
                "description": "使用 ArcPy 计算植被指数",
            },
        },
        supported_params=[
            "input_raster", "red_band", "nir_band", "index_type",
            "engine", "output_format",
        ],
        output_format="raster",
        notes="Rasterio + NumPy 方案轻量易部署",
    ),

    # -------------------------------------------------------------------------
    # 热力分析 (hotspot)
    # -------------------------------------------------------------------------
    "hotspot": ScenarioConfig(
        scenario_name="hotspot",
        executor_key="hotspot",
        display_name="热点分析",
        description="Getis-Ord Gi* 或 Moran's I 空间热点/冷点检测",
        available_engines=["pysal", "arcpy"],
        default_engine="pysal",
        engine_routing={
            "engine == arcpy": {
                "engine": "arcpy",
                "description": "使用 ArcPy 空间统计",
            },
        },
        supported_params=[
            "input_layer", "distance_band", "conceptualization",
            "engine", "output_format",
        ],
        output_format="geojson",
        notes="PySAL 功能全面且免费",
    ),

    # -------------------------------------------------------------------------
    # 阴影分析 (shadow_analysis)
    # -------------------------------------------------------------------------
    "shadow_analysis": ScenarioConfig(
        scenario_name="shadow_analysis",
        executor_key="shadow_analysis",
        display_name="建筑阴影分析",
        description="根据建筑高度和太阳位置计算阴影范围",
        available_engines=["shapely3d", "arcpy"],
        default_engine="shapely3d",
        engine_routing={
            "engine == arcpy": {
                "engine": "arcpy",
                "description": "使用 ArcGIS 3D Analyst",
            },
        },
        supported_params=[
            "buildings_layer", "latitude", "longitude", "date", "time",
            "sun_elevation", "azimuth", "engine", "output_format",
        ],
        output_format="geojson",
        notes="优先使用 Shapely 3D 几何方案",
    ),

    # -------------------------------------------------------------------------
    # 可视化 (visualization)
    # -------------------------------------------------------------------------
    "visualization": ScenarioConfig(
        scenario_name="visualization",
        executor_key="visualization",
        display_name="地图可视化",
        description="交互式地图、3D 视图、热力图、静态图等多种可视化",
        available_engines=["folium", "pydeck", "matplotlib"],
        default_engine="folium",
        engine_routing={
            "viz_type == heatmap": {
                "engine": "folium",
                "description": "热力图使用 Folium",
            },
            "viz_type == 3d": {
                "engine": "pydeck",
                "description": "3D 可视化使用 PyDeck",
            },
            "viz_type == static": {
                "engine": "matplotlib",
                "description": "静态图使用 Matplotlib",
            },
        },
        supported_params=[
            "input_layer", "viz_type", "color_scheme", "height", "width",
            "zoom_level", "center", "engine", "output_format",
        ],
        output_format="html",
        notes="交互优先用 Folium/PyDeck，报告用 Matplotlib",
    ),

    # -------------------------------------------------------------------------
    # PostGIS 查询 (postgis)
    # -------------------------------------------------------------------------
    "postgis": ScenarioConfig(
        scenario_name="postgis",
        executor_key="postgis",
        display_name="PostGIS 空间查询",
        description="对 PostGIS 数据库执行空间查询和数据管理",
        available_engines=["psycopg2"],
        default_engine="psycopg2",
        supported_params=[
            "connection", "table", "geometry_column", "bbox",
            "spatial_filter", "sql", "engine",
        ],
        output_format="geojson",
        notes="使用 psycopg2 + GeoPandas 进行空间查询",
    ),

    # -------------------------------------------------------------------------
    # 通用分析 (general)
    # -------------------------------------------------------------------------
    "general": ScenarioConfig(
        scenario_name="general",
        executor_key="general",
        display_name="通用 GIS 分析",
        description="复杂或多步骤 GIS 分析，支持沙箱 Python 代码执行",
        available_engines=["sandbox"],
        default_engine="sandbox",
        supported_params=[
            "code", "input_layers", "steps", "output_format",
        ],
        output_format="geojson",
        notes="使用 py_repl 沙箱执行任意 Python GIS 代码",
    ),

    # -------------------------------------------------------------------------
    # 空间邻近筛选 (proximity_filter)
    # -------------------------------------------------------------------------
    "proximity_filter": ScenarioConfig(
        scenario_name="proximity_filter",
        executor_key="proximity_filter",
        display_name="空间邻近筛选",
        description="筛选不在指定缓冲区范围内的目标要素（如不在地铁站400米内的星巴克）",
        available_engines=["geopandas"],
        default_engine="geopandas",
        supported_params=[
            "target_layer", "filter_layer", "buffer_distance",
            "buffer_unit", "mode", "output_file",
        ],
        output_format="geojson",
        notes="mode: outside（不在范围内）/ inside（在范围内）",
    ),

    # -------------------------------------------------------------------------
    # POI 搜索 (poi_search / osm_poi)
    # -------------------------------------------------------------------------
    "poi_search": ScenarioConfig(
        scenario_name="poi_search",
        executor_key="osm_poi",
        display_name="POI 搜索",
        description="通过 Overpass API 搜索 OpenStreetMap 上的 POI 数据（星巴克、地铁站等）",
        available_engines=["overpass_api"],
        default_engine="overpass_api",
        supported_params=[
            "poi_types", "center_point", "radius", "bbox",
            "query_type", "tags", "timeout",
        ],
        output_format="geojson",
        notes="无需 API Key，直接连接 OSM 数据库",
    ),

    # -------------------------------------------------------------------------
    # OSM 数据下载 (fetch_osm / overpass)
    # -------------------------------------------------------------------------
    "fetch_osm": ScenarioConfig(
        scenario_name="fetch_osm",
        executor_key="fetch_osm",
        display_name="OSM 数据下载",
        description="从 OpenStreetMap 下载建筑、道路、水体等矢量数据",
        available_engines=["overpass_api"],
        default_engine="overpass_api",
        supported_params=[
            "center_point", "radius", "bbox", "data_type",
            "tags", "timeout", "output_file",
        ],
        output_format="geojson",
        notes="bbox 或 center_point+radius 二选一",
    ),

    # -------------------------------------------------------------------------
    # 高德 Web 服务 (amap)
    # -------------------------------------------------------------------------
    "amap": ScenarioConfig(
        scenario_name="amap",
        executor_key="amap",
        display_name="高德 Web 服务",
        description="高德地图 Web API 服务（POI搜索、地理编码、路径规划等）",
        available_engines=["amap_api"],
        default_engine="amap_api",
        supported_params=[
            "address", "keywords", "city", "types",
            "origin", "destination", "strategy",
        ],
        output_format="json",
        notes="需要高德 API Key",
    ),

    # -------------------------------------------------------------------------
    # GDAL 工具 (gdal)
    # -------------------------------------------------------------------------
    "gdal": ScenarioConfig(
        scenario_name="gdal",
        executor_key="gdal",
        display_name="GDAL 工具",
        description="GDAL/OGR 工具集（裁剪、重投影、格式转换等）",
        available_engines=["gdal"],
        default_engine="gdal",
        supported_params=[
            "operation", "input_file", "output_file",
            "src_crs", "dst_crs", "bbox", "resolution",
        ],
        output_format="auto",
        notes="支持 raster_clip, raster_reproject, vector_buffer 等",
    ),

    # -------------------------------------------------------------------------
    # 代码沙箱 (code_sandbox)
    # -------------------------------------------------------------------------
    "code_sandbox": ScenarioConfig(
        scenario_name="code_sandbox",
        executor_key="code_sandbox",
        display_name="代码沙箱",
        description="在受限环境中执行 Python 代码进行 GIS 分析",
        available_engines=["sandbox"],
        default_engine="sandbox",
        supported_params=[
            "code", "input_layers", "output_variable",
        ],
        output_format="geojson",
        notes="使用 py_repl 沙箱执行",
    ),

    # -------------------------------------------------------------------------
    # 遥感分析 (remote_sensing)
    # -------------------------------------------------------------------------
    "remote_sensing": ScenarioConfig(
        scenario_name="remote_sensing",
        executor_key="remote_sensing",
        display_name="遥感分析",
        description="遥感影像处理（NDWI、云掩膜、波段合成等）",
        available_engines=["rasterio"],
        default_engine="rasterio",
        supported_params=[
            "input_raster", "operation", "bands", "output_file",
        ],
        output_format="raster",
        notes="支持 ndwi, cloud_mask, band_composite 等",
    ),

    # -------------------------------------------------------------------------
    # 三维地形分析 (lidar_3d)
    # -------------------------------------------------------------------------
    "lidar_3d": ScenarioConfig(
        scenario_name="lidar_3d",
        executor_key="lidar_3d",
        display_name="三维地形分析",
        description="基于 DEM/LiDAR 的三维分析（体积计算、坡度坡向、地形起伏等）",
        available_engines=["gdal"],
        default_engine="gdal",
        supported_params=[
            "dem_file", "operation", "output_file",
            "parameters",
        ],
        output_format="raster",
        notes="支持 volume, hillshade, roughness, curvature 等",
    ),

    # -------------------------------------------------------------------------
    # 可达性分析 (accessibility)
    # -------------------------------------------------------------------------
    "accessibility": ScenarioConfig(
        scenario_name="accessibility",
        executor_key="accessibility",
        display_name="可达性分析",
        description="基于路网计算从起点可达的范围（等时圈/等距离圈）",
        available_engines=["amap", "networkx"],
        default_engine="amap",
        supported_params=[
            "center_point", "time_threshold", "distance_threshold",
            "mode", "network_data", "output_file",
        ],
        output_format="geojson",
        notes="使用高德路径规划或本地路网",
    ),

    # -------------------------------------------------------------------------
    # 选址分析 (suitability)
    # -------------------------------------------------------------------------
    "suitability": ScenarioConfig(
        scenario_name="suitability",
        executor_key="suitability",
        display_name="选址分析",
        description="多准则决策分析（MCDA）进行适宜性评价和选址",
        available_engines=["geopandas"],
        default_engine="geopandas",
        supported_params=[
            "criteria_layers", "weights", "method",
            "top_n", "output_file",
        ],
        output_format="geojson",
        notes="使用加权叠加分析",
    ),

    # -------------------------------------------------------------------------
    # STAC 搜索 (stac_search)
    # -------------------------------------------------------------------------
    "stac_search": ScenarioConfig(
        scenario_name="stac_search",
        executor_key="stac_search",
        display_name="STAC 影像搜索",
        description="通过 STAC API 搜索遥感卫星影像",
        available_engines=["stac_api"],
        default_engine="stac_api",
        supported_params=[
            "bbox", "datetime", "collections", "query",
            "output_file",
        ],
        output_format="json",
        notes="支持 Landsat, Sentinel, MODIS 等",
    ),

    # -------------------------------------------------------------------------
    # 视域分析 (viewshed)
    # -------------------------------------------------------------------------
    "viewshed": ScenarioConfig(
        scenario_name="viewshed",
        executor_key="viewshed",
        display_name="视域分析",
        description="基于 DEM 计算观察点的可视范围",
        available_engines=["gdal"],
        default_engine="gdal",
        supported_params=[
            "dem_file", "observer_point", "observer_height",
            "max_distance", "output_file",
        ],
        output_format="raster",
        notes="计算观察点的可视区域",
    ),

    # -------------------------------------------------------------------------
    # 地理编码 (geocode)
    # -------------------------------------------------------------------------
    "geocode": ScenarioConfig(
        scenario_name="geocode",
        executor_key="amap",
        display_name="地理编码",
        description="将地址转换为坐标，或将坐标转换为地址",
        available_engines=["amap_api"],
        default_engine="amap_api",
        supported_params=[
            "address", "city", "coordinates",
        ],
        output_format="json",
        notes="支持正向（地址→坐标）和反向（坐标→地址）",
    ),

    # -------------------------------------------------------------------------
    # 多条件综合搜索 (multi_criteria_search)
    # -------------------------------------------------------------------------
    "multi_criteria_search": ScenarioConfig(
        scenario_name="multi_criteria_search",
        executor_key="multi_criteria_search",
        display_name="多条件综合搜索",
        description="基于多个条件搜索 POI 或地点",
        available_engines=["amap_api"],
        default_engine="amap_api",
        supported_params=[
            "keywords", "types", "city", "offset",
        ],
        output_format="json",
        notes="结合网络搜索和地理编码",
    ),
}


# =============================================================================
# 辅助函数
# =============================================================================

def get_scenario_config(scenario: str) -> Optional[ScenarioConfig]:
    """获取场景配置"""
    return SCENARIO_CONFIGS.get(scenario.lower().strip())


def get_all_scenarios() -> List[str]:
    """获取所有可用场景"""
    return list(SCENARIO_CONFIGS.keys())


def get_executor_key(scenario: str) -> str:
    """从 scenario 名称获取对应的 executor key"""
    config = get_scenario_config(scenario)
    return config.executor_key if config else "general"


def resolve_engine(scenario: str, task_params: Dict[str, Any]) -> str:
    """
    根据 scenario 和任务参数解析引擎

    Args:
        scenario: 场景名称
        task_params: 任务参数字典

    Returns:
        引擎名称（如 "geopandas", "amap", "networkx", "arcpy" 等）
    """
    config = get_scenario_config(scenario)
    if config is None:
        return "unknown"
    return config.get_engine(task_params)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "ScenarioConfig",
    "EngineStrategy",
    "SCENARIO_CONFIGS",
    "get_scenario_config",
    "get_all_scenarios",
    "get_executor_key",
    "resolve_engine",
]
