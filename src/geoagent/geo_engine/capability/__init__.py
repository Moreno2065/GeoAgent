"""
GIS Capability Registry - 能力注册表核心
=========================================
54 个标准化 GIS 能力节点的统一注册与管理。

架构设计：
┌─────────────────────────────────────────────────────────────┐
│                    CAPABILITY_REGISTRY                       │
│  ┌─────────────┬──────────────┬────────────────────────┐  │
│  │ Vector(15)  │ Raster(15)   │ Network(8)             │  │
│  │             │              │                        │  │
│  │ buffer     │ clip         │ shortest_path          │  │
│  │ dissolve   │ mask         │ k_shortest_paths       │  │
│  │ union      │ merge        │ isochrone              │  │
│  │ intersect  │ resample     │ service_area           │  │
│  │ clip       │ reproject    │ closest_facility       │  │
│  │ erase      │ calculator   │ location_allocation    │  │
│  │ split      │ slope        │ flow_analysis          │  │
│  │ merge      │ aspect       │ accessibility_score    │  │
│  │ simplify   │ hillshade    │                        │  │
│  │ reproject  │ ndvi         ├────────────────────────┤  │
│  │ centroid   │ zonal_stats  │ Spatial Analysis(8)    │  │
│  │ convex_hull│ contour      │                        │  │
│  │ spatial_join│ reclassify  │ idw                    │  │
│  │ nearest_join│ fill_nodata │ kriging                │  │
│  │ calc_area  │ warp         │ kde                    │  │
│  │ calc_length│              │ hotspot                │  │
│  │             │              │ cluster_kmeans         │  │
│  │             │              │ spatial_autocorrelation│  │
│  │             │              │ distance_matrix        │  │
│  │             │              │ weighted_overlay       │  │
│  │             │              ├────────────────────────┤  │
│  │             │              │ IO / Data(8)          │  │
│  │             │              │                        │  │
│  │             │              │ read_vector           │  │
│  │             │              │ read_raster           │  │
│  │             │              │ write_vector          │  │
│  │             │              │ write_raster          │  │
│  │             │              │ geocode               │  │
│  │             │              │ reverse_geocode       │  │
│  │             │              │ fetch_osm             │  │
│  │             │              │ fetch_stac            │  │
│  └─────────────┴──────────────┴────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

使用方式：
    from geoagent.geo_engine.capability import CAPABILITY_REGISTRY, get_capability

    # 获取能力函数
    cap = get_capability("vector_buffer")
    result = cap(
        inputs={"layer": "roads.shp"},
        params={"distance": 500, "unit": "meters"}
    )

    # 列出所有能力
    CAPABILITY_REGISTRY.list_all()

    # 按类别查询
    CAPABILITY_REGISTRY.list_by_category("vector")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import inspect
import traceback


# =============================================================================
# 能力类别枚举
# =============================================================================

class CapabilityCategory(str, Enum):
    """能力类别枚举"""
    VECTOR = "vector"
    RASTER = "raster"
    NETWORK = "network"
    ANALYSIS = "analysis"
    IO = "io"


# =============================================================================
# 能力元数据
# =============================================================================

@dataclass
class CapabilityMeta:
    """能力元数据"""
    name: str
    category: CapabilityCategory
    description: str
    engine: str
    params_schema: Dict[str, Any] = field(default_factory=dict)
    inputs_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


# =============================================================================
# 标准能力签名
# =============================================================================

CapabilityFunc = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


# =============================================================================
# 标准返回格式
# =============================================================================

CAPABILITY_OUTPUT_SCHEMA = {
    "type": "...",
    "data": "...",
    "summary": "...",
    "output_path": "...",
    "metadata": {...},
    "success": True,
    "error": "...",
}


# =============================================================================
# 能力注册表
# =============================================================================

class CapabilityRegistry:
    """
    GIS 能力注册表

    统一管理所有 54 个 GIS 能力节点。

    设计原则：
    1. 统一接口：inputs + params → dict
    2. 无跨函数调用，全部通过 Router
    3. 确定性执行，无 LLM 决策
    4. 标准化返回格式
    """

    def __init__(self):
        self._capabilities: Dict[str, CapabilityFunc] = {}
        self._metadata: Dict[str, CapabilityMeta] = {}
        self._category_index: Dict[str, List[str]] = {
            cat.value: [] for cat in CapabilityCategory
        }
        self._engine_index: Dict[str, List[str]] = {}

    def register(
        self,
        name: str,
        func: CapabilityFunc,
        category: CapabilityCategory,
        description: str,
        engine: str,
        params_schema: Dict[str, Any] = None,
        inputs_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None,
        examples: List[Dict[str, Any]] = None,
        dependencies: List[str] = None,
        tags: List[str] = None,
    ) -> None:
        """
        注册能力

        Args:
            name: 能力名称（唯一标识）
            func: 能力函数
            category: 能力类别
            description: 功能描述
            engine: 底层引擎
            params_schema: 参数模式
            inputs_schema: 输入模式
            output_schema: 输出模式
            examples: 使用示例
            dependencies: 依赖库
            tags: 标签
        """
        self._capabilities[name] = func
        self._metadata[name] = CapabilityMeta(
            name=name,
            category=category,
            description=description,
            engine=engine,
            params_schema=params_schema or {},
            inputs_schema=inputs_schema or {},
            output_schema=output_schema or CAPABILITY_OUTPUT_SCHEMA,
            examples=examples or [],
            dependencies=dependencies or [],
            tags=tags or [],
        )
        self._category_index[category.value].append(name)
        if engine not in self._engine_index:
            self._engine_index[engine] = []
        self._engine_index[engine].append(name)

    def get(self, name: str) -> Optional[CapabilityFunc]:
        """获取能力函数"""
        return self._capabilities.get(name)

    def get_metadata(self, name: str) -> Optional[CapabilityMeta]:
        """获取能力元数据"""
        return self._metadata.get(name)

    def execute(self, name: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行能力

        Args:
            name: 能力名称
            inputs: 输入数据
            params: 参数

        Returns:
            标准执行结果
        """
        func = self._capabilities.get(name)
        if func is None:
            return {
                "success": False,
                "error": f"能力 '{name}' 不存在",
                "type": "error",
                "data": None,
                "summary": "",
            }

        try:
            result = func(inputs, params)
            if not isinstance(result, dict):
                return {
                    "success": False,
                    "error": f"能力 '{name}' 返回格式错误",
                    "type": "error",
                    "data": None,
                    "summary": "",
                }
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_detail": traceback.format_exc(),
                "type": "error",
                "data": None,
                "summary": f"能力 '{name}' 执行失败",
            }

    def list_all(self) -> List[str]:
        """列出所有能力名称"""
        return list(self._capabilities.keys())

    def list_by_category(self, category: str) -> List[str]:
        """按类别列出能力"""
        return self._category_index.get(category, [])

    def list_by_engine(self, engine: str) -> List[str]:
        """按引擎列出能力"""
        return self._engine_index.get(engine, [])

    def get_by_tag(self, tag: str) -> List[str]:
        """按标签查找能力"""
        result = []
        for name, meta in self._metadata.items():
            if tag in meta.tags:
                result.append(name)
        return result

    def search(self, query: str) -> List[str]:
        """模糊搜索能力"""
        query = query.lower()
        result = []
        for name, meta in self._metadata.items():
            if (query in name or
                query in meta.description.lower() or
                query in meta.category.value):
                result.append(name)
        return result

    def info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取能力完整信息"""
        meta = self._metadata.get(name)
        if meta is None:
            return None
        return {
            "name": meta.name,
            "category": meta.category.value,
            "description": meta.description,
            "engine": meta.engine,
            "params_schema": meta.params_schema,
            "inputs_schema": meta.inputs_schema,
            "output_schema": meta.output_schema,
            "examples": meta.examples,
            "dependencies": meta.dependencies,
            "tags": meta.tags,
        }

    def validate_inputs(self, name: str, inputs: Dict[str, Any]) -> tuple[bool, List[str]]:
        """验证输入参数"""
        meta = self._metadata.get(name)
        if meta is None:
            return False, [f"能力 '{name}' 不存在"]

        errors = []
        required_inputs = meta.inputs_schema.get("required", [])
        for req in required_inputs:
            if req not in inputs:
                errors.append(f"缺少必需输入参数: {req}")

        return len(errors) == 0, errors

    def validate_params(self, name: str, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """验证配置参数"""
        meta = self._metadata.get(name)
        if meta is None:
            return False, [f"能力 '{name}' 不存在"]

        errors = []
        required_params = meta.params_schema.get("required", [])
        for req in required_params:
            if req not in params:
                errors.append(f"缺少必需参数: {req}")

        return len(errors) == 0, errors

    def stats(self) -> Dict[str, Any]:
        """获取注册统计"""
        return {
            "total": len(self._capabilities),
            "by_category": {cat: len(items) for cat, items in self._category_index.items()},
            "by_engine": {eng: len(items) for eng, items in self._engine_index.items()},
        }


# =============================================================================
# 全局注册表实例
# =============================================================================

_CAPABILITY_REGISTRY: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    """获取全局能力注册表"""
    global _CAPABILITY_REGISTRY
    if _CAPABILITY_REGISTRY is None:
        _CAPABILITY_REGISTRY = CapabilityRegistry()
        _register_all_capabilities(_CAPABILITY_REGISTRY)
    return _CAPABILITY_REGISTRY


# 延迟导入并注册所有能力
def _register_all_capabilities(registry: CapabilityRegistry) -> None:
    """注册所有能力到注册表"""
    # Vector Engine
    from geoagent.geo_engine.capability.vector_capabilities import (
        vector_buffer, vector_dissolve, vector_union, vector_intersect,
        vector_clip, vector_erase, vector_split, vector_merge,
        vector_simplify, vector_reproject, vector_centroid, vector_convex_hull,
        vector_spatial_join, vector_nearest_join, vector_calculate_area,
        vector_calculate_length,
    )

    vector_caps = [
        ("vector_buffer", vector_buffer, "缓冲区分析", "geopandas"),
        ("vector_dissolve", vector_dissolve, "矢量融合", "geopandas"),
        ("vector_union", vector_union, "矢量合并（Union）", "geopandas"),
        ("vector_intersect", vector_intersect, "矢量相交", "geopandas"),
        ("vector_clip", vector_clip, "矢量裁剪", "geopandas"),
        ("vector_erase", vector_erase, "矢量擦除", "geopandas"),
        ("vector_split", vector_split, "矢量分割", "geopandas"),
        ("vector_merge", vector_merge, "矢量合并", "geopandas"),
        ("vector_simplify", vector_simplify, "矢量简化", "geopandas"),
        ("vector_reproject", vector_reproject, "矢量投影转换", "geopandas"),
        ("vector_centroid", vector_centroid, "计算质心", "geopandas"),
        ("vector_convex_hull", vector_convex_hull, "凸包计算", "geopandas"),
        ("vector_spatial_join", vector_spatial_join, "空间连接", "geopandas"),
        ("vector_nearest_join", vector_nearest_join, "最近邻连接", "geopandas"),
        ("vector_calculate_area", vector_calculate_area, "计算面积", "geopandas"),
        ("vector_calculate_length", vector_calculate_length, "计算长度", "geopandas"),
    ]

    for name, func, desc, engine in vector_caps:
        registry.register(
            name=name,
            func=func,
            category=CapabilityCategory.VECTOR,
            description=desc,
            engine=engine,
            dependencies=["geopandas", "shapely"],
            tags=["vector", "geometry", "analysis"],
        )

    # Raster Engine
    from geoagent.geo_engine.capability.raster_capabilities import (
        raster_clip, raster_mask, raster_merge, raster_resample,
        raster_reproject, raster_calculator, raster_slope, raster_aspect,
        raster_hillshade, raster_ndvi, raster_zonal_stats, raster_contour,
        raster_reclassify, raster_fill_nodata, raster_warp,
    )

    raster_caps = [
        ("raster_clip", raster_clip, "栅格裁剪", "rasterio"),
        ("raster_mask", raster_mask, "栅格掩膜", "rasterio"),
        ("raster_merge", raster_merge, "栅格合并", "rasterio"),
        ("raster_resample", raster_resample, "栅格重采样", "rasterio"),
        ("raster_reproject", raster_reproject, "栅格重投影", "rasterio"),
        ("raster_calculator", raster_calculator, "栅格计算器", "rasterio"),
        ("raster_slope", raster_slope, "坡度计算", "rasterio"),
        ("raster_aspect", raster_aspect, "坡向计算", "rasterio"),
        ("raster_hillshade", raster_hillshade, "山体阴影", "rasterio"),
        ("raster_ndvi", raster_ndvi, "NDVI计算", "rasterio"),
        ("raster_zonal_stats", raster_zonal_stats, "分区统计", "rasterio"),
        ("raster_contour", raster_contour, "等值线提取", "rasterio"),
        ("raster_reclassify", raster_reclassify, "栅格重分类", "rasterio"),
        ("raster_fill_nodata", raster_fill_nodata, "填充nodata", "rasterio"),
        ("raster_warp", raster_warp, "栅格仿射变换", "rasterio"),
    ]

    for name, func, desc, engine in raster_caps:
        registry.register(
            name=name,
            func=func,
            category=CapabilityCategory.RASTER,
            description=desc,
            engine=engine,
            dependencies=["rasterio", "numpy"],
            tags=["raster", "terrain", "analysis"],
        )

    # Network Engine
    from geoagent.geo_engine.capability.network_capabilities import (
        network_shortest_path, network_k_shortest_paths, network_isochrone,
        network_service_area, network_closest_facility, network_location_allocation,
        network_flow_analysis, network_accessibility_score,
    )

    network_caps = [
        ("network_shortest_path", network_shortest_path, "最短路径分析", "osmnx"),
        ("network_k_shortest_paths", network_k_shortest_paths, "K条最短路径", "osmnx"),
        ("network_isochrone", network_isochrone, "等时圈分析", "osmnx"),
        ("network_service_area", network_service_area, "服务区分析", "osmnx"),
        ("network_closest_facility", network_closest_facility, "最近设施分析", "osmnx"),
        ("network_location_allocation", network_location_allocation, "选址分配分析", "osmnx"),
        ("network_flow_analysis", network_flow_analysis, "网络流量分析", "networkx"),
        ("network_accessibility_score", network_accessibility_score, "可达性评分", "osmnx"),
    ]

    for name, func, desc, engine in network_caps:
        registry.register(
            name=name,
            func=func,
            category=CapabilityCategory.NETWORK,
            description=desc,
            engine=engine,
            dependencies=["osmnx", "networkx"],
            tags=["network", "routing", "accessibility"],
        )

    # Analysis Engine
    from geoagent.geo_engine.capability.analysis_capabilities import (
        analysis_idw, analysis_kriging, analysis_kde, analysis_hotspot,
        analysis_cluster_kmeans, analysis_spatial_autocorrelation,
        analysis_distance_matrix, analysis_weighted_overlay,
    )

    analysis_caps = [
        ("analysis_idw", analysis_idw, "反距离加权插值", "scipy"),
        ("analysis_kriging", analysis_kriging, "克里金插值", "pykrige"),
        ("analysis_kde", analysis_kde, "核密度估计", "scipy"),
        ("analysis_hotspot", analysis_hotspot, "热点分析", "pysal"),
        ("analysis_cluster_kmeans", analysis_cluster_kmeans, "K-Means聚类", "sklearn"),
        ("analysis_spatial_autocorrelation", analysis_spatial_autocorrelation, "空间自相关", "pysal"),
        ("analysis_distance_matrix", analysis_distance_matrix, "距离矩阵", "scipy"),
        ("analysis_weighted_overlay", analysis_weighted_overlay, "加权叠置分析", "geopandas"),
    ]

    for name, func, desc, engine in analysis_caps:
        registry.register(
            name=name,
            func=func,
            category=CapabilityCategory.ANALYSIS,
            description=desc,
            engine=engine,
            dependencies=["scipy", "geopandas"],
            tags=["analysis", "interpolation", "statistics"],
        )

    # IO Engine
    from geoagent.geo_engine.capability.io_capabilities import (
        io_read_vector, io_read_raster, io_write_vector, io_write_raster,
        io_geocode, io_reverse_geocode, io_fetch_osm, io_fetch_stac,
    )

    io_caps = [
        ("io_read_vector", io_read_vector, "读取矢量数据", "fiona"),
        ("io_read_raster", io_read_raster, "读取栅格数据", "rasterio"),
        ("io_write_vector", io_write_vector, "写入矢量数据", "fiona"),
        ("io_write_raster", io_write_raster, "写入栅格数据", "rasterio"),
        ("io_geocode", io_geocode, "地理编码", "geopy"),
        ("io_reverse_geocode", io_reverse_geocode, "反向地理编码", "geopy"),
        ("io_fetch_osm", io_fetch_osm, "获取OSM数据", "osmnx"),
        ("io_fetch_stac", io_fetch_stac, "搜索STAC影像", "pystac-client"),
    ]

    for name, func, desc, engine in io_caps:
        registry.register(
            name=name,
            func=func,
            category=CapabilityCategory.IO,
            description=desc,
            engine=engine,
            dependencies=["geopandas", "rasterio"],
            tags=["io", "read", "write", "geocoding"],
        )


# =============================================================================
# 便捷访问函数
# =============================================================================

def get_capability(name: str) -> Optional[CapabilityFunc]:
    """获取能力函数"""
    return get_capability_registry().get(name)


def execute_capability(name: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """执行能力"""
    return get_capability_registry().execute(name, inputs, params)


def list_capabilities(category: str = None, engine: str = None) -> List[str]:
    """列出能力"""
    if category:
        return get_capability_registry().list_by_category(category)
    if engine:
        return get_capability_registry().list_by_engine(engine)
    return get_capability_registry().list_all()


def search_capabilities(query: str) -> List[str]:
    """搜索能力"""
    return get_capability_registry().search(query)


def capability_info(name: str) -> Optional[Dict[str, Any]]:
    """获取能力信息"""
    return get_capability_registry().info(name)


# =============================================================================
# 向后兼容别名
# =============================================================================

CAPABILITY_REGISTRY = get_capability_registry()


__all__ = [
    # 类别枚举
    "CapabilityCategory",
    # 元数据
    "CapabilityMeta",
    "CapabilityFunc",
    # 注册表类
    "CapabilityRegistry",
    # 全局实例
    "CAPABILITY_REGISTRY",
    "get_capability_registry",
    # 便捷函数
    "get_capability",
    "execute_capability",
    "list_capabilities",
    "search_capabilities",
    "capability_info",
]
