"""
GeoEngine 数据标准化层
=====================
统一所有模块间的数据格式：
  - 矢量 → GeoDataFrame
  - 栅格 → xarray / rasterio dataset
  - 输出 → GeoJSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass


# =============================================================================
# 数据类型标识
# =============================================================================

class DataType:
    """数据类型标识"""
    GEOJSON = "geojson"
    GDF = "gdf"
    RASTER = "raster"
    XARRAY = "xarray"
    GRAPH = "graph"
    COORDS = "coords"
    UNKNOWN = "unknown"


# =============================================================================
# 标准化结果
# =============================================================================

@dataclass
class NormalizedResult:
    """标准化后的数据结果"""
    data: Any
    dtype: str
    crs: Optional[str] = None
    bounds: Optional[tuple] = None
    feature_count: Optional[int] = None
    shape: Optional[tuple] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_geojson(self) -> Dict[str, Any]:
        """转换为 GeoJSON 格式"""
        if self.dtype == DataType.GEOJSON:
            return self.data
        if self.dtype == DataType.GDF:
            return self.data.__geo_interface__
        if self.dtype == DataType.GRAPH:
            import networkx as nx
            return nx.node_link_data(self.data)
        return self.data

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dtype": self.dtype,
            "crs": self.crs,
            "bounds": self.bounds,
            "feature_count": self.feature_count,
            "shape": self.shape,
            "metadata": self.metadata,
        }


# =============================================================================
# 工作空间路径解析
# =============================================================================

def get_workspace() -> Path:
    """获取工作空间路径"""
    return Path(__file__).parent.parent.parent / "workspace"


def resolve_path(file_name: str) -> Path:
    """解析文件路径（相对路径 → workspace/）"""
    f = Path(file_name)
    if f.is_absolute():
        return f
    return get_workspace() / file_name


def ensure_dir(filepath: str) -> None:
    """确保输出目录存在"""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 数据标准化
# =============================================================================

def normalize_to_gdf(data: Any) -> "geopandas.GeoDataFrame":
    """
    将各种输入标准化为 GeoDataFrame

    支持的输入：
    - GeoDataFrame
    - GeoJSON dict
    - GeoJSON file path
    - CSV with lat/lon columns
    - Shapely geometry
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if isinstance(data, gpd.GeoDataFrame):
        return data

    if isinstance(data, dict):
        if "type" in data and data["type"] == "FeatureCollection":
            return gpd.GeoDataFrame.from_features(data["features"], crs=data.get("crs"))
        elif "type" in data and data["type"] == "Feature":
            return gpd.GeoDataFrame.from_features([data], crs=data.get("crs"))
        return gpd.GeoDataFrame.from_features(data["features"])

    if isinstance(data, str):
        path = resolve_path(data)
        if path.exists():
            return gpd.read_file(path)
        return None

    if hasattr(data, "__geo_interface__"):
        return gpd.GeoDataFrame.from_features([data])

    return None


def normalize_to_raster(data: Any) -> "rasterio.DatasetReader":
    """
    将各种输入标准化为 rasterio dataset

    支持的输入：
    - rasterio dataset
    - GeoTIFF file path
    - COG URL
    """
    import rasterio

    if hasattr(data, "read"):
        return data

    if isinstance(data, str):
        path = resolve_path(data)
        if path.exists():
            return rasterio.open(path)
        return None

    return None


def normalize_to_graph(data: Any) -> "networkx.Graph":
    """
    将各种输入标准化为 NetworkX 图

    支持的输入：
    - networkx.Graph
    - GraphML file path
    - OSM data
    """
    import networkx as nx

    if isinstance(data, nx.Graph):
        return data

    if isinstance(data, str):
        path = resolve_path(data)
        if path.exists():
            if path.suffix == ".graphml":
                return nx.read_graphml(path)
        return None

    return None


# =============================================================================
# 统一 normalize 函数
# =============================================================================

def normalize(data: Any, target_type: str = DataType.GDF) -> NormalizedResult:
    """
    将数据标准化为指定格式

    Args:
        data: 输入数据
        target_type: 目标类型 (gdf, raster, graph, geojson)

    Returns:
        NormalizedResult 对象
    """
    if target_type == DataType.GDF or target_type == DataType.GEOJSON:
        gdf = normalize_to_gdf(data)
        if gdf is not None:
            return NormalizedResult(
                data=gdf,
                dtype=DataType.GDF,
                crs=str(gdf.crs) if gdf.crs else None,
                bounds=gdf.total_bounds.tolist() if len(gdf) > 0 else None,
                feature_count=len(gdf),
                metadata={"columns": list(gdf.columns)},
            )

    if target_type == DataType.RASTER:
        raster = normalize_to_raster(data)
        if raster is not None:
            return NormalizedResult(
                data=raster,
                dtype=DataType.RASTER,
                crs=str(raster.crs) if raster.crs else None,
                bounds=raster.bounds,
                shape=(raster.count, raster.height, raster.width),
                metadata={"driver": raster.driver, "dtype": raster.dtypes},
            )

    if target_type == DataType.GRAPH:
        graph = normalize_to_graph(data)
        if graph is not None:
            return NormalizedResult(
                data=graph,
                dtype=DataType.GRAPH,
                feature_count=graph.number_of_nodes(),
                metadata={
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                },
            )

    return NormalizedResult(data=data, dtype=DataType.UNKNOWN)


# =============================================================================
# 输出格式化
# =============================================================================

def format_geojson(data: Any) -> Dict[str, Any]:
    """
    将数据格式化为 GeoJSON

    这是统一的输出格式，确保所有引擎输出一致。
    """
    gdf = normalize_to_gdf(data)
    if gdf is not None:
        return gdf.__geo_interface__

    graph = normalize_to_graph(data)
    if graph is not None:
        import networkx as nx
        return nx.node_link_data(graph)

    if isinstance(data, dict):
        return data

    return {}


def format_result(
    success: bool,
    data: Any = None,
    message: str = "",
    output_path: str = None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    统一的执行结果格式

    所有 GeoEngine 操作返回统一格式：
    {
        "success": bool,
        "data": Any,          # 标准化后的数据
        "geojson": Dict,      # GeoJSON 格式
        "message": str,       # 人类可读消息
        "output_path": str,   # 输出文件路径（如果有）
        "metadata": Dict,     # 元数据
    }
    """
    result = {
        "success": success,
        "message": message,
    }

    if data is not None:
        result["data"] = data
        result["geojson"] = format_geojson(data)

    if output_path:
        result["output_path"] = output_path

    if metadata:
        result["metadata"] = metadata

    return result
