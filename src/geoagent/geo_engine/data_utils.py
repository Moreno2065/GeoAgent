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


# 支持的文件扩展名（按优先级排序）
VECTOR_EXTENSIONS = [".shp", ".geojson", ".json", ".gpkg", ".gjson"]
RASTER_EXTENSIONS = [".tif", ".tiff", ".img", ".asc"]
ALL_EXTENSIONS = VECTOR_EXTENSIONS + RASTER_EXTENSIONS


def resolve_path(file_name: str, fuzzy: bool = True) -> Path:
    """
    解析文件路径（相对路径 → workspace/）

    增强特性：
    1. 精确匹配（带扩展名）
    2. 扩展名自动补全（无扩展名时尝试常见 GIS 格式）
    3. 模糊匹配（文件名片段包含，大小写不敏感）
    4. 如果都找不到，返回默认路径（让调用方决定如何处理）

    Args:
        file_name: 文件名（可能无扩展名）
        fuzzy: 是否启用模糊匹配（默认 True）

    Returns:
        解析后的文件路径
    """
    f = Path(file_name)
    if f.is_absolute():
        return f

    workspace = get_workspace()

    # 策略1：精确匹配（带扩展名）
    direct = workspace / file_name
    if direct.exists():
        return direct

    # 策略2：扩展名自动补全
    if not f.suffix:
        for ext in ALL_EXTENSIONS:
            candidate = workspace / f"{file_name}{ext}"
            if candidate.exists():
                return candidate

    # 策略3：模糊匹配（文件名片段包含）
    if fuzzy:
        name_lower = f.stem.lower()
        for existing in workspace.iterdir():
            if not existing.is_file():
                continue
            existing_stem = existing.stem.lower()
            # 完全包含关系
            if name_lower in existing_stem or existing_stem in name_lower:
                return existing

    # 策略4：大小写不敏感匹配
    if fuzzy:
        for existing in workspace.iterdir():
            if not existing.is_file():
                continue
            if existing.stem.lower() == f.stem.lower():
                return existing

    # 默认返回不存在的路径，让调用方决定处理方式
    return direct


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
            # 启用 GDAL 自动修复缺失的 .shx 文件
            import os
            os.environ["SHAPE_RESTORE_SHX"] = "YES"
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
# Shapefile 完整保存辅助函数
# =============================================================================

def save_shapefile(gdf, output_path: Path, encoding: str = "utf-8") -> None:
    """
    保存 GeoDataFrame 为 Shapefile，确保生成完整的辅助文件
    
    Shapefile 由多个文件组成：
    - .shp: 主文件（几何数据）
    - .shx: 索引文件
    - .dbf: 属性数据
    - .prj: 投影文件（CRS）
    - .cpg: 编码文件
    - .sbn/.sbx: 空间索引
    - .shp.xml: 元数据
    
    Args:
        gdf: GeoDataFrame
        output_path: 输出路径（.shp 文件路径）
        encoding: 属性数据编码（默认 utf-8）
    """
    import os
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 设置 GDAL 环境变量，启用自动生成缺失文件
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    
    # 保存主文件
    gdf.to_file(output_path, encoding=encoding)
    
    # 获取不带扩展名的基路径
    base_path = output_path.with_suffix("")
    
    # 手动补充 .prj 文件（某些驱动可能不会自动生成）
    if gdf.crs is not None:
        prj_path = base_path.with_suffix(".prj")
        if not prj_path.exists():
            try:
                crs_wkt = gdf.crs.to_wkt()
                with open(prj_path, "w", encoding="utf-8") as f:
                    f.write(crs_wkt)
            except Exception:
                pass
    
    # 生成 .cpg 文件（编码标识）
    cpg_path = base_path.with_suffix(".cpg")
    if not cpg_path.exists():
        with open(cpg_path, "w", encoding="utf-8") as f:
            f.write(encoding)


def save_vector_file(gdf, output_path: Path, driver: str = None, encoding: str = "utf-8") -> None:
    """
    智能保存矢量数据，根据扩展名选择合适的保存方式
    
    Args:
        gdf: GeoDataFrame
        output_path: 输出路径
        driver: 驱动名称（如 "ESRI Shapefile"、"GeoJSON"、"GPKG"）
        encoding: 属性数据编码（仅对 Shapefile 有效）
    """
    suffix = output_path.suffix.lower()
    
    if suffix == ".shp":
        save_shapefile(gdf, output_path, encoding=encoding)
    else:
        # 其他格式（GeoJSON、GPKG 等）直接保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if driver:
            gdf.to_file(output_path, driver=driver, encoding=encoding if suffix == ".shp" else None)
        else:
            gdf.to_file(output_path, encoding=encoding if suffix == ".shp" else None)


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
