"""
数据探针（Data Profiler）
========================
极速提取空间文件的 Schema 元数据，用于注入 Prompt 让大模型精准识别字段名。

核心设计原则：
- 只读前 5 行数据，极速不卡顿
- 提取 dtype、样本数据，让大模型"看见"数据长什么样
- 带 TTL 缓存，避免重复 IO
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geoagent.gis_tools.fixed_tools import get_workspace_dir


# =============================================================================
# 依赖检查
# =============================================================================

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    gpd = None
    HAS_GEOPANDAS = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False


# =============================================================================
# dtype → 人类可读类型映射
# =============================================================================

def _map_dtype(dtype: str) -> str:
    """将 pandas/numpy dtype 映射为人类可读的中文类型"""
    dtype_lower = str(dtype).lower()
    if dtype_lower in ("int64", "int32", "int16", "int8"):
        return "整数"
    if dtype_lower in ("float64", "float32"):
        return "浮点数"
    if dtype_lower in ("object", "string", "str"):
        return "文本"
    if dtype_lower in ("bool", "boolean"):
        return "布尔"
    if dtype_lower.startswith("datetime") or dtype_lower.startswith("timestamp"):
        return "日期时间"
    if dtype_lower in ("geometry", "point", "linestring", "polygon", "multipolygon",
                       "multilinestring", "multipoint"):
        return "几何对象"
    return "未知"


# =============================================================================
# 核心探针函数：sniff_spatial_file
# =============================================================================

def sniff_spatial_file(file_path: Path) -> Dict[str, Any]:
    """
    数据探针：极速提取单个空间文件的 Schema 元数据

    Args:
        file_path: 文件路径（Path 对象）

    Returns:
        包含以下键的字典：
        - success: bool，读取是否成功
        - file_name: str，文件名
        - file_type: str，"vector" 或 "raster"
        - geometry_type: str，几何类型（如 "Polygon", "LineString"）
        - crs: str，坐标系（如 "EPSG:4326" 或 "Unknown"）
        - columns: List[str]，字段列表（不含 geometry 列）
        - dtypes: Dict[str, str]，字段类型映射（字段名 → dtype）
        - column_types: Dict[str, str]，字段类型映射（字段名 → 人类可读类型）
        - numeric_columns: List[str]，数值字段列表
        - text_columns: List[str]，文本字段列表
        - sample_data: List[Dict]，前2条记录的 dict（不含 geometry 列）
        - feature_count: int，要素数量（估算）
        - error: str（可选），失败时的错误信息
    """
    suffix = file_path.suffix.lower()

    if suffix in (".shp", ".json", ".geojson", ".gpkg", ".gjson", ".kml", ".gml", ".fgb"):
        return _sniff_vector(file_path)
    elif suffix in (".tif", ".tiff", ".img", ".asc", ".rst", ".nc"):
        return _sniff_raster(file_path)
    else:
        return {
            "success": False,
            "file_name": file_path.name,
            "error": f"不支持的文件格式: {suffix}",
        }


def _sniff_vector(file_path: Path) -> Dict[str, Any]:
    """探针：矢量文件"""
    if not HAS_GEOPANDAS:
        return {
            "success": False,
            "file_name": file_path.name,
            "error": "geopandas 库未安装，无法读取矢量文件",
        }

    try:
        # 🚀 关键：只读前 5 行，极速不卡顿
        gdf = gpd.read_file(file_path, rows=5)

        # 几何类型
        geom_type = "Unknown"
        if not gdf.empty:
            unique_types = gdf.geometry.type.unique()
            if len(unique_types) > 0:
                geom_type = unique_types[0]

        # 坐标系
        crs_str = "Unknown"
        if gdf.crs is not None:
            try:
                epsg = gdf.crs.to_epsg()
                crs_str = f"EPSG:{epsg}" if epsg else "Unknown"
            except Exception:
                pass

        # 字段列表（去掉 geometry 列）
        all_columns = list(gdf.columns)
        columns = [c for c in all_columns if c.lower() != "geometry"]

        # dtype 映射
        dtypes_raw = {col: str(dtype) for col, dtype in gdf.dtypes.items()}
        column_types = {col: _map_dtype(dtypes_raw.get(col, "")) for col in columns}

        # 数值/文本字段分类
        numeric_columns = [
            c for c in columns
            if dtypes_raw.get(c, "") in ("int64", "float64", "int32", "float32", "int16", "float16")
        ]
        text_columns = [
            c for c in columns
            if dtypes_raw.get(c, "") in ("object", "string", "str")
        ]

        # 🚀 样本数据：取前 2 条，不含 geometry 列
        sample_data = []
        if not gdf.empty and columns:
            # 去掉 geometry 之后再 to_dict
            df_without_geom = gdf.drop(columns=["geometry"], errors="ignore")
            # 只保留前 2 行
            df_sample = df_without_geom.head(2)
            # 转成 dict list，处理可能的不可序列化对象
            for _, row in df_sample.iterrows():
                record = {}
                for col in df_sample.columns:
                    val = row[col]
                    # 尝试转为可序列化类型
                    if hasattr(val, "item"):  # numpy 类型
                        try:
                            val = val.item()
                        except Exception:
                            val = str(val)
                    elif hasattr(val, "isoformat"):  # datetime 类型
                        val = val.isoformat()
                    else:
                        try:
                            str(val)  # 测试是否可序列化
                        except Exception:
                            val = str(val)
                    # 确保字符串可读（处理乱码和编码问题）
                    if isinstance(val, bytes):
                        for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
                            try:
                                val = val.decode(enc)
                                break
                            except Exception:
                                continue
                    record[col] = val
                sample_data.append(record)

        return {
            "success": True,
            "file_name": file_path.name,
            "file_type": "vector",
            "geometry_type": geom_type,
            "crs": crs_str,
            "columns": columns,
            "dtypes": {k: v for k, v in dtypes_raw.items() if k in columns},
            "column_types": {k: v for k, v in column_types.items() if k in columns},
            "numeric_columns": numeric_columns,
            "text_columns": text_columns,
            "sample_data": sample_data,
            "feature_count": len(gdf),  # 注意：这里只读了 5 条，是估算值
        }

    except Exception as e:
        return {
            "success": False,
            "file_name": file_path.name,
            "error": f"读取矢量文件元数据失败: {str(e)}",
        }


def _sniff_raster(file_path: Path) -> Dict[str, Any]:
    """探针：栅格文件"""
    if not HAS_RASTERIO:
        return {
            "success": False,
            "file_name": file_path.name,
            "error": "rasterio 库未安装，无法读取栅格文件",
        }

    try:
        with rasterio.open(file_path) as src:
            crs_str = "Unknown"
            if src.crs is not None:
                try:
                    epsg = src.crs.to_epsg()
                    crs_str = f"EPSG:{epsg}" if epsg else "Unknown"
                except Exception:
                    pass

            band_count = src.count
            res_x, res_y = src.res
            dtypes = list(src.dtypes)

            # 栅格不做样本数据提取（几何类型不同）
            return {
                "success": True,
                "file_name": file_path.name,
                "file_type": "raster",
                "geometry_type": "Raster",
                "crs": crs_str,
                "columns": [],  # 栅格无字段列
                "dtypes": {},
                "column_types": {},
                "numeric_columns": [],
                "text_columns": [],
                "sample_data": [],
                "feature_count": band_count,  # 波段数
                "raster_info": {
                    "band_count": band_count,
                    "resolution": f"{res_x:.2f} x {res_y:.2f}",
                    "dtypes": dtypes,
                },
            }

    except Exception as e:
        return {
            "success": False,
            "file_name": file_path.name,
            "error": f"读取栅格文件元数据失败: {str(e)}",
        }


# =============================================================================
# 格式化输出：sniff_spatial_file_as_text
# =============================================================================

def sniff_spatial_file_as_text(file_path: Path) -> str:
    """
    数据探针：极速提取并返回格式化的文本，用于直接注入 Prompt

    Args:
        file_path: 文件路径

    Returns:
        格式化的人类可读文本块，如：
        - 📄 土地利用.shp
          - 几何类型: Polygon
          - 坐标系: EPSG:4326
          - 属性字段:
            - Id (整数)
            - landuse (文本)
            - Area (浮点数)
          - 数据样例:
            - {'Id': 1, 'landuse': '工业用地', 'Area': 1234.5}
    """
    result = sniff_spatial_file(file_path)

    if not result.get("success"):
        return f"- {result.get('file_name', file_path.name)} (读取元数据失败)"

    lines = []
    lines.append(f"- 📄 {result['file_name']}")

    # 几何类型 + 坐标系
    geom_type = result.get("geometry_type", "Unknown")
    crs = result.get("crs", "Unknown")
    lines.append(f"  - 几何类型: {geom_type}")
    lines.append(f"  - 坐标系: {crs}")

    # 属性字段
    columns = result.get("columns", [])
    column_types = result.get("column_types", {})
    if columns:
        lines.append("  - 属性字段:")
        for col in columns:
            type_label = column_types.get(col, "未知")
            lines.append(f"    - {col} ({type_label})")
    elif result.get("file_type") == "raster":
        raster_info = result.get("raster_info", {})
        band_count = raster_info.get("band_count", "?")
        res = raster_info.get("resolution", "?")
        lines.append(f"  - 波段数: {band_count}")
        lines.append(f"  - 分辨率: {res}")

    # 数据样例
    sample_data = result.get("sample_data", [])
    if sample_data:
        lines.append("  - 数据样例:")
        for record in sample_data:
            lines.append(f"    - {record}")

    return "\n".join(lines)


# =============================================================================
# 工作区批量探针
# =============================================================================

def sniff_workspace_dir(workspace_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    扫描工作区目录，批量探针所有空间文件

    Args:
        workspace_dir: 工作区目录路径，默认为 get_workspace_dir()

    Returns:
        每个文件的探针结果字典列表
    """
    if workspace_dir is None:
        workspace_dir = get_workspace_dir()

    if not workspace_dir.exists():
        return []

    gis_extensions = {
        ".shp", ".geojson", ".json", ".gpkg", ".gjson", ".kml", ".gml", ".fgb",
        ".tif", ".tiff", ".img", ".asc", ".rst", ".nc",
    }

    results = []
    for f in workspace_dir.iterdir():
        if f.is_file() and f.suffix.lower() in gis_extensions:
            result = sniff_spatial_file(f)
            if result.get("success") is not False:
                results.append(result)

    return results


# =============================================================================
# 缓存层（TTL 机制，避免重复 IO）
# =============================================================================

_profiler_cache: Dict[str, List[Dict[str, Any]]] = {}  # key = workspace 路径
_profiler_cache_time: Dict[str, float] = {}  # key = workspace 路径
_PROFILER_CACHE_TTL: float = 60.0  # 缓存 60 秒
_cache_lock = threading.Lock()


def sniff_workspace_dir_cached(workspace_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    带 TTL 缓存的工作区批量探针（推荐使用）

    关键修复：缓存 key = workspace 路径。
    这样当 set_conversation_workspace 切换到对话目录时，
    能正确扫描对话目录里的文件，而不是永远返回主 workspace 的缓存结果。

    Args:
        workspace_dir: 工作区目录路径

    Returns:
        每个文件的探针结果字典列表（带缓存）
    """
    global _profiler_cache, _profiler_cache_time

    if workspace_dir is None:
        workspace_dir = get_workspace_dir()

    cache_key = str(workspace_dir.resolve())
    current_time = time.time()

    with _cache_lock:
        expired = (
            cache_key not in _profiler_cache
            or (current_time - _profiler_cache_time.get(cache_key, 0)) > _PROFILER_CACHE_TTL
        )
        if expired:
            # 检查文件是否更新了（如果缓存过期，重新扫描）
            _profiler_cache[cache_key] = sniff_workspace_dir(workspace_dir)
            _profiler_cache_time[cache_key] = current_time

    return _profiler_cache[cache_key]


def clear_profiler_cache() -> None:
    """手动清除探针缓存（工作区文件更新后调用）"""
    global _profiler_cache, _profiler_cache_time
    with _cache_lock:
        _profiler_cache.clear()
        _profiler_cache_time.clear()


# =============================================================================
# Profile Block 构建（用于 Prompt 注入）
# =============================================================================

def build_workspace_profile_block(
    profiles: Optional[List[Dict[str, Any]]] = None,
    workspace_dir: Optional[Path] = None,
) -> str:
    """
    构建工作区文件的 Profile Block，用于注入 Prompt

    Args:
        profiles: 已探针的结果列表（可选，不提供则自动探针）
        workspace_dir: 工作区目录（profiles 为 None 时使用）

    Returns:
        格式化的文本块，用于注入 Prompt。示例：

        【工作区文件详细情报】
        - 📄 土地利用.shp
          - 几何类型: Polygon
          - 坐标系: EPSG:4326
          - 属性字段:
            - Id (整数)
            - landuse (文本)
            - Area (浮点数)
          - 数据样例:
            - {'Id': 1, 'landuse': '工业用地', 'Area': 1234.5}
        - 📄 大厦小区.shp
          - 几何类型: Polygon
          - 坐标系: EPSG:4490
          - 属性字段:
            - name (文本)
            - 类型 (文本)
            - price (浮点数)
          - 数据样例:
            - {'name': '阳光小区', '类型': '住宅小区', 'price': 15000.0}

        【铁律】：当你需要进行属性筛选(filter)或分类渲染时，
        必须严格使用上方情报中提供的【属性字段】名！
    """
    if profiles is None:
        profiles = sniff_workspace_dir_cached(workspace_dir)

    if not profiles:
        return ""

    lines = ["【工作区文件详细情报】"]

    for result in profiles:
        lines.append(f"- 📄 {result['file_name']}")
        lines.append(f"  - 几何类型: {result.get('geometry_type', 'Unknown')}")
        lines.append(f"  - 坐标系: {result.get('crs', 'Unknown')}")

        columns = result.get("columns", [])
        column_types = result.get("column_types", {})
        if columns:
            lines.append("  - 属性字段:")
            for col in columns:
                type_label = column_types.get(col, "未知")
                lines.append(f"    - {col} ({type_label})")
        elif result.get("file_type") == "raster":
            raster_info = result.get("raster_info", {})
            lines.append(f"  - 波段数: {raster_info.get('band_count', '?')}")
            lines.append(f"  - 分辨率: {raster_info.get('resolution', '?')}")

        sample_data = result.get("sample_data", [])
        if sample_data:
            lines.append("  - 数据样例:")
            for record in sample_data:
                lines.append(f"    - {record}")

        lines.append("")  # 空行分隔

    # 🚨 铁律提醒：强制大模型使用真实字段名
    lines.append("【铁律】：当你需要进行属性筛选(filter)或分类渲染时，")
    lines.append("必须严格使用上方情报中提供的【属性字段】名，")
    lines.append("禁止瞎猜字段名（如 type、category、name 等），")
    lines.append("必须对照上方的真实字段列表！")

    return "\n".join(lines)


# =============================================================================
# 便捷入口：快速探针并格式化
# =============================================================================

def quick_profile(workspace_dir: Optional[Path] = None) -> str:
    """
    快速入口：探针工作区所有文件并返回格式化文本（用于直接注入 Prompt）

    Args:
        workspace_dir: 工作区目录，默认使用 get_workspace_dir()

    Returns:
        格式化的工作区情报文本
    """
    return build_workspace_profile_block(workspace_dir=workspace_dir)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "sniff_spatial_file",
    "sniff_spatial_file_as_text",
    "sniff_workspace_dir",
    "sniff_workspace_dir_cached",
    "clear_profiler_cache",
    "build_workspace_profile_block",
    "quick_profile",
]
