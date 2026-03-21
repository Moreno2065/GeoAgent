"""
固化的高频 GIS 基础工具
"""

from pathlib import Path
import json
import threading

# 直接导入第三方库
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


# 全局工作目录配置（支持按对话设置独立的 workspace）
_workspace_lock = threading.Lock()
_current_workspace_dir: Path | None = None


def get_workspace_dir() -> Path:
    """获取 workspace 目录路径"""
    global _current_workspace_dir
    with _workspace_lock:
        if _current_workspace_dir is not None:
            return _current_workspace_dir
    return Path(__file__).resolve().parents[3] / "workspace"


def set_conversation_workspace(conv_id: str | None) -> None:
    """
    设置当前对话的工作目录

    Args:
        conv_id: 对话 ID，如果为 None 则使用默认的 workspace 目录
    """
    global _current_workspace_dir
    with _workspace_lock:
        if conv_id is None:
            _current_workspace_dir = None
        else:
            base_workspace = Path(__file__).resolve().parents[3] / "workspace"
            conv_dir = base_workspace / "conversation_files" / conv_id
            conv_dir.mkdir(parents=True, exist_ok=True)
            _current_workspace_dir = conv_dir


def get_data_info(file_name: str) -> str:
    """
    读取 workspace/ 下的矢量或栅格文件，返回其元数据信息。

    Args:
        file_name: 文件名（不含路径），如 "data.shp" 或 "dem.tif"

    Returns:
        包含 EPSG 坐标系代码、边界(BBox)和字段列表/波段数的 JSON 字符串
    """
    workspace = get_workspace_dir()
    file_path = workspace / file_name

    if not file_path.exists():
        return json.dumps({
            "success": False,
            "error": f"文件不存在: {file_name}",
            "workspace_path": str(workspace)
        }, ensure_ascii=False, indent=2)

    suffix = file_path.suffix.lower()

    # 处理矢量数据
    if suffix in ['.shp', '.json', '.geojson', '.gpkg', '.gjson']:
        return _get_vector_info(file_path)
    # 处理栅格数据
    elif suffix in ['.tif', '.tiff', '.img', '.asc', '.rst', '.nc']:
        return _get_raster_info(file_path)
    else:
        return json.dumps({
            "success": False,
            "error": f"不支持的文件格式: {suffix}",
            "supported_formats": {
                "vector": [".shp", ".json", ".geojson", ".gpkg"],
                "raster": [".tif", ".tiff", ".img", ".asc", ".rst"]
            }
        }, ensure_ascii=False, indent=2)


def _get_vector_info(file_path: Path) -> str:
    """获取矢量数据信息"""
    if not HAS_GEOPANDAS:
        return json.dumps({
            "success": False,
            "error": "geopandas 库未安装",
            "install_hint": "pip install geopandas"
        }, ensure_ascii=False, indent=2)

    try:
        gdf = gpd.read_file(file_path)

        # 获取 CRS 信息
        crs = gdf.crs
        epsg_code = None
        crs_wkt = None
        if crs is not None:
            try:
                epsg_code = crs.to_epsg()
            except Exception:
                pass
            try:
                crs_wkt = crs.to_wkt()
            except Exception:
                pass

        # 获取边界
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        bbox = {
            "min_x": float(bounds[0]),
            "min_y": float(bounds[1]),
            "max_x": float(bounds[2]),
            "max_y": float(bounds[3])
        }

        # 获取字段列表
        columns = list(gdf.columns)

        # 获取几何类型
        geom_type = gdf.geom_type.value_counts().to_dict() if len(gdf) > 0 else {}

        # 获取要素数量
        feature_count = len(gdf)

        result = {
            "file_type": "vector",
            "file_name": file_path.name,
            "feature_count": feature_count,
            "geometry_type": geom_type,
            "crs": {
                "epsg": epsg_code,
                "wkt": crs_wkt
            },
            "bbox": bbox,
            "columns": columns,
            "dtypes": {col: str(dtype) for col, dtype in gdf.dtypes.items()}
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"读取矢量文件失败: {str(e)}",
            "file_name": str(file_path)
        }, ensure_ascii=False, indent=2)


def _get_raster_info(file_path: Path) -> str:
    """获取栅格数据信息"""
    if not HAS_RASTERIO:
        return json.dumps({
            "success": False,
            "error": "rasterio 库未安装",
            "install_hint": "pip install rasterio"
        }, ensure_ascii=False, indent=2)

    try:
        with rasterio.open(file_path) as src:
            # 获取 CRS
            crs = src.crs
            epsg_code = None
            crs_wkt = None
            if crs is not None:
                try:
                    epsg_code = crs.to_epsg()
                except Exception:
                    pass
                try:
                    crs_wkt = crs.to_wkt()
                except Exception:
                    pass

            # 获取边界
            bbox = {
                "left": src.bounds.left,
                "right": src.bounds.right,
                "bottom": src.bounds.bottom,
                "top": src.bounds.top
            }

            # 获取波段数
            count = src.count

            # 获取分辨率
            res_x = src.res[0]
            res_y = src.res[1]

            # 获取数据类型
            dtypes = src.dtypes

            # 获取 nodata 值
            nodata_values = src.nodatavals

            # 获取变换矩阵
            transform = src.transform

            result = {
                "file_type": "raster",
                "file_name": file_path.name,
                "band_count": count,
                "crs": {
                    "epsg": epsg_code,
                    "wkt": crs_wkt
                },
                "bbox": bbox,
                "resolution": {
                    "x": res_x,
                    "y": res_y
                },
                "dtypes": dtypes,
                "nodata_values": nodata_values,
                "transform": {
                    "a": transform.a,
                    "b": transform.b,
                    "c": transform.c,
                    "d": transform.d,
                    "e": transform.e,
                    "f": transform.f
                }
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"读取栅格文件失败: {str(e)}",
            "file_name": str(file_path)
        }, ensure_ascii=False, indent=2)


def list_workspace_files() -> list:
    """
    列出 workspace 目录下的所有 GIS 文件

    Returns:
        文件名列表
    """
    workspace = get_workspace_dir()
    if not workspace.exists():
        return []

    gis_extensions = {'.shp', '.json', '.geojson', '.gpkg', '.gjson',
                      '.tif', '.tiff', '.img', '.asc', '.rst', '.nc'}

    files = []
    for f in workspace.iterdir():
        if f.is_file() and f.suffix.lower() in gis_extensions:
            files.append(f.name)

    return sorted(files)
