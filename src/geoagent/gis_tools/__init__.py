"""
GIS Tools Module for GeoAgent
提供固化的基础工具和动态代码沙盒引擎
"""

from geoagent.gis_tools.fixed_tools import get_data_info, list_workspace_files

# 栅格工具（可选依赖，失败时优雅降级）
try:
    from geoagent.gis_tools.raster_ops import (
        get_raster_metadata,
        calculate_raster_index,
        run_gdal_algorithm,
        list_gdal_algorithms,
    )
except ImportError:
    get_raster_metadata = None
    calculate_raster_index = None
    run_gdal_algorithm = None
    list_gdal_algorithms = None

# 第三方库直接导入（可选依赖，失败时设为 None）
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    gpd = None
    HAS_GEOPANDAS = False

try:
    import shapely
    HAS_SHAPELY = True
except ImportError:
    shapely = None
    HAS_SHAPELY = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    folium = None
    HAS_FOLIUM = False

try:
    import osmnx
    HAS_OSMNX = True
except ImportError:
    osmnx = None
    HAS_OSMNX = False

try:
    import networkx
    HAS_NETWORKX = True
except ImportError:
    networkx = None
    HAS_NETWORKX = False

try:
    from arcgis.gis import GIS
    from arcgis.features import FeatureLayer
    import arcgis
    HAS_ARCGIS = True
except ImportError:
    arcgis = None
    GIS = None
    FeatureLayer = None
    HAS_ARCGIS = False

try:
    import whitebox
    HAS_WHITEBOX = True
except ImportError:
    whitebox = None
    HAS_WHITEBOX = False

try:
    import numpy as np
    HAS_NUMPY = True
    # 直接从 numpy 导出常用函数
    zeros = np.zeros
    ones = np.ones
    arange = np.arange
    linspace = np.linspace
    nanmean = np.nanmean
    nanstd = np.nanstd
    nanmin = np.nanmin
    nanmax = np.nanmax
    isnan = np.isnan
    isfinite = np.isfinite
    clip = np.clip
except ImportError:
    np = None
    HAS_NUMPY = False
    zeros = ones = arange = linspace = None
    nanmean = nanstd = nanmin = nanmax = None
    isnan = isfinite = clip = None

try:
    from rasterio.mask import mask as rasterio_mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling, CRS
    HAS_RASTERIO_FEATURES = True
except ImportError:
    rasterio_mask = None
    calculate_default_transform = reproject = Resampling = CRS = None
    HAS_RASTERIO_FEATURES = False

# Wrapper 类（保留兼容，但不再依赖 wrapper 模块）
class GeoDataFrameWrapper:
    """GeoDataFrame 包装器"""
    def __init__(self, gdf):
        self.gdf = gdf

class PointWrapper:
    """Shapely Point 包装器"""
    def __init__(self, geom):
        self.geom = geom

class PolygonWrapper:
    """Shapely Polygon 包装器"""
    def __init__(self, geom):
        self.geom = geom

class MultiPolygonWrapper:
    """Shapely MultiPolygon 包装器"""
    def __init__(self, geom):
        self.geom = geom

class LineStringWrapper:
    """Shapely LineString 包装器"""
    def __init__(self, geom):
        self.geom = geom

class MemoryFileWrapper:
    """Rasterio MemoryFile 包装器"""
    def __init__(self):
        if rasterio:
            self.mf = rasterio.io.MemoryFile()
        else:
            self.mf = None

class RasterioWrapper:
    """Rasterio 包装器"""
    def __init__(self, dataset):
        self.dataset = dataset

class MapWrapper:
    """Folium Map 包装器"""
    def __init__(self, m):
        self.m = m

class OsmnxWrapper:
    """OSMnx 包装器"""
    def __init__(self, G=None):
        self.G = G

class NetworkXWrapper:
    """NetworkX 包装器"""
    def __init__(self, G=None):
        self.G = G

class ArcGISWrapper:
    """ArcGIS 包装器"""
    def __init__(self, item=None):
        self.item = item

class WhiteboxWrapper:
    """Whitebox 包装器"""
    def __init__(self):
        if whitebox:
            self.wbt = whitebox.WBT()
        else:
            self.wbt = None

class ArrayWrapper:
    """NumPy Array 包装器"""
    def __init__(self, arr):
        self.arr = arr

class FoliumPlugins:
    """Folium 插件集合"""
    def __init__(self):
        pass

__all__ = [
    # 基础工具
    'get_data_info',
    'list_workspace_files',
    # 栅格工具
    'get_raster_metadata',
    'calculate_raster_index',
    'run_gdal_algorithm',
    'list_gdal_algorithms',
    # geopandas 包装器
    'gpd',
    'HAS_GEOPANDAS',
    'GeoDataFrameWrapper',
    # shapely 包装器
    'shapely',
    'HAS_SHAPELY',
    'PointWrapper',
    'PolygonWrapper',
    'MultiPolygonWrapper',
    'LineStringWrapper',
    # rasterio 包装器
    'rasterio',
    'HAS_RASTERIO',
    'MemoryFileWrapper',
    'RasterioWrapper',
    # numpy 包装器
    'np',
    'HAS_NUMPY',
    'ArrayWrapper',
    'zeros',
    'ones',
    'arange',
    'linspace',
    'nanmean',
    'nanstd',
    'nanmin',
    'nanmax',
    'isnan',
    'isfinite',
    'clip',
    # folium 包装器
    'folium',
    'HAS_FOLIUM',
    'MapWrapper',
    'FoliumPlugins',
    # osmnx 包装器
    'osmnx',
    'HAS_OSMNX',
    'OsmnxWrapper',
    # networkx 包装器
    'networkx',
    'HAS_NETWORKX',
    'NetworkXWrapper',
    # arcgis 包装器
    'arcgis',
    'HAS_ARCGIS',
    'ArcGISWrapper',
    # whitebox 包装器
    'whitebox',
    'HAS_WHITEBOX',
    'WhiteboxWrapper',
]
