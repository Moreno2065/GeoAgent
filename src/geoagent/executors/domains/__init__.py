"""
Executor 功能域 (Executor Domains)
===================================
按功能域组织 Executor 模块：

├── vector/     : 基础矢量分析 (route, buffer, overlay, idw, hotspot, suitability)
├── terrain/     : 三维地形分析 (shadow, lidar_3d, sun_position)
├── web/         : Web 服务 (amap, overpass, osm, stac)
├── remote/      : 遥感分析 (ndvi, remote_sensing)
├── viz/         : 可视化 (visualization)
└── core/        : 核心/通用 (general, gdal, postgis, sandbox, arcgis)

使用方式：
    from geoagent.executors.domains.vector import RouteExecutor
    from geoagent.executors.domains.terrain import ShadowExecutor
"""

from geoagent.executors.domains.vector import (
    RouteExecutor,
    BufferExecutor,
    OverlayExecutor,
    IDWExecutor,
    HotspotExecutor,
    SuitabilityExecutor,
)
from geoagent.executors.domains.terrain import (
    ShadowExecutor,
    LiDAR3DExecutor,
    calculate_sun_position,
)
from geoagent.executors.domains.web import (
    AmapExecutor,
    OverpassExecutor,
    OSMExecutor,
    STACSearchExecutor,
)
from geoagent.executors.domains.remote import (
    NdviExecutor,
    RemoteSensingExecutor,
    RemoteSensingIndex,
    BandMapping,
)
from geoagent.executors.domains.viz import VisualizationExecutor
from geoagent.executors.domains.core import (
    GeneralExecutor,
    GDALExecutor,
    PostGISExecutor,
    CodeSandboxExecutor,
    ArcGISExecutor,
    WorkflowEngine,
)

__all__ = [
    # Vector Domain
    "RouteExecutor",
    "BufferExecutor",
    "OverlayExecutor",
    "IDWExecutor",
    "HotspotExecutor",
    "SuitabilityExecutor",
    # Terrain Domain
    "ShadowExecutor",
    "LiDAR3DExecutor",
    "calculate_sun_position",
    # Web Domain
    "AmapExecutor",
    "OverpassExecutor",
    "OSMExecutor",
    "STACSearchExecutor",
    # Remote Domain
    "NdviExecutor",
    "RemoteSensingExecutor",
    "RemoteSensingIndex",
    "BandMapping",
    # Viz Domain
    "VisualizationExecutor",
    # Core Domain
    "GeneralExecutor",
    "GDALExecutor",
    "PostGISExecutor",
    "CodeSandboxExecutor",
    "ArcGISExecutor",
    "WorkflowEngine",
]
