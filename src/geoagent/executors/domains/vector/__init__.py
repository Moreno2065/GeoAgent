"""
矢量分析域 (Vector Domain)
===========================
包含基础矢量空间分析 Executor。

模块：
  - route_executor.py    : 路径/可达性分析
  - buffer_executor.py   : 缓冲区分析
  - overlay_executor.py  : 叠置分析
  - idw_executor.py     : IDW 插值
  - hotspot_executor.py  : 热点分析
  - suitability_executor.py : 适宜性分析
"""

from geoagent.executors.domains.vector.route_executor import RouteExecutor
from geoagent.executors.domains.vector.buffer_executor import BufferExecutor
from geoagent.executors.domains.vector.overlay_executor import OverlayExecutor
from geoagent.executors.domains.vector.idw_executor import IDWExecutor
from geoagent.executors.domains.vector.hotspot_executor import HotspotExecutor
from geoagent.executors.domains.vector.suitability_executor import SuitabilityExecutor

__all__ = [
    "RouteExecutor",
    "BufferExecutor",
    "OverlayExecutor",
    "IDWExecutor",
    "HotspotExecutor",
    "SuitabilityExecutor",
]
