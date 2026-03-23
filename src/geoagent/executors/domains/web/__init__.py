"""
Web 服务域 (Web Domain)
========================
包含高德/Amap Web 服务 Executor。

模块：
  - amap_executor.py       : 高德 Web API
  - overpass_executor.py   : OSM Overpass API
  - osm_executor.py        : OSMnx 下载
  - stac_search_executor.py : STAC 搜索
"""

from geoagent.executors.domains.web.amap_executor import AmapExecutor
from geoagent.executors.domains.web.overpass_executor import OverpassExecutor
from geoagent.executors.domains.web.osm_executor import OSMExecutor
from geoagent.executors.domains.web.stac_search_executor import STACSearchExecutor

__all__ = [
    "AmapExecutor",
    "OverpassExecutor",
    "OSMExecutor",
    "STACSearchExecutor",
]
