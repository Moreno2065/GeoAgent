"""
GeoAgent 插件模块
"""

from geoagent.plugins.base import BasePlugin
from geoagent.plugins.amap_plugin import AmapPlugin
from geoagent.plugins.osm_plugin import OsmPlugin

__all__ = [
    "BasePlugin",
    "AmapPlugin",
    "OsmPlugin",
]
