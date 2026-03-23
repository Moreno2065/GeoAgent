"""
地形/三维域 (Terrain Domain)
============================
包含三维地形分析 Executor。

模块：
  - shadow_executor.py      : 视域/阴影分析
  - lidar_3d_executor.py    : LiDAR 三维分析
  - sun_position.py        : 太阳位置计算
"""

from geoagent.executors.domains.terrain.shadow_executor import ShadowExecutor
from geoagent.executors.domains.terrain.lidar_3d_executor import LiDAR3DExecutor
from geoagent.executors.domains.terrain.sun_position import calculate_sun_position

__all__ = [
    "ShadowExecutor",
    "LiDAR3DExecutor",
    "calculate_sun_position",
]
