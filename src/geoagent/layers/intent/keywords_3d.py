"""
意图识别关键词 - 三维分析
==========================

定义 LiDAR/3D 分析相关场景的意图关键词。
"""

from typing import List, Dict
from geoagent.layers.architecture import Scenario


# =============================================================================
# 三维分析
# =============================================================================

LIDAR_3D_KEYWORDS: List[str] = [
    # 中文
    "三维", "3d", "lidar", "LiDAR", "点云",
    "体积", "volume", "剖面", "profile",
    "山体阴影", "hillshade", "坡度", "坡向",
    "粗糙度", "roughness", "曲率", "curvature",
    "流域", "watershed", "流向", "flow direction",
    "流量累积", "flow accumulation", "填挖方", "cut fill",
    # 英文
    "lidar", "3d analysis", "point cloud", "volume calculation",
]


# =============================================================================
# 导出合并的关键词
# =============================================================================

D3_KEYWORDS: Dict[Scenario, List[str]] = {
    Scenario.LIDAR_3D: LIDAR_3D_KEYWORDS,
    Scenario.VOLUME: LIDAR_3D_KEYWORDS,
    Scenario.PROFILE: LIDAR_3D_KEYWORDS,
    Scenario.HILLSHADE: LIDAR_3D_KEYWORDS,
    Scenario.ROUGHNESS: LIDAR_3D_KEYWORDS,
    Scenario.CURVATURE: LIDAR_3D_KEYWORDS,
    Scenario.WATERSHED: LIDAR_3D_KEYWORDS,
    Scenario.FLOW_DIRECTION: LIDAR_3D_KEYWORDS,
    Scenario.FLOW_ACCUMULATION: LIDAR_3D_KEYWORDS,
    Scenario.CUT_FILL: LIDAR_3D_KEYWORDS,
}
