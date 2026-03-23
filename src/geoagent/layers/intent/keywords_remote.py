"""
意图识别关键词 - 遥感分析
==============================

定义遥感分析相关场景的意图关键词。
"""

from typing import List, Dict
from geoagent.layers.architecture import Scenario


# =============================================================================
# 遥感分析
# =============================================================================

REMOTE_SENSING_KEYWORDS: List[str] = [
    # 通用遥感
    "遥感", "遥感分析", "remote sensing", "卫星影像",
    "sentinel", "landsat", "modis", "smodis",
    "遥感数据", "卫星数据", "影像处理",
]


# =============================================================================
# NDVI
# =============================================================================

NDVI_KEYWORDS: List[str] = [
    # 中文
    "ndvi", "植被指数", "归一化植被指数", "ndwi", "evi", "水体指数",
    "遥感指数", "绿度指数", "归一化植被", "卫星影像指数",
    "计算ndvi", "计算植被",
    # 英文
    "ndvi", "ndwi", "evi", "vegetation index", "vegetation analysis",
]


# =============================================================================
# 导出合并的关键词
# =============================================================================

REMOTE_KEYWORDS: Dict[Scenario, List[str]] = {
    Scenario.REMOTE_SENSING: REMOTE_SENSING_KEYWORDS,
    Scenario.NDVI: NDVI_KEYWORDS,
    Scenario.NDWI: NDVI_KEYWORDS,  # 共享
}
