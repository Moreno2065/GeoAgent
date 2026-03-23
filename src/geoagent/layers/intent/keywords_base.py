"""
意图识别关键词 - 基础 GIS 分析
================================

定义 route、buffer、overlay 等基础 GIS 分析场景的意图关键词。
"""

from typing import List, Dict
from geoagent.layers.architecture import Scenario


# =============================================================================
# 路径/可达性分析
# =============================================================================

ROUTE_KEYWORDS: List[str] = [
    # 中文
    "路径", "路线", "route", "步行", "导航", "最短路径", "寻路", "routing",
    "从...到", "到...的", "出发地", "目的地", "起点", "终点",
    "可达", "可达性", "等时圈", "服务范围", "通行时间", "出行时间",
    "15分钟", "10分钟", "5分钟", "30分钟生活圈",
    # 英文
    "driving", "walking", "walk", "drive", "bike", "cycling",
    "shortest path", "shortest route", "navigation", "directions",
    "origin", "destination", "from to", "accessibility", "isochrone",
    "service area", "travel time", "walking distance",
]


# =============================================================================
# 缓冲/邻近分析
# =============================================================================

BUFFER_KEYWORDS: List[str] = [
    # 中文
    "缓冲", "buffer", "缓冲区", "方圆",
    "500米范围", "1公里内", "xx米", "xx公里",
    # 英文
    "buffer zone", "buffering", "proximity", "within distance",
    "within radius", "buffer analysis",
]


# =============================================================================
# 叠置/裁剪分析
# =============================================================================

OVERLAY_KEYWORDS: List[str] = [
    # 中文
    "叠加", "overlay", "相交", "intersect", "合并", "union",
    "clip", "裁剪", "擦除", "difference", "对称差", "交集",
    "空间叠置", "空间分析", "叠图", "叠合", "选址分析",
    # 英文
    "spatial overlay", "intersection", "clipping", "erasing",
    "map overlay", "layer combination", "site selection",
]


# =============================================================================
# 导出合并的关键词
# =============================================================================

BASE_KEYWORDS: Dict[Scenario, List[str]] = {
    Scenario.ROUTE: ROUTE_KEYWORDS,
    Scenario.BUFFER: BUFFER_KEYWORDS,
    Scenario.OVERLAY: OVERLAY_KEYWORDS,
}
