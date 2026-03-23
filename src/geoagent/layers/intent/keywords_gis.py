"""
意图识别关键词 - 高级 GIS 分析
================================

定义 interpolation、viewshed、statistics 等高级 GIS 分析场景的意图关键词。
"""

from typing import List, Dict
from geoagent.layers.architecture import Scenario


# =============================================================================
# 插值/表面分析
# =============================================================================

INTERPOLATION_KEYWORDS: List[str] = [
    # 中文
    "插值", "interpolation", "IDW", "kriging", "空间插值",
    "离散点", "连续表面", "反距离加权", "克里金",
    "插值分析", "空间预测", "生成表面",
    # 英文
    "idw", "inverse distance", "spatial interpolation",
    "surface generation", "grid generation", "rasterize",
]


# =============================================================================
# 视域/阴影
# =============================================================================

VIEWSHED_KEYWORDS: List[str] = [
    # 中文
    "视域", "视域分析", "通视", "通视分析", "可见性",
    "阴影", "shadow", "日照", "遮挡", "采光",
    "可视范围", "可视区域", "视野范围", "视野分析",
    "建筑阴影", "日照分析", "阴影分析",
    # 英文
    "viewshed", "viewshed analysis", "visibility", "visible",
    "line of sight", "los", "viewpoint",
    "shadow analysis", "sun shadow", "sunlight", "solar access",
]


# =============================================================================
# 统计/聚合
# =============================================================================

STATISTICS_KEYWORDS: List[str] = [
    # 中文
    "热点", "hotspot", "选址", "mcda", "site selection",
    "冷热点", "冷热点分析", "莫兰指数", "morans i", "lisa",
    "getis-ord", "空间自相关", "空间聚集", "热点分析",
    "智能选址", "多准则决策", "适宜性",
    "统计", "聚合", "分区统计",
    # 英文
    "hot spot", "cold spot", "spatial autocorrelation",
    "moran", "lisa", "getis", "site selection", "mcda",
    "multi-criteria", "optimal location", "spatial clustering",
    "statistics", "aggregation", "zonal",
]


# =============================================================================
# 适宜性分析/选址
# =============================================================================

SUITABILITY_KEYWORDS: List[str] = [
    # 中文
    "适宜性", "适宜性分析", "适宜区", "suitability",
    "选址", "选址分析", "工厂选址", "仓库选址", "垃圾场选址",
    "垃圾场", "新建", "多准则", "mcda", "多目标",
    "加权叠加", "权重分析", "因素叠加",
    "新建垃圾场", "最佳位置", "最优位置", "合适位置",
    "条件选址", "约束选址", "避开", "远离",
    # 英文
    "suitability analysis", "site selection", "suitability analysis",
    "mcda", "multi-criteria decision", "weighted overlay",
    "optimal location", "best location", "land suitability",
    "garbage site", "waste facility", "landfill site",
]


# =============================================================================
# 栅格分析
# =============================================================================

RASTER_KEYWORDS: List[str] = [
    # 中文
    "ndvi", "植被", "植被指数", "ndwi", "evi", "水体指数",
    "遥感指数", "绿度指数", "归一化植被", "卫星影像指数",
    "计算ndvi", "计算植被",
    "坡度", "坡向", "dem", "高程",
    # 英文
    "ndvi", "ndwi", "evi", "vegetation index", "vegetation analysis",
    "satellite index", "remote sensing", "leaf area", "lai",
    "slope", "aspect", "dem", "elevation", "terrain",
]


# =============================================================================
# 导出合并的关键词
# =============================================================================

GIS_KEYWORDS: Dict[Scenario, List[str]] = {
    Scenario.INTERPOLATION: INTERPOLATION_KEYWORDS,
    Scenario.VIEWSHED: VIEWSHED_KEYWORDS,
    Scenario.SHADOW_ANALYSIS: VIEWSHED_KEYWORDS,  # 共享阴影关键词
    Scenario.STATISTICS: STATISTICS_KEYWORDS,
    Scenario.SUITABILITY: SUITABILITY_KEYWORDS,
    Scenario.RASTER: RASTER_KEYWORDS,
}
