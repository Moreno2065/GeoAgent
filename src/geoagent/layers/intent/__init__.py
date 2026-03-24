"""
意图识别关键词模块
=====================

将 INTENT_KEYWORDS 按功能域拆分到多个子模块：
- keywords_base.py   - 基础 GIS 分析（route/buffer/overlay）
- keywords_gis.py   - 高级 GIS 分析（interpolation/viewshed/statistics）
- keywords_web.py   - Web 服务（高德/Amap）
- keywords_remote.py - 遥感分析
- keywords_3d.py    - 三维分析

使用方式：
    from geoagent.layers.intent import INTENT_KEYWORDS
    
    # 或直接从子模块导入
    from geoagent.layers.intent.keywords_base import BASE_KEYWORDS
"""

from typing import Dict, List

from geoagent.layers.architecture import Scenario

# 导入各子模块的关键词
from geoagent.layers.intent.keywords_base import BASE_KEYWORDS
from geoagent.layers.intent.keywords_gis import GIS_KEYWORDS
from geoagent.layers.intent.keywords_web import WEB_KEYWORDS
from geoagent.layers.intent.keywords_remote import REMOTE_KEYWORDS
from geoagent.layers.intent.keywords_3d import D3_KEYWORDS


# =============================================================================
# 代码沙盒关键词
# =============================================================================

CODE_SANDBOX_KEYWORDS: List[str] = [
    # 中文 - 显式触发
    "写一段代码", "写python", "python代码", "写脚本",
    "生成测试数据", "代码实现", "计算面积", "编程", "用代码",
    "写代码", "帮我写", "python实现", "写个脚本",
    "生成数据", "用python", "script", "compute",
    "写个函数", "代码计算", "脚本", "计算一下",
    "代码生成", "python编程", "写段代码",
    # 中文 - 隐式/计算类触发（空间数学疑难杂症）
    "生成", "随机生成", "随机点", "面积计算", "长度计算", "距离计算",
    "统计", "算法", "提取坐标", "自定义公式", "加权求和",
    "坐标转换", "数学公式", "自定义逻辑", "迭代计算",
    "拟合", "插值自定义", "算一下", "帮我算", "帮我生成",
    "帮我统计", "批量处理", "循环处理",
    # 中文 - 几何计算类（圆、矩形、三角形等纯数学计算）
    "圆心", "半径", "交集面积", "并集面积", "弧长", "弦长",
    "三角形", "矩形", "正方形", "多边形", "椭圆",
    "周长", "对角线", "夹角", "相距", "重叠",
    "求解", "几何计算", "数学计算",
    # 英文 - 显式触发
    "write code", "python code", "write script", "generate test data",
    "code implementation", "compute area", "programming",
    "custom calculation", "custom logic",
    # 英文 - 隐式/计算类触发
    "generate random points", "random geometry", "compute area",
    "calculate distance", "custom formula", "custom algorithm",
    "iterative", "statistical analysis", "coordinate transformation",
    "run python", "execute code", "script execution",
]


# =============================================================================
# OSM 地图下载关键词
# =============================================================================

FETCH_OSM_KEYWORDS: List[str] = [
    # 中文 - 显式 OSM 关键词（必须包含 osm/openstreetmap）
    "osm下载", "用osm", "osm抓取", "下载osm", "osm数据", "osm地图",
    "openstreetmap下载", "获取osm", "下载路网", "下载建筑",
    "周边地图", "周围地图",
    # 🛡️ 移除宽泛的"下载"关键词，避免干扰 buffer/overlay 等场景
    # 中文 - 显式导出（必须有明确的方向）
    "把地图下载", "把数据下载", "把区域下载",
    # 英文 - 显式 OSM 关键词
    "osm download", "osm fetch", "fetch osm", "download osm",
    "osm data", "openstreetmap", "download buildings", "download network",
    "download roads",
    # 🛡️ 移除宽泛的"download"/"export"关键词，避免干扰其他场景
]


# =============================================================================
# Overpass API 关键词
# =============================================================================

OVERPASS_KEYWORDS: List[str] = [
    # 中文 - Overpass 显式关键词
    "overpass", "overpass api", "用overpass", "overpass查询",
    "overpass下载", "overpass抓取", "overpass获取",
    "osmid", "osm id",
    # 矩形区域查询
    "bbox查询", "矩形查询", "范围查询osm", "经纬度范围下载",
    "31.23", "121.48", "坐标范围",
    # 英文 - Overpass 显式关键词
    "overpass query", "query overpass", "overpass download",
    "bbox query", "rectangle query", "bounding box",
]


# =============================================================================
# 合并所有关键词（按 Scenario）
# =============================================================================

INTENT_KEYWORDS: Dict[Scenario, List[str]] = {
    # 基础 GIS 分析
    **{k: v for k, v in BASE_KEYWORDS.items()},
    # 高级 GIS 分析
    **{k: v for k, v in GIS_KEYWORDS.items()},
    # Web 服务
    **{k: v for k, v in WEB_KEYWORDS.items()},
    # 遥感分析
    **{k: v for k, v in REMOTE_KEYWORDS.items()},
    # 三维分析
    **{k: v for k, v in D3_KEYWORDS.items()},
    # 代码沙盒
    Scenario.CODE_SANDBOX: CODE_SANDBOX_KEYWORDS,
    # OSM 地图下载
    Scenario.FETCH_OSM: FETCH_OSM_KEYWORDS,
    # Overpass API
    Scenario.OVERPASS: OVERPASS_KEYWORDS,
}


__all__ = [
    "INTENT_KEYWORDS",
    # 子模块导出
    "BASE_KEYWORDS",
    "GIS_KEYWORDS",
    "WEB_KEYWORDS",
    "REMOTE_KEYWORDS",
    "D3_KEYWORDS",
    # 独立关键词列表（用于特定场景）
    "CODE_SANDBOX_KEYWORDS",
    "FETCH_OSM_KEYWORDS",
    "OVERPASS_KEYWORDS",
]
