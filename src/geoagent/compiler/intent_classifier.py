"""
Intent Classifier - 意图分类器
============================
基于关键词的第一层意图分类，稳定高效。

核心设计：
- 不依赖 LLM，用关键词匹配实现意图分类
- 返回固定枚举，不让 LLM 自由发挥
- 支持多意图识别（返回 Set）
"""

from __future__ import annotations

import re
from typing import Literal, Set, Optional, List, Dict, Any
from dataclasses import dataclass


# =============================================================================
# 意图关键词配置
# =============================================================================

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "route": [
        # 中文
        "路径", "route", "步行", "导航", "最短路径", "寻路", "routing",
        "从...到", "到...的", "出发地", "目的地", "起点", "终点",
        "到", "前往", "去", "怎么走", "路线",
        # 英文
        "driving", "walking", "walk", "drive", "bike", "cycling",
        "shortest path", "shortest route", "navigation", "directions",
        "origin", "destination", "from to", "how to get", "directions",
    ],
    "buffer": [
        # 中文 — 仅限明确包含"缓冲"或"缓冲区"的场景
        "缓冲", "buffer", "缓冲区", "生成缓冲区", "缓冲区分析",
        "做缓冲区", "做个缓冲区", "缓冲分析",
        # 英文
        "buffer zone", "buffering", "buffer analysis",
    ],
    "overlay": [
        # 中文
        "叠加", "overlay", "相交", "intersect", "合并", "union",
        "clip", "裁剪", "擦除", "difference", "对称差", "交集",
        "空间叠置", "空间分析", "叠图", "叠合",
        # 英文
        "spatial overlay", "intersection", "clipping", "erasing",
        "map overlay", "layer combination",
    ],
    "interpolation": [
        # 中文
        "插值", "interpolation", "IDW", "kriging", "空间插值",
        "离散点", "连续表面", "反距离加权", "克里金",
        "插值分析", "空间预测",
        # 英文
        "idw", "inverse distance", "spatial interpolation",
        "surface generation", "grid generation", "rasterize",
    ],
    "shadow_analysis": [
        # 中文
        "阴影", "shadow", "日照", "遮挡", "采光", "太阳阴影",
        "建筑阴影", "日照分析", "阴影分析",
        # 英文
        "shadow analysis", "sun shadow", "sunlight", "solar access",
        "building shadow", "shadow calculation", "shadow geometry",
    ],
    "ndvi": [
        # 中文
        "ndvi", "植被", "植被指数", "ndwi", "evi", "水体指数",
        "遥感指数", "绿度指数", "归一化植被", "卫星影像指数",
        "计算ndvi", "计算植被",
        # 英文
        "ndvi", "ndwi", "evi", "vegetation index", "vegetation analysis",
        "satellite index", "remote sensing", "leaf area", "lai",
    ],
    "hotspot": [
        # 中文
        "热点", "hotspot", "选址", "mcda", "site selection",
        "冷热点", "冷热点分析", "莫兰指数", "morans i", "lisa",
        "getis-ord", "空间自相关", "空间聚集", "热点分析",
        "智能选址", "多准则决策",
        # 英文
        "hot spot", "cold spot", "spatial autocorrelation",
        "moran", "lisa", "getis", "site selection", "mcda",
        "multi-criteria", "optimal location", "spatial clustering",
    ],
    "visualization": [
        # 中文
        "可视化", "可视化地图", "3d地图", "交互地图",
        "folium", "热力图", "choropleth", "出图", "渲染",
        "绘制地图", "展示地图", "web地图", "html地图",
        # 英文
        "visualization", "3d map", "interactive map",
        "folium", "heatmap", "choropleth", "render", "plot map",
        "web map", "html map", "deck.gl", "pydeck",
    ],
    # 新增场景
    "accessibility": [
        # 中文
        "可达性", "可达", "等时圈", "服务范围", "步行可达",
        "时间距离", "出行时间", "通行时间", "到达时间",
        "15分钟", "10分钟", "5分钟", "30分钟生活圈",
        "辐射范围", "覆盖范围", "服务半径",
        # 英文
        "accessibility", "accessible", "isochrone", "travel time",
        "walking distance", "service area", "service radius",
        "coverage", "reach", "10-minute", "15-minute city",
        "walkability", "walk score",
    ],
    "suitability": [
        # 中文
        "选址", "适宜性", "适宜区", "适宜性分析", "选址分析",
        "适合", "合适位置", "最佳位置", "最优位置",
        "多准则", "加权分析", "权重", "因素叠加",
        "开店", "选址", "工厂选址", "仓库选址",
        # 英文
        "suitability", "site selection", "site selection analysis",
        "location selection", "optimal location", "best location",
        "multi-criteria", "weighted overlay", "mcda",
        "site suitability", "suitable area", "suitability analysis",
        "land suitability",
    ],
    # ── 🟣 多条件综合搜索（联网推理）───────────────────────────────
    "multi_criteria_search": [
        # 中文 — 复合条件搜索（简单匹配词）
        "找一个", "找一下", "帮我找", "搜索", "查找",
        "附近", "周边", "附近找", "附近有",
        "摸鱼", "摸鱼地点", "适合摸鱼",
        "距离星巴克", "距离地铁", "距离小于", "距离大于",
        "附近.*推荐", "推荐.*附近",
        # 英文
        "find me a", "find a", "search for", "look for",
        "nearby", "nearest", "close to",
        "site selection", "optimal location", "best spot",
    ],
    "viewshed": [
        # 中文
        "视域", "视域分析", "通视", "通视分析", "可见性",
        "观察点", "视线分析", "视野", "观景点",
        "建筑物高度", "遮挡分析", "信号覆盖",
        # 英文
        "viewshed", "viewshed analysis", "visibility", "visible",
        "line of sight", "los", "viewpoint", "visual impact",
        "observer point", "tower placement", "telecom coverage",
    ],
    # ── 🟣 代码沙盒（受限代码执行）────────────────────────────────
    "code_sandbox": [
        # 中文 — 显式触发
        "写一段代码", "写python", "python代码", "写脚本",
        "生成测试数据", "代码实现", "计算面积", "编程", "用代码",
        "写代码", "帮我写", "python实现", "写个脚本",
        "生成数据", "用python", "script", "compute",
        "写个函数", "代码计算", "脚本", "计算一下",
        "代码生成", "python编程", "写段代码",
        # 中文 — 隐式/计算类触发（空间数学疑难杂症）
        # 🛡️ 移除过于通用的"生成"，避免误捕获"生成地图"等GIS意图
        "随机生成", "随机点", "面积计算", "长度计算", "距离计算",
        "统计", "算法", "提取坐标", "自定义公式", "加权", "加权求和",
        "坐标转换", "数学公式", "自定义逻辑", "迭代计算",
        "拟合", "插值自定义", "算一下", "帮我算", "帮我生成",
        "帮我统计", "批量处理", "循环处理", "得分",
        # 英文 — 显式触发
        "write code", "python code", "write script", "generate test data",
        "code implementation", "compute area", "programming",
        "custom calculation", "custom logic",
        # 英文 — 隐式/计算类触发
        "generate random points", "random geometry", "compute area",
        "calculate distance", "custom formula", "custom algorithm",
        "iterative", "statistical analysis", "coordinate transformation",
        "run python", "execute code", "script execution",
    ],
    # ── 🟣 OSM 地图下载 ───────────────────────────────────────────────
    "fetch_osm": [
        # 中文 — 显式关键词
        "osm下载", "用osm", "osm抓取", "下载osm", "osm数据", "osm地图",
        "openstreetmap下载", "获取osm", "osm周围", "osm地图下载",
        "openstreetmap", "open street map",  # 英文 OSM 全称
        # 扩展：更口语化的表达
        "osm范围", "osm区域", "osm附近", "osm周边",
        # 英文扩展
        "osm", "osm fetch", "osm download", "fetch osm", "download osm",
        "osm data", "download map", "fetch map",
        "download buildings", "download network", "download roads",
        "osm map", "osm area",
        # ── 🆕 隐式下载意图（无需显式 "osm" 关键词）────────────────────
        # 核心下载动作词
        "下载地图", "下载周边", "下载周围", "下载附近", "下载这个区域",
        "下载天安门", "下载复旦大学", "下载上海", "下载北京",
        "抓取地图", "抓取周边", "抓取周围", "抓取附近",
        "获取地图", "获取周边", "获取周围", "获取附近",
        # 下载 + 地点 + 地图
        "下载天安门地图", "下载故宫地图", "下载长城地图",
        "下载上海地图", "下载北京地图", "下载广州地图",
        "下载复旦大学周边地图", "下载交大附近地图",
        # 通用下载请求
        "下载这个地方的地图", "下载那个区域的地图",
        "下载这块区域的", "下载这片区域",
        # 生成/显示/展示地图（隐式下载）
        "生成地图", "生成周边地图", "生成周围地图", "生成附近地图",
        "显示地图", "显示周边", "显示周围", "显示附近",
        "展示地图", "展示周边", "展示周围", "展示附近",
        "看看这里", "看看这里周边", "看看这里附近",
        "看看那里", "看看那周边", "看看那附近",
        # 看看 + 地点
        "看看天安门", "看看故宫", "看看长城",
        "看看复旦大学", "看看上海", "看看琶洲",
        # 给我看 + 地点
        "给我看天安门", "给我看故宫", "给我看这里",
        # 高优先级数字 + 米/公里 + 地图（优先于 buffer）
        "五百米地图", "500米地图", "1公里地图", "1000米地图",
        "五百米osm", "500mosm", "1公里osm",
        "米范围内的地图", "米范围地图", "米地图",
        # 周边/周围/附近 + 地图（无明确中心点时用当前位置或上次位置）
        "周边地图", "周围地图", "附近地图",
        "周边osm地图", "周围osm地图", "附近osm地图",
        # ── 🆕 泛化隐式下载：任意"下载"或"生成" + 地名
        # 下载/生成 + 任意地标/POI（不限定具体地点）
        "下载", "下载地图", "下载周边", "下载周围", "下载附近",
        # 🛡️ 移除单独的"生成"以避免误捕获"生成缓冲区"等请求
        "生成地图", "生成周边", "生成周围", "生成附近",
        "查看", "查看地图", "查看周边", "查看周围", "查看附近",
        "显示", "显示地图", "显示周边", "显示周围", "显示附近",
        "展示", "展示地图", "展示周边", "展示周围", "展示附近",
        "看看", "看看地图", "看看周边", "看看周围", "看看附近",
        "获取", "获取地图", "获取周边", "获取周围", "获取附近",
        "抓取", "抓取地图", "抓取周边", "抓取周围", "抓取附近",
        # 任意动作 + 地名（POI/地标自动触发）
        "到", "去", "前往",  # "到天安门"、"去复旦大学"等隐含需要地图
        # 地名类关键词（可与上面的动作词组合）
        "天安门", "故宫", "长城", "天坛", "颐和园",
        "复旦大学", "交通大学", "华东师范大学", "同济大学",
        "上海", "北京", "广州", "深圳", "杭州", "成都",
        "琶洲", "珠江新城", "陆家嘴", "外滩",
        "武汉", "黄鹤楼", "珞珈山", "光谷",
        "内蒙古", "包头", "呼和浩特",
        "首钢", "鞍钢", "马钢", "宝钢",
        "师范大学", "理工大学", "工业大学",
        # 无动作的纯地名（默认需要地图）
        "附近", "周边", "周围",
    ],
    # ── 🟢 坐标转换 ─────────────────────────────────────────────────
    "coord_convert": [
        # 中文关键词
        "坐标转换", "坐标转", "转换坐标", "坐标变换",
        "gcj02", "wgs84", "bd09", "bd-09",
        "高德坐标", "谷歌坐标", "百度坐标", "gps坐标",
        "图吧坐标", "坐标系统转换", "坐标系转换",
        "投影转换", "投影变换", "地图投影",
        "反投影", "逆投影",
        # 投影类型
        "web mercator", "墨卡托", "球面墨卡托",
        "utm", "通用横轴墨卡托", "分带",
        "epsg:3857", "epsg:4326",
        # 显式转换动作
        "转换为", "转成", "转换成", "转换到",
        "投影到", "转换到",
        # 英文关键词
        "coordinate conversion", "coordinate transform",
        "convert coordinates", "projection conversion",
        "mercator", "wgs84 to", "to gcj", "to wgs84",
        "from wgs84", "from gcj", "to mercator",
        "reproject", "reprojection", "crs transform",
        "zone", "utm zone",
    ],
}


# =============================================================================
# 意图分类结果
# =============================================================================

@dataclass
class IntentResult:
    """意图分类结果"""
    primary: str  # 主要意图
    confidence: float  # 置信度 0-1
    matched_keywords: List[str]  # 匹配的关键词
    all_intents: Set[str]  # 所有识别到的意图

    def __str__(self) -> str:
        return f"IntentResult(primary={self.primary}, confidence={self.confidence:.2f}, intents={self.all_intents})"


# =============================================================================
# 意图分类器
# =============================================================================

class IntentClassifier:
    """
    意图分类器

    基于关键词的意图分类，稳定高效。
    支持单意图和多意图识别。

    使用方式：
        classifier = IntentClassifier()
        result = classifier.classify("芜湖南站到方特的步行路径")
        print(result.primary)  # "route"
        print(result.confidence)  # 0.95
    """

    def __init__(self, threshold: float = 0.0):
        """
        初始化意图分类器

        Args:
            threshold: 置信度阈值，低于此值则返回 "general"
        """
        self.threshold = threshold
        self._build_index()

    def _build_index(self) -> None:
        """构建关键词索引以加速匹配"""
        # 关键词 -> 意图的映射
        self._keyword_to_intent: Dict[str, List[str]] = {}
        # 意图 -> 关键词列表
        self._intent_keywords: Dict[str, List[str]] = {}

        for intent, keywords in INTENT_KEYWORDS.items():
            self._intent_keywords[intent] = keywords
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower not in self._keyword_to_intent:
                    self._keyword_to_intent[kw_lower] = []
                self._keyword_to_intent[kw_lower].append(intent)

    def classify(self, query: str, multi: bool = False) -> IntentResult:
        """
        对用户 query 进行意图分类

        Args:
            query: 用户输入的自然语言查询
            multi: 是否返回多意图（True）或仅返回主要意图（False）

        Returns:
            IntentResult 对象
        """
        if not query or not query.strip():
            return IntentResult(
                primary="general",
                confidence=0.0,
                matched_keywords=[],
                all_intents={"general"}
            )

        # ==========================================
        # 🚨 前置拦截器 (Early Exit Pattern)
        # ==========================================
        sandbox_dictators = ["用代码", "沙盒", "写一段代码", "写脚本", "写python", "代码算"]

        if any(trigger in query for trigger in sandbox_dictators):
            print("[SANDBOX INTERCEPTOR] Detected code sandbox trigger, routing to 'code_sandbox'")
            return IntentResult(
                primary="code_sandbox",
                confidence=1.0,
                matched_keywords=["显式编程触发词"],
                all_intents={"code_sandbox"}
            )

        # ── 🆕 BUFFER 拦截器（优先级高于 fetch_osm）────────────────────
        # 如果 query 中包含"缓冲"或"缓冲区"，强制路由到 buffer
        buffer_triggers = ["缓冲", "缓冲区"]
        if any(trigger in query for trigger in buffer_triggers):
            print("[BUFFER INTERCEPTOR] Detected buffer trigger, routing to 'buffer'")
            return IntentResult(
                primary="buffer",
                confidence=1.0,
                matched_keywords=["缓冲区触发词"],
                all_intents={"buffer"}
            )
        # ==========================================

        query_lower = query.lower()
        matched: Dict[str, List[str]] = {}  # intent -> matched keywords

        # 精确匹配
        for kw, intents in self._keyword_to_intent.items():
            if kw in query_lower:
                for intent in intents:
                    if intent not in matched:
                        matched[intent] = []
                    matched[intent].append(kw)

        # 如果没有精确匹配，尝试分词匹配
        if not matched:
            words = re.split(r'[\s,，、。.;:/\\]+', query_lower)
            for word in words:
                word = word.strip()
                if len(word) < 2:
                    continue
                for kw, intents in self._keyword_to_intent.items():
                    if len(kw) >= 2 and (kw in word or word in kw):
                        for intent in intents:
                            if intent not in matched:
                                matched[intent] = []
                            matched[intent].append(kw)

        if not matched:
            return IntentResult(
                primary="general",
                confidence=0.0,
                matched_keywords=[],
                all_intents={"general"}
            )

        # 计算每个意图的得分
        scores: Dict[str, float] = {}
        for intent, keywords in matched.items():
            # 得分计算（改进版）：
            # 1. 绝对字符得分：所有匹配关键词的字符总数
            # 2. 数量得分：匹配关键词数量
            # 3. 平均长度奖励：鼓励更长的关键词匹配
            n_matched = len(keywords)
            total_kw_chars = sum(len(kw) for kw in keywords)
            avg_kw_length = total_kw_chars / n_matched if n_matched > 0 else 0
            
            # 绝对得分（主要）：匹配关键词的字符总数
            abs_char_score = total_kw_chars
            # 数量得分（次要）：匹配关键词数量
            count_score = n_matched * 2  # 乘以2平衡权重
            # 长关键词额外奖励（超过5字符的关键词）
            long_kw_bonus = sum(1 for kw in keywords if len(kw) >= 5)
            
            # 综合得分
            scores[intent] = abs_char_score + count_score + long_kw_bonus * 2

        # 按得分排序
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 置信度：使用 sigmoid 变换使得分更合理
        # 新的绝对得分：5分左右约0.5，10分约0.9，15分接近1.0
        def _sigmoid(x):
            return 1 / (1 + 2 ** (-x / 5))
        raw_score = sorted_intents[0][1] if sorted_intents else 0
        confidence = _sigmoid(raw_score)

        if multi:
            all_intents = {intent for intent, _ in sorted_intents}
        else:
            # 单意图模式：返回得分最高的意图
            all_intents = {sorted_intents[0][0]} if sorted_intents else {"general"}

        # 检查是否低于阈值
        if confidence < self.threshold:
            return IntentResult(
                primary="general",
                confidence=confidence,
                matched_keywords=[],
                all_intents={"general"}
            )

        primary_intent = sorted_intents[0][0] if sorted_intents else "general"
        primary_keywords = matched.get(primary_intent, [])

        return IntentResult(
            primary=primary_intent,
            confidence=confidence,
            matched_keywords=primary_keywords,
            all_intents=all_intents
        )

    def classify_simple(self, query: str) -> str:
        """
        简单分类：直接返回意图字符串

        Args:
            query: 用户输入

        Returns:
            意图类型字符串
        """
        result = self.classify(query)
        return result.primary


# =============================================================================
# 便捷函数
# =============================================================================

# 默认分类器实例（延迟初始化）
_classifier: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    """获取默认分类器实例"""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def classify_intent(query: str, multi: bool = False) -> IntentResult:
    """
    便捷函数：对用户 query 进行意图分类

    Args:
        query: 用户输入的自然语言查询
        multi: 是否返回多意图

    Returns:
        IntentResult 对象
    """
    classifier = get_classifier()
    return classifier.classify(query, multi=multi)


def classify_intent_simple(query: str) -> str:
    """
    便捷函数：直接返回意图字符串

    Args:
        query: 用户输入

    Returns:
        意图类型字符串
    """
    result = classify_intent(query)
    return result.primary


# =============================================================================
# 意图路由映射（用于动态 schema 加载）
# =============================================================================

# 意图 -> 任务类型的映射
INTENT_TO_TASK_TYPE: Dict[str, str] = {
    "route": "route",
    "buffer": "buffer",
    "overlay": "overlay",
    "interpolation": "interpolation",
    "shadow_analysis": "shadow_analysis",
    "ndvi": "ndvi",
    "hotspot": "hotspot",
    "visualization": "visualization",
    "accessibility": "accessibility",
    "suitability": "suitability",
    "viewshed": "viewshed",
    "multi_criteria_search": "multi_criteria_search",
    "fetch_osm": "fetch_osm",
    "coord_convert": "coord_convert",
    "general": "general",
}


def get_task_type_for_intent(intent: str) -> str:
    """
    获取意图对应的任务类型

    Args:
        intent: 意图类型

    Returns:
        任务类型字符串
    """
    return INTENT_TO_TASK_TYPE.get(intent, "general")


# =============================================================================
# 场景子类型定义
# =============================================================================

SCENARIO_SUBTYPES: Dict[str, Dict[str, List[str]]] = {
    "route": {
        "walking": ["步行", "走路", "徒步", "walk", "walking"],
        "driving": ["开车", "驾车", "自驾", "drive", "driving"],
        "transit": ["公交", "地铁", "公共交通", "transit", "bus", "metro"],
        "cycling": ["骑行", "骑车", "自行车", "bike", "cycling"],
    },
    "buffer": {
        "point": ["周边", "附近", "方圆", "周围", "nearby"],
        "line": ["沿线", "走廊", "廊道", "corridor", "along"],
        "polygon": ["内部", "范围内", "within", "inside"],
    },
    "accessibility": {
        "isochrone": ["等时圈", "等时线", "isochrone", "isodistance"],
        "service_area": ["服务范围", "服务区", "service area"],
        "walkability": ["步行可达", "步行指数", "walkability"],
    },
    "overlay": {
        "intersect": ["相交", "交集", "intersect", "intersection"],
        "union": ["合并", "并集", "union", "merge"],
        "clip": ["裁剪", "clip"],
        "difference": ["擦除", "差集", "difference"],
    },
    "suitability": {
        "mcda": ["多准则", "mcda", "多目标", "multi-criteria"],
        "weighted": ["加权", "权重", "weighted overlay"],
        "site_selection": ["选址", "选位置", "site selection"],
    },
    "multi_criteria_search": {
        "poi_search": ["附近", "周边", "找", "搜索", "nearby", "near"],
        "distance_filter": ["距离", "小于", "大于", "within", "beyond"],
        "comparison": ["最近", "最远", "哪个", "nearest", "farthest"],
    },
}


# =============================================================================
# 澄清引擎
# =============================================================================

# 澄清模板：为每个场景定义缺失参数的追问话术
CLARIFICATION_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "route": {
        "start": {
            "question": "请问起点位置是哪里？",
            "options": None,
            "required": True,
            "examples": ["芜湖南站", "市中心", "我的位置"],
        },
        "end": {
            "question": "请问终点位置是哪里？",
            "options": None,
            "required": True,
            "examples": ["方特欢乐世界", "高铁站", "商场"],
        },
        "mode": {
            "question": "请问出行方式是？",
            "options": ["步行", "驾车", "公交", "骑行"],
            "required": False,
            "default": "步行",
        },
        "city": {
            "question": "请问在哪个城市？",
            "options": None,
            "required": False,
            "default": "自动检测",
        },
    },
    "buffer": {
        "input_layer": {
            "question": "请问要对哪个图层做缓冲分析？",
            "options": None,
            "required": True,
            "examples": ["学校", "地铁站", "河流"],
        },
        "distance": {
            "question": "请问缓冲半径是多少米？",
            "options": ["500米", "1公里", "2公里", "5公里"],
            "required": True,
            "examples": ["500", "1000", "2000"],
        },
        "unit": {
            "question": "请问距离单位是？",
            "options": ["米", "公里"],
            "required": False,
            "default": "meters",
        },
    },
    "overlay": {
        "layer1": {
            "question": "请选择第一个图层？",
            "options": None,
            "required": True,
            "examples": ["土地利用", "行政区划", "道路"],
        },
        "layer2": {
            "question": "请选择第二个图层？",
            "options": None,
            "required": True,
            "examples": ["洪涝区", "保护区", "商业区"],
        },
        "operation": {
            "question": "请问要进行什么叠加操作？",
            "options": ["交集", "并集", "裁剪", "差集"],
            "required": False,
            "default": "intersect",
        },
    },
    "interpolation": {
        "input_points": {
            "question": "请提供包含采样点的数据文件？",
            "options": None,
            "required": True,
            "examples": ["监测站.csv", "采样点.geojson"],
        },
        "value_field": {
            "question": "请指定用于插值的数值字段？",
            "options": None,
            "required": True,
            "examples": ["PM2.5", "温度", "降水", "浓度"],
        },
        "method": {
            "question": "请问使用什么插值方法？",
            "options": ["IDW（反距离加权）", "克里金", "最近邻"],
            "required": False,
            "default": "IDW",
        },
        "output_resolution": {
            "question": "请问输出栅格分辨率是多少米？",
            "options": ["100米", "50米", "200米"],
            "required": False,
            "default": "100",
        },
    },
    "accessibility": {
        "location": {
            "question": "请问分析的中心位置是哪里？",
            "options": None,
            "required": True,
            "examples": ["芜湖南站", "市中心", "某个地铁站"],
        },
        "mode": {
            "question": "请问用什么交通方式？",
            "options": ["步行", "驾车", "骑行"],
            "required": False,
            "default": "walking",
        },
        "time_threshold": {
            "question": "请问分析多长时间的可达范围？",
            "options": ["5分钟", "10分钟", "15分钟", "30分钟"],
            "required": True,
            "examples": ["5", "10", "15", "30"],
        },
    },
    "suitability": {
        "criteria_layers": {
            "question": "请提供参与选址分析的图层（多个）？",
            "options": None,
            "required": True,
            "examples": ["人口密度", "交通便利度", "租金", "竞争度"],
        },
        "weights": {
            "question": "请为各因素指定权重（总和应为100）？",
            "options": None,
            "required": False,
            "default": "等权重",
        },
        "area": {
            "question": "请指定分析区域的边界？",
            "options": None,
            "required": True,
            "examples": ["芜湖市", "某个区", "框选区域"],
        },
    },
    "viewshed": {
        "location": {
            "question": "请问观察点位置在哪里？",
            "options": None,
            "required": True,
            "examples": ["某楼顶", "某山顶", "某观景点"],
        },
        "observer_height": {
            "question": "请问观察点高度是多少米？",
            "options": None,
            "required": False,
            "default": "1.7",
        },
        "target_layer": {
            "question": "请指定要分析的视线目标区域？",
            "options": None,
            "required": False,
            "examples": ["全区域", "某建筑物", "某范围"],
        },
    },
    "shadow_analysis": {
        "buildings": {
            "question": "请提供建筑物数据文件？",
            "options": None,
            "required": True,
            "examples": ["建筑物.shp", "建筑物.geojson"],
        },
        "time": {
            "question": "请问分析哪个时间点的阴影？",
            "options": None,
            "required": True,
            "examples": ["2026-03-21 09:00", "2026-03-21 12:00", "2026-03-21 15:00"],
        },
    },
    "multi_criteria_search": {
        "center": {
            "question": "请问搜索的中心位置是哪里？",
            "options": None,
            "required": False,
            "default": "",
            "examples": ["天河体育中心", "琶洲", "珠江新城"],
        },
        "criteria": {
            "question": "请描述搜索条件？（如：距离星巴克小于200米，距离地铁站大于500米）",
            "options": None,
            "required": False,
            "default": "",
        },
        "radius": {
            "question": "请问搜索半径是多少？",
            "options": ["1公里", "2公里", "3公里", "5公里"],
            "required": False,
            "default": "3公里",
        },
    },
    "ndvi": {
        "input_file": {
            "question": "请提供遥感影像文件？",
            "options": None,
            "required": True,
            "examples": ["Landsat.tif", "Sentinel2.tif", "卫星影像"],
        },
        "sensor": {
            "question": "请问影像是什么传感器类型？",
            "options": ["Sentinel-2", "Landsat 8", "Landsat 9", "自动检测"],
            "required": False,
            "default": "auto",
        },
    },
    "hotspot": {
        "input_file": {
            "question": "请提供要分析的数据文件？",
            "options": None,
            "required": True,
            "examples": ["房价.shp", "销售点.geojson"],
        },
        "value_field": {
            "question": "请指定要分析的数值字段？",
            "options": None,
            "required": True,
            "examples": ["价格", "销量", "人口"],
        },
            "analysis_type": {
            "question": "请问使用什么分析方法？",
            "options": ["自动选择", "Getis-Ord Gi*（热点分析）", "Moran's I（全局自相关）"],
            "required": False,
            "default": "auto",
        },
    },
    "visualization": {
        "input_files": {
            "question": "请提供要可视化的数据文件？",
            "options": None,
            "required": True,
            "examples": ["建筑物.shp", "道路.geojson", "人口密度.tif"],
        },
        "viz_type": {
            "question": "请问要生成什么类型的可视化？",
            "options": ["交互式地图", "静态专题图", "3D地图", "热力图"],
            "required": False,
            "default": "interactive_map",
        },
        "color_column": {
            "question": "请问用什么字段着色？",
            "options": None,
            "required": False,
            "examples": ["人口", "房价", "密度"],
        },
        "height_column": {
            "question": "请问用什么字段作为3D高度？",
            "options": None,
            "required": False,
            "examples": ["高度", "楼层", "building_height"],
        },
    },
    # ── 🟣 OSM 地图下载 ────────────────────────────────────────────────
    "fetch_osm": {
        "center_point": {
            "question": "请问要下载哪个地点周围的地图？（输入地址或地标名称）",
            "options": None,
            "required": True,
            "examples": ["武汉黄鹤楼", "北京天安门", "上海外滩"],
            "step": 1,
        },
        "radius": {
            "question": "请问下载范围半径是多少米？",
            "options": ["500米", "1000米", "2000米", "3000米", "5000米"],
            "required": False,
            "default": 1000,
            "examples": ["500", "1000", "2000"],
            "step": 2,
        },
        "data_type": {
            "question": "请问要下载什么类型的数据？",
            "options": ["路网和建筑物", "仅路网", "仅建筑物"],
            "required": False,
            "default": "all",
            "step": 3,
        },
        "network_type": {
            "question": "请问路网的类型是？",
            "options": ["步行网络", "车行网络", "骑行网络", "所有道路"],
            "required": False,
            "default": "walk",
            "step": 3,
        },
    },
    # ── 🟢 坐标转换 ───────────────────────────────────────────────────
    "coord_convert": {
        "coordinates": {
            "question": "请输入要转换的坐标（如：116.404,39.915）",
            "options": None,
            "required": True,
            "examples": ["116.404,39.915", "116.410,39.921", "120,30"],
            "step": 1,
        },
        "from_crs": {
            "question": "请选择源坐标系",
            "options": ["WGS84", "GCJ-02", "BD-09", "EPSG:3857", "EPSG:4326"],
            "required": True,
            "default": "WGS84",
            "step": 2,
        },
        "to_crs": {
            "question": "请选择目标坐标系",
            "options": ["GCJ-02", "WGS84", "BD-09", "EPSG:3857", "UTM"],
            "required": True,
            "default": "GCJ-02",
            "step": 3,
        },
    },
}


class ClarificationEngine:
    """
    澄清引擎

    负责检测任务参数是否完整，生成追问问题。

    使用方式：
        engine = ClarificationEngine()
        result = engine.check_params("route", {"start": "芜湖南站"})
        if result.needs_clarification:
            for q in result.questions:
                print(f"{q.field}: {q.question}")
    """

    def __init__(self):
        self.templates = CLARIFICATION_TEMPLATES

    def check_params(
        self,
        scenario: str,
        extracted_params: Dict[str, Any],
    ) -> "ClarificationResult":
        """
        检查参数是否完整，生成追问问题

        Args:
            scenario: 场景类型
            extracted_params: 已提取的参数

        Returns:
            ClarificationResult 对象
        """
        from geoagent.dsl.protocol import ClarificationQuestion, ClarificationResult

        template = self.templates.get(scenario, {})
        if not template:
            return ClarificationResult(
                needs_clarification=False,
                questions=[],
                auto_filled={},
            )

        questions = []
        auto_filled = {}

        for field, spec in template.items():
            # 检查参数是否存在
            if field not in extracted_params or not extracted_params[field]:
                # 必填参数需要追问
                if spec.get("required", True):
                    q = ClarificationQuestion(
                        field=field,
                        question=spec["question"],
                        options=spec.get("options"),
                        required=True,
                    )
                    questions.append(q)
                else:
                    # 可选参数使用默认值
                    default = spec.get("default")
                    if default:
                        auto_filled[field] = default

        return ClarificationResult(
            needs_clarification=len(questions) > 0,
            questions=questions,
            auto_filled=auto_filled,
        )

    def generate_follow_up(
        self,
        scenario: str,
        context: Dict[str, Any],
    ) -> str:
        """
        生成追问话术

        Args:
            scenario: 场景类型
            context: 当前上下文

        Returns:
            追问话术字符串
        """
        result = self.check_params(scenario, context)
        if not result.needs_clarification:
            return ""

        lines = []
        if len(result.questions) == 1:
            q = result.questions[0]
            lines.append(q.question)
            if q.options:
                opts = "、".join(q.options)
                lines.append(f"（可选：{opts}）")
        else:
            lines.append("为了完成分析，我需要确认以下几点：")
            for i, q in enumerate(result.questions, 1):
                lines.append(f"\n{i}. {q.question}")
                if q.options:
                    opts = "、".join(q.options)
                    lines.append(f"   可选：{opts}")

        return "".join(lines)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "IntentClassifier",
    "IntentResult",
    "classify_intent",
    "classify_intent_simple",
    "get_classifier",
    "get_task_type_for_intent",
    "INTENT_KEYWORDS",
    "INTENT_TO_TASK_TYPE",
    # 场景子类型
    "SCENARIO_SUBTYPES",
    # 追问模板
    "CLARIFICATION_TEMPLATES",
    "ClarificationEngine",
]
