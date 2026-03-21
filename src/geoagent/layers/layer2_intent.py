"""
第2层：意图识别层（Intent Classifier）
======================================
核心职责：
1. 判断用户输入属于哪个大场景（严格枚举，不让 LLM 自由发挥）
2. 返回固定的 Scenario 枚举值
3. 提供置信度和匹配的关键词

设计原则：
- 不依赖 LLM，用关键词匹配实现意图分类
- 返回固定枚举，不让 LLM 自由发挥
- 支持多意图识别（但 MVP 只返回单意图）
- 稳定高效，可测试
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

from geoagent.layers.architecture import Scenario


# =============================================================================
# 意图关键词配置（严格限定的 7 类场景）
# =============================================================================

INTENT_KEYWORDS: Dict[Scenario, List[str]] = {
    # ── 路径/可达性分析 ────────────────────────────────────────────────
    Scenario.ROUTE: [
        # 中文
        "路径", "route", "步行", "导航", "最短路径", "寻路", "routing",
        "从...到", "到...的", "出发地", "目的地", "起点", "终点",
        "可达", "可达性", "等时圈", "服务范围", "通行时间", "出行时间",
        "15分钟", "10分钟", "5分钟", "30分钟生活圈",
        # 英文
        "driving", "walking", "walk", "drive", "bike", "cycling",
        "shortest path", "shortest route", "navigation", "directions",
        "origin", "destination", "from to", "accessibility", "isochrone",
        "service area", "travel time", "walking distance",
    ],

    # ── 缓冲/邻近分析 ────────────────────────────────────────────────
    Scenario.BUFFER: [
        # 中文
        "缓冲", "buffer", "缓冲区", "方圆", "周围", "附近范围",
        "500米范围", "1公里内", "xx米", "xx公里",
        "周边", "周边适合", "周边分析",
        # 英文
        "buffer zone", "buffering", "proximity", "within distance",
        "within radius", "buffer analysis",
    ],

    # ── 叠置/裁剪分析 ────────────────────────────────────────────────
    Scenario.OVERLAY: [
        # 中文
        "叠加", "overlay", "相交", "intersect", "合并", "union",
        "clip", "裁剪", "擦除", "difference", "对称差", "交集",
        "空间叠置", "空间分析", "叠图", "叠合", "选址分析",
        # 英文
        "spatial overlay", "intersection", "clipping", "erasing",
        "map overlay", "layer combination", "site selection",
    ],

    # ── 插值/表面分析 ────────────────────────────────────────────────
    Scenario.INTERPOLATION: [
        # 中文
        "插值", "interpolation", "IDW", "kriging", "空间插值",
        "离散点", "连续表面", "反距离加权", "克里金",
        "插值分析", "空间预测", "生成表面",
        # 英文
        "idw", "inverse distance", "spatial interpolation",
        "surface generation", "grid generation", "rasterize",
    ],

    # ── 视域/阴影 ───────────────────────────────────────────────────
    Scenario.VIEWSHED: [
        # 中文
        "视域", "视域分析", "通视", "通视分析", "可见性",
        "阴影", "shadow", "日照", "遮挡", "采光",
        "建筑阴影", "日照分析", "阴影分析",
        # 英文
        "viewshed", "viewshed analysis", "visibility", "visible",
        "line of sight", "los", "viewpoint",
        "shadow analysis", "sun shadow", "sunlight", "solar access",
    ],

    # ── 统计/聚合 ───────────────────────────────────────────────────
    Scenario.STATISTICS: [
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
    ],

    # ── 栅格分析 ───────────────────────────────────────────────────
    Scenario.RASTER: [
        # 中文
        "ndvi", "植被", "植被指数", "ndwi", "evi", "水体指数",
        "遥感指数", "绿度指数", "归一化植被", "卫星影像指数",
        "计算ndvi", "计算植被",
        "坡度", "坡向", "dem", "高程",
        # 英文
        "ndvi", "ndwi", "evi", "vegetation index", "vegetation analysis",
        "satellite index", "remote sensing", "leaf area", "lai",
        "slope", "aspect", "dem", "elevation", "terrain",
    ],
}


# =============================================================================
# 意图分类结果
# =============================================================================

@dataclass
class IntentResult:
    """意图分类结果"""
    primary: Scenario           # 主要意图
    confidence: float          # 置信度 0-1
    matched_keywords: List[str]  # 匹配的关键词
    all_intents: Set[Scenario]  # 所有识别到的意图

    def __str__(self) -> str:
        return f"IntentResult(primary={self.primary.value}, confidence={self.confidence:.2f})"


# =============================================================================
# 意图分类器
# =============================================================================

class IntentClassifier:
    """
    意图分类器

    基于关键词的意图分类，稳定高效。

    设计原则：
    - 不依赖 LLM，用关键词匹配
    - 返回固定 Scenario 枚举
    - 置信度基于匹配得分

    使用方式：
        classifier = IntentClassifier()
        result = classifier.classify("芜湖南站到方特的步行路径")
        print(result.primary)  # Scenario.ROUTE
        print(result.confidence)  # 0.95
    """

    def __init__(self, threshold: float = 0.0):
        """
        初始化意图分类器

        Args:
            threshold: 置信度阈值，低于此值则返回 None
        """
        self.threshold = threshold
        self._build_index()

    def _build_index(self) -> None:
        """构建关键词索引以加速匹配"""
        # 关键词 -> 意图的映射
        self._keyword_to_intent: Dict[str, List[Scenario]] = {}
        # 意图 -> 关键词列表
        self._intent_keywords: Dict[Scenario, List[str]] = {}

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
                primary=Scenario.ROUTE,
                confidence=0.0,
                matched_keywords=[],
                all_intents=set()
            )

        query_lower = query.lower()
        matched: Dict[Scenario, List[str]] = {}  # intent -> matched keywords

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
            # 没有匹配，返回默认意图
            return IntentResult(
                primary=Scenario.ROUTE,
                confidence=0.0,
                matched_keywords=[],
                all_intents=set()
            )

        # 计算每个意图的得分
        scores: Dict[Scenario, float] = {}
        for intent, keywords in matched.items():
            # 得分 = 匹配关键词数量 * 关键词长度权重（归一化）
            n_matched = len(keywords)
            # 关键词总字符长度
            total_kw_chars = sum(len(kw) for kw in keywords)
            # 关键词数量得分
            all_kws = self._intent_keywords.get(intent, [])
            char_score = total_kw_chars / max(sum(len(k) for k in all_kws), 1)
            count_score = n_matched / max(len(all_kws), 1)
            # 综合得分：字符权重 + 数量权重
            scores[intent] = char_score * 0.6 + count_score * 0.4

        # 按得分排序
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 置信度：使用 sigmoid 变换
        def _sigmoid(x):
            return 1 / (1 + 2 ** (-x))
        raw_score = sorted_intents[0][1] if sorted_intents else 0
        confidence = _sigmoid(raw_score * 10)

        if multi:
            all_intents = {intent for intent, _ in sorted_intents}
        else:
            # 单意图模式：返回得分最高的意图
            all_intents = {sorted_intents[0][0]} if sorted_intents else set()

        # 检查是否低于阈值
        if confidence < self.threshold:
            return IntentResult(
                primary=Scenario.ROUTE,
                confidence=confidence,
                matched_keywords=[],
                all_intents=set()
            )

        primary_intent = sorted_intents[0][0] if sorted_intents else Scenario.ROUTE
        primary_keywords = matched.get(primary_intent, [])

        return IntentResult(
            primary=primary_intent,
            confidence=confidence,
            matched_keywords=primary_keywords,
            all_intents=all_intents
        )

    def classify_simple(self, query: str) -> Scenario:
        """
        简单分类：直接返回 Scenario 枚举

        Args:
            query: 用户输入

        Returns:
            Scenario 枚举值
        """
        result = self.classify(query)
        return result.primary


# =============================================================================
# 便捷函数
# =============================================================================

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


def classify_intent_simple(query: str) -> Scenario:
    """
    便捷函数：直接返回 Scenario 枚举

    Args:
        query: 用户输入

    Returns:
        Scenario 枚举值
    """
    result = classify_intent(query)
    return result.primary


__all__ = [
    "INTENT_KEYWORDS",
    "IntentResult",
    "IntentClassifier",
    "get_classifier",
    "classify_intent",
    "classify_intent_simple",
]
