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
        # 英文
        "driving", "walking", "walk", "drive", "bike", "cycling",
        "shortest path", "shortest route", "navigation", "directions",
        "origin", "destination", "from to",
    ],
    "buffer": [
        # 中文
        "缓冲", "buffer", "缓冲区", "方圆", "周围", "附近范围",
        "500米范围", "1公里内", "xx米", "xx公里",
        # 英文
        "buffer zone", "buffering", "proximity", "within distance",
        "within radius", "buffer analysis",
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
        "可视化", "地图", "可视化地图", "3d地图", "交互地图",
        "folium", "热力图", "choropleth", "出图", "渲染",
        "绘制地图", "展示地图", "web地图", "html地图",
        # 英文
        "visualization", "map", "3d map", "interactive map",
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
        "生成", "随机生成", "随机点", "面积计算", "长度计算", "距离计算",
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
            # 得分 = 匹配关键词数量 * 关键词长度权重（归一化）
            n_matched = len(keywords)
            # 关键词总字符长度
            total_kw_chars = sum(len(kw) for kw in keywords)
            # 长关键词匹配给予更高权重
            char_score = total_kw_chars / max(sum(len(k) for k in self._intent_keywords.get(intent, [])), 1)
            # 关键词数量得分
            count_score = n_matched / max(len(self._intent_keywords.get(intent, [])), 1)
            # 综合得分：字符权重 + 数量权重
            scores[intent] = char_score * 0.6 + count_score * 0.4

        # 按得分排序
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 置信度：使用 sigmoid 变换使得分更合理
        # 1个匹配关键词约 0.5 分，2个约 0.8 分，3个以上接近 1.0
        def _sigmoid(x):
            return 1 / (1 + 2 ** (-x))
        raw_score = sorted_intents[0][1] if sorted_intents else 0
        confidence = _sigmoid(raw_score * 10)

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
