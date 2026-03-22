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

import os
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
        "路径", "路线", "route", "步行", "导航", "最短路径", "寻路", "routing",
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
        "缓冲", "buffer", "缓冲区", "方圆",
        "500米范围", "1公里内", "xx米", "xx公里",
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
        "可视范围", "可视区域", "视野范围", "视野分析",
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

    # ── 适宜性分析/选址 (MCDA) ─────────────────────────────────────────
    Scenario.SUITABILITY: [
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

    # ── 🟢 高德基础 Web 服务 ───────────────────────────────────────────

    # 地理编码
    Scenario.GEOCODE: [
        "地理编码", "geocode", "地址转坐标", "地址解析",
        "把地址转成经纬度", "地址转成坐标", "查询地址坐标",
        "这个地址在哪", "想知道地址的坐标", "获取地址坐标",
        "address to coordinates", "geocode address",
    ],

    # 逆地理编码
    Scenario.REGEOCODE: [
        "逆地理编码", "regeocode", "坐标转地址", "坐标解析",
        "把坐标转成地址", "经纬度转地址", "查询坐标地址",
        "这个位置是哪里", "想知道坐标的地址", "获取位置地址",
        "reverse geocode", "coordinates to address",
    ],

    # 行政区域查询
    Scenario.DISTRICT: [
        "行政区域", "行政区划", "district", "查询省市县",
        "下辖哪些", "有哪些区", "区县边界", "行政区边界",
        "城市列表", "省份列表", "获取边界", "polyline",
        "administrative boundaries", "city district",
    ],

    # 静态地图
    Scenario.STATIC_MAP: [
        "静态地图", "static map", "生成地图图片", "地图截图",
        "地图链接", "地图URL", "带标记的地图", "标注地图",
        "生成一张地图", "创建地图图片",
        "map image", "map snapshot",
    ],

    # 坐标转换
    Scenario.COORD_CONVERT: [
        "坐标转换", "convert", "坐标转换", "gcj02", "wgs84",
        "百度坐标", "gps坐标", "图吧坐标", "坐标系统转换",
        "坐标系统", "坐标系转换",
        "coordinate conversion", "coordinate transform",
    ],

    # 轨迹纠偏
    Scenario.GRASP_ROAD: [
        "轨迹纠偏", "gps轨迹", "纠偏", "轨迹修复", "漂移修正",
        "grasp road", "map matching", "轨迹匹配", "road snapping",
        "gps漂移", "轨迹点纠正",
    ],

    # ── 🔵 高德高级 Web 服务 ───────────────────────────────────────────

    # POI 搜索（扩展关键词：覆盖"查询附近餐厅"等自然表达）
    Scenario.POI_SEARCH: [
        # 扩展中文关键词：包含"找"、"查"、"附近"、"周边"、"餐厅"等自然表达
        "搜索", "search", "poi", "地点", "查找附近", "周边搜索",
        "附近有什么", "搜索餐厅", "搜索酒店", "搜索银行",
        "多边形搜索", "关键字搜索", "类型搜索",
        "附近500米", "周边1公里", "3公里内",
        "地点搜索", "场所搜索", "设施搜索",
        # 新增：自然语言表达（"找xxx"模式）
        "找附近", "找周边", "附近找", "周边找",
        "查询附近", "查周边", "附近查询", "周边查询",
        "附近餐厅", "周边餐厅", "附近酒店", "周边酒店",
        "附近银行", "周边银行", "附近超市", "周边超市",
        "附近学校", "周边学校", "附近医院", "周边医院",
        "附近商场", "周边商场", "附近加油站", "周边加油站",
        "附近停车场", "周边停车场", "附近地铁站", "周边地铁站",
        "附近公交站", "周边公交站", "附近景点", "周边景点",
        "附近公园", "周边公园", "附近健身房", "周边健身房",
        "附近药店", "周边药店", "附近便利店", "周边便利店",
        "找餐厅", "找酒店", "找银行", "找超市",
        "找学校", "找医院", "找商场", "找加油站",
        "找停车场", "找地铁站", "找公交站", "找景点",
        "找公园", "找健身房", "找药店", "找便利店",
        "饭店", "餐厅", "美食", "小吃", "咖啡", "酒吧",
        "KTV", "网吧", "电影院", "健身房",
        # 🆕 Step 1: 查店面/查数量意图（"XX周围有多少个XX"模式）
        "多少家", "有哪些", "星巴克", "查一下", "搜索周边",
        "周边设施", "找找", "POI", "网点", "门店",
        "附近有多少", "周边有多少", "附近有几家", "周边有几家",
        "附近有", "周边有", "附近哪些", "周边哪些",
        "找找附近", "找找周边", "查查附近", "查查周边",
        "搜一下", "搜附近", "搜周边",
        "附近商家", "周边商家", "附近商铺", "周边商铺",
        "附近网点", "周边网点", "附近门店", "周边门店",
        "附近有多少家", "附近有几家", "周围有多少",
        "搜索附近", "搜索周边",
        # 英文
        "find places", "nearby search", "place search",
        "find nearby", "around me", "near me",
        "restaurant", "hotel", "bank", "supermarket",
        "school", "hospital", "mall", "gas station",
        "parking", "subway", "metro", "attractions",
        "park", "gym", "pharmacy", "convenience store",
        "how many", "how many nearby", "how many around",
        "places nearby", "nearby places", "shops nearby",
    ],

    # ── 可视化（扩展关键词：地图渲染、热力图、分类图）───────────────
    Scenario.VISUALIZATION: [
        # 中文
        "可视化", "显示在地图", "画在地图", "展示地图",
        "渲染", "绘制", "绘制地图", "呈现在地图上",
        "热力图", "heatmap", "热图", "密度图",
        "分类图", "分级图", "choropleth", "分级设色",
        "专题地图", "专题图", "数据地图",
        "把结果显示", "结果显示在地图", "显示结果",
        "图层", "多图层", "叠加显示",
        # 英文
        "visualize", "visualization", "render", "display on map",
        "heatmap", "heat map", "density map",
        "choropleth", "thematic map", "classification map",
        "layer", "overlay", "show on map",
    ],

    # 输入提示
    Scenario.INPUT_TIPS: [
        "输入提示", "autocomplete", "自动补全", "search suggestions",
        "搜索建议", "补全", "提示",
    ],

    # 交通态势
    Scenario.TRAFFIC_STATUS: [
        "交通态势", "交通状况", "路况", "拥堵", "traffic",
        "实时路况", "是否拥堵", "道路状况", "traffic status",
        "交通流量", "通行状况", "拥堵情况",
        "road congestion", "traffic condition",
    ],

    # 交通事件
    Scenario.TRAFFIC_EVENTS: [
        "交通事件", "traffic events", "封路", "施工", "事故",
        "交通管制", "封道", "绕行", "道路封闭",
        "traffic accident", "road closure", "construction",
        "道路施工", "交通管制", "突发事件",
    ],

    # 公交信息
    Scenario.TRANSIT_INFO: [
        "公交线路", "公交", "地铁", "bus", "metro", "subway",
        "transit info", "公交信息", "地铁线路", "公交换乘",
        "公交站点", "地铁站", "公交查询",
        "bus route", "subway line", "transit",
    ],

    # IP 定位
    Scenario.IP_LOCATION: [
        "ip定位", "ip location", "查询ip", "IP所在地",
        "ip address location", "ip归属地",
        "根据ip查位置", "ip查询位置",
    ],

    # 天气查询
    Scenario.WEATHER: [
        "天气", "weather", "气温", "温度", "下雨", "晴天",
        "weather query", "weather forecast", "天气预报",
        "今天天气", "明天天气", "湿度", "风力", "风速",
        "穿衣指数", "空气指数", "空气质量",
    ],

    # ── 🟣 代码沙盒（受限代码执行）────────────────────────────────
    Scenario.CODE_SANDBOX: [
        # 中文 — 显式触发
        "写一段代码", "写python", "python代码", "写脚本",
        "生成测试数据", "代码实现", "计算面积", "编程", "用代码",
        "写代码", "帮我写", "python实现", "写个脚本",
        "生成数据", "用python", "script", "compute",
        "写个函数", "代码计算", "脚本", "计算一下",
        "代码生成", "python编程", "写段代码",
        # 中文 — 隐式/计算类触发（空间数学疑难杂症）
        "生成", "随机生成", "随机点", "面积计算", "长度计算", "距离计算",
        "统计", "算法", "提取坐标", "自定义公式", "加权求和",
        "坐标转换", "数学公式", "自定义逻辑", "迭代计算",
        "拟合", "插值自定义", "算一下", "帮我算", "帮我生成",
        "帮我统计", "批量处理", "循环处理",
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
    primary: Scenario           # 主要意图
    confidence: float          # 置信度 0-1
    matched_keywords: List[str]  # 匹配的关键词
    all_intents: Set[Scenario]  # 所有识别到的意图

    def __str__(self) -> str:
        primary = self.primary.value if hasattr(self.primary, 'value') else str(self.primary)
        return f"IntentResult(primary={primary}, confidence={self.confidence:.2f})"


# =============================================================================
# 意图分类器
# =============================================================================

class IntentClassifier:
    """
    意图分类器

    采用两阶段架构：
    1. 第一阶段：LLM 语义理解（优先）
    2. 第二阶段：关键词匹配（兜底）

    设计原则：
    - LLM 优先处理复杂语义理解
    - 关键词匹配作为降级方案
    - 返回固定 Scenario 枚举
    - LLM 不可用时优雅降级

    使用方式：
        classifier = IntentClassifier()
        result = classifier.classify("芜湖南站到方特的步行路径")
        print(result.primary)  # Scenario.ROUTE
        print(result.confidence)  # 0.99
    """

    def __init__(self, threshold: float = 0.05):
        """
        初始化意图分类器

        Args:
            threshold: 置信度阈值，低于此值则返回 "general"（无法确定场景）
        """
        self.threshold = threshold
        self._build_index()

    def _build_index(self) -> None:
        """构建关键词索引以加速匹配"""
        self._keyword_to_intent: Dict[str, List[Scenario]] = {}
        self._intent_keywords: Dict[Scenario, List[str]] = {}

        for intent, keywords in INTENT_KEYWORDS.items():
            self._intent_keywords[intent] = keywords
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower not in self._keyword_to_intent:
                    self._keyword_to_intent[kw_lower] = []
                self._keyword_to_intent[kw_lower].append(intent)

    def _call_llm_intent(self, query: str) -> Optional[IntentResult]:
        """
        调用 LLM 进行意图分类。

        返回 IntentResult 或 None（失败时降级）。

        Args:
            query: 用户输入的自然语言查询

        Returns:
            IntentResult 或 None（LLM 不可用或调用失败）
        """
        import json

        # 检查 API Key
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None

        # 检查是否禁用 LLM
        if os.getenv("GEOAGENT_DISABLE_LLM", "").lower() in ("1", "true", "yes"):
            return None

        SCENARIOS = [s.value for s in Scenario]
        SCENARIO_NAMES = [s.name for s in Scenario]

        prompt = f"""你是专业的 GIS 空间意图识别器。
请将用户输入分类到以下场景之一：
{', '.join(SCENARIOS)}

涉及复杂空间运算、多条件距离计算、无本地数据的分析 → CODE_SANDBOX
输入："{query}"
只输出枚举单词，不要解释。"""

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )

            response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=20,
                timeout=30.0,
            )

            reply = response.choices[0].message.content or ""
            if not reply:
                return None

            reply = reply.strip().upper()

            # 尝试匹配 Scenario 枚举
            for scenario in Scenario:
                if scenario.name == reply or scenario.value.upper() == reply:
                    return IntentResult(
                        primary=scenario,
                        confidence=0.99,  # LLM 置信度高
                        matched_keywords=["llm_detected"],
                        all_intents={scenario}
                    )

            # 尝试 JSON 解析（某些模型可能返回 JSON）
            try:
                parsed = json.loads(reply)
                if isinstance(parsed, dict) and "scenario" in parsed:
                    scenario_name = parsed["scenario"].upper()
                    for scenario in Scenario:
                        if scenario.name == scenario_name or scenario.value.upper() == scenario_name:
                            return IntentResult(
                                primary=scenario,
                                confidence=0.99,
                                matched_keywords=["llm_detected"],
                                all_intents={scenario}
                            )
            except (json.JSONDecodeError, TypeError):
                pass

        except Exception:
            pass

        return None

    def _classify_by_keywords(self, query: str, multi: bool = False) -> IntentResult:
        """
        关键词匹配兜底方法（原 classify 核心逻辑）。

        当 LLM 不可用或调用失败时使用。

        Args:
            query: 用户输入的自然语言查询
            multi: 是否返回多意图（True）或仅返回主要意图（False）

        Returns:
            IntentResult 对象
        """
        # ==========================================
        # 🚨 哈基米前置拦截器 (Early Exit Pattern)
        # ==========================================
        sandbox_dictators = ["用代码", "沙盒", "写一段代码", "写脚本", "写python", "代码算"]

        if any(trigger in query for trigger in sandbox_dictators):
            return IntentResult(
                primary=Scenario.CODE_SANDBOX,
                confidence=1.0,
                matched_keywords=["显式编程触发词"],
                all_intents={Scenario.CODE_SANDBOX}
            )
        # ==========================================

        query_lower = query.lower()
        matched: Dict[Scenario, List[str]] = {}

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

        # 无匹配 → 无法确定具体场景
        if not matched:
            stripped = query.strip()
            is_garbage = stripped.isdigit() or len(stripped) < 2
            if is_garbage:
                return IntentResult(
                    primary="general",
                    confidence=1.0,
                    matched_keywords=[],
                    all_intents=set()
                )
            return IntentResult(
                primary="general",
                confidence=0.0,
                matched_keywords=[],
                all_intents=set()
            )

        # 计算每个意图的得分
        scores: Dict[Scenario, float] = {}
        for intent, keywords in matched.items():
            n_matched = len(keywords)
            total_kw_chars = sum(len(kw) for kw in keywords)
            all_kws = self._intent_keywords.get(intent, [])
            char_score = total_kw_chars / max(sum(len(k) for k in all_kws), 1)
            count_score = n_matched / max(len(all_kws), 1)
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
            all_intents = {sorted_intents[0][0]} if sorted_intents else set()

        # 检查是否低于阈值
        if confidence < self.threshold:
            return IntentResult(
                primary="general",
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

    def classify(self, query: str, multi: bool = False) -> IntentResult:
        """
        对用户 query 进行意图分类（两阶段架构）。

        1. 第一阶段：LLM 语义理解（优先）
        2. 第二阶段：关键词匹配兜底（降级）

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

        # ===== 第一阶段：LLM 语义理解（降维打击）=====
        try:
            llm_reply = self._call_llm_intent(query)
            if llm_reply:
                return llm_reply
        except Exception:
            pass  # LLM 失败，默默降级

        # ===== 第二阶段：关键词匹配兜底（保留原逻辑）=====
        return self._classify_by_keywords(query, multi)

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
