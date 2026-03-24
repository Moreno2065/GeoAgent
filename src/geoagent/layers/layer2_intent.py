"""
第2层：意图识别层（Intent Classifier）
======================================
核心职责：
1. 判断用户输入属于哪个大场景（严格枚举，不让 LLM 自由发挥）
2. 返回固定的 Scenario 枚举值
3. 提供置信度和匹配的关键词

设计原则：
- Embedding 语义匹配优先（参考 Copilot 94.5% Tool Use Coverage）
- LLM 作为降级方案
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
# Embedding 路由器导入（延迟加载，避免循环导入）
# =============================================================================

_embedding_router = None


def _get_embedding_router():
    """延迟加载 Embedding 路由器"""
    global _embedding_router
    if _embedding_router is None:
        try:
            from geoagent.tools.embedding_router import get_embedding_router
            # 设置环境变量控制是否启用 Embedding
            if os.getenv("GEOAGENT_USE_EMBEDDING", "1").lower() in ("1", "true", "yes"):
                _embedding_router = get_embedding_router(precompute=True)
        except ImportError:
            pass
    return _embedding_router


# =============================================================================
# 意图关键词配置（从模块化子模块导入）
# =============================================================================

# 从新模块化结构导入 INTENT_KEYWORDS
try:
    from geoagent.layers.intent import INTENT_KEYWORDS
except ImportError:
    # 如果模块导入失败，使用内联定义（向后兼容）
    from geoagent.layers.architecture import Scenario as _Scenario
    
    # 内联定义（简化版本，用于兼容）
    INTENT_KEYWORDS: Dict[_Scenario, List[str]] = {
        _Scenario.ROUTE: ["路径", "路线", "route", "步行", "导航", "可达"],
        _Scenario.BUFFER: ["缓冲", "buffer", "缓冲区", "方圆"],
        _Scenario.OVERLAY: ["叠加", "overlay", "相交", "intersect", "clip"],
        _Scenario.INTERPOLATION: ["插值", "interpolation", "IDW"],
        _Scenario.VIEWSHED: ["视域", "viewshed", "阴影", "shadow"],
        _Scenario.CODE_SANDBOX: ["代码", "python", "script", "生成"],
        _Scenario.FETCH_OSM: ["osm", "openstreetmap", "下载"],
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

        # 🆕 修复：多条件 POI 搜索 → MULTI_CRITERIA_SEARCH，不是 CODE_SANDBOX
        # CODE_SANDBOX 只用于"生成测试数据"、"随机点"等纯代码计算场景
        prompt = f"""你是专业的 GIS 空间意图识别器。
请将用户输入分类到以下场景之一：
{', '.join(SCENARIOS)}

## 场景判断规则（必须严格遵守）

### 判断为 MULTI_CRITERIA_SEARCH 的场景：
- 多条件 POI 搜索（同时满足多个距离条件）
- 例如："距离星巴克<200米且距离地铁站>500米的地方"
- 例如："找一个离星巴克近但离地铁站远的地方"
- 特点：有具体的 POI 类型（星巴克、地铁站）+ 具体的距离阈值

### 判断为 CODE_SANDBOX 的场景：
- 生成随机测试数据（随机点、随机多边形）
- 纯数学计算（面积、周长、距离的具体数值计算）
- 需要执行自定义 Python 代码的场景
- 特点：没有具体的 POI 搜索，只是代码计算

### 其他场景判断：
- 单一 POI 搜索（"附近有哪些星巴克"）→ POI_SEARCH
- 路径规划（"从 A 到 B"）→ ROUTE
- 缓冲分析（"某图层周边500米"）→ BUFFER

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
        关键词匹配兜底方法。

        当 Embedding 和 LLM 都不可用时使用。

        Args:
            query: 用户输入的自然语言查询
            multi: 是否返回多意图（True）或仅返回主要意图（False）

        Returns:
            IntentResult 对象
        """
        # 注意：硬编码拦截器已经在 classify() 中处理，这里不需要重复

        query_lower = query.lower()

        # 注意：下载拦截器已经在 classify() 中通过 _check_interceptors 处理

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
            base_score = char_score * 0.6 + count_score * 0.4

            # 🟣 CODE_SANDBOX 优先级加成
            if intent == Scenario.CODE_SANDBOX:
                # 明确的编程意图关键词
                explicit_keywords = ['python', '代码', '写', 'script', '编程',
                                   '用python', '用代码', '写代码', '帮我算', '算一下',
                                   '帮我计算', '执行', '运行代码']
                if any(kw in query.lower() for kw in explicit_keywords):
                    base_score *= 3.0  # 明确的编程意图，加权 3 倍

            scores[intent] = base_score

        # 🔵 检查是否没有匹配到 GIS 关键词，但有"计算"类关键词
        # 如果只有"交集"、"叠加"等模糊词，没有图层引用，则走 code_sandbox
        query_lower = query.lower()
        has_gis_layer_keywords = any(kw in query_lower for kw in [
            '图层', 'shp', 'geojson', '矢量', '缓冲区', '道路', '建筑',
            '范围内', '以内的', '圆内', '方形', '多边形', '500米', '1公里', '半径'
        ])

        if not has_gis_layer_keywords:
            # 没有 GIS 图层引用，检查是否有计算关键词
            has_calc_keywords = any(kw in query_lower for kw in [
                '计算', '求', '面积', '周长', '距离', '长度', '角度',
                '弧度', '体积', '容积', '坐标', '交集', '并集', '差集'
            ])
            if has_calc_keywords:
                # 没有 GIS 图层但有计算意图，优先走 code_sandbox
                scores[Scenario.CODE_SANDBOX] = max(scores.get(Scenario.CODE_SANDBOX, 0), 0.8)

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
        对用户 query 进行意图分类（三阶段架构）。

        1. 硬编码拦截器（立即返回，无延迟）
        2. Embedding 语义匹配（低延迟，高准确率）
        3. LLM 语义理解（降级方案）
        4. 关键词匹配兜底（最终降级）

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

        # ===== 阶段 1: 硬编码拦截器（最高优先级，无延迟）=====
        intercept_result = self._check_interceptors(query)
        if intercept_result:
            return intercept_result

        # ===== 阶段 2: Embedding 语义匹配（低延迟，高准确率）=====
        try:
            embed_router = _get_embedding_router()
            if embed_router:
                embed_result = self._classify_by_embedding(query)
                if embed_result and embed_result.confidence > 0.7:
                    return embed_result
        except Exception:
            pass  # Embedding 失败，继续降级

        # ===== 阶段 3: LLM 语义理解（降级方案）=====
        try:
            llm_reply = self._call_llm_intent(query)
            if llm_reply:
                return llm_reply
        except Exception:
            pass  # LLM 失败，默默降级

        # ===== 阶段 4: 关键词匹配兜底（最终降级）=====
        return self._classify_by_keywords(query, multi)

    def _check_interceptors(self, query: str) -> Optional[IntentResult]:
        """
        检查硬编码拦截器

        这些是必须立即拦截的场景，用于处理特殊指令。
        """
        query_lower = query.lower()

        # ── 代码沙盒拦截器 ────────────────────────────────
        sandbox_dictators = ["用代码", "沙盒", "写一段代码", "写脚本", "写python", "代码算",
                            "写段代码", "python代码", "python实现", "用python"]

        if any(trigger in query_lower for trigger in sandbox_dictators):
            return IntentResult(
                primary=Scenario.CODE_SANDBOX,
                confidence=1.0,
                matched_keywords=["显式编程触发词"],
                all_intents={Scenario.CODE_SANDBOX}
            )

        # ── 下载拦截器 ────────────────────────────────
        download_dictators = ["下载", "download", "下载地图", "地图下载",
                            "下载osm", "osm下载", "抓取地图", "下载路网", "下载建筑",
                            "下载区域", "下载矢量", "导出数据", "导出地图"]

        if any(trigger in query_lower for trigger in download_dictators):
            return IntentResult(
                primary=Scenario.FETCH_OSM,
                confidence=1.0,
                matched_keywords=["下载触发词"],
                all_intents={Scenario.FETCH_OSM}
            )

        # ── POI 搜索拦截器 ────────────────────────────────
        # 优先于 STATISTICS：查询某个地点有多少个 XX 的场景
        poi_count_patterns = [
            "有多少", "有几家", "有多少家", "有几个",
            "有多少个", "有多少家", "多少家", "几家",
            "星巴克", "瑞幸", "喜茶", "麦当劳", "肯德基",
            "门店", "网点", "店铺", "商铺",
        ]
        # 必须同时包含地点词和数量词，才是 POI 搜索
        location_words = ["区", "市", "县", "镇", "街", "路", "天河", "广州", "深圳", "北京", "上海", "杭州", "南京", "武汉", "成都"]
        has_location = any(loc in query for loc in location_words)
        has_count = any(pattern in query_lower for pattern in poi_count_patterns)

        if has_location and has_count:
            return IntentResult(
                primary=Scenario.POI_SEARCH,
                confidence=1.0,
                matched_keywords=["POI数量查询"],
                all_intents={Scenario.POI_SEARCH}
            )

        return None

    def _classify_by_embedding(self, query: str) -> Optional[IntentResult]:
        """
        Embedding 语义匹配

        使用预计算的 Embedding 向量进行快速语义匹配。
        参考 GitHub Copilot 2025 的 Embedding-guided Tool Routing。
        """
        try:
            embed_router = _get_embedding_router()
            if not embed_router:
                return None

            result = embed_router.route(query, top_k=3, include_tools=False)

            if result.primary_scenario:
                return IntentResult(
                    primary=result.primary_scenario,
                    confidence=result.primary.score,
                    matched_keywords=[f"embedding:{result.primary.name}"],
                    all_intents={result.primary_scenario}
                )
        except Exception:
            pass

        return None

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
