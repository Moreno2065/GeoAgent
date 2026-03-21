"""
第3层：场景编排层（Scenario Orchestrator）
==========================================
核心职责：
1. 接收第2层的意图分类结果
2. 决定请求属于哪类任务
3. 需不需要追问
4. 是否可以自动补参数
5. 走哪个分析模板

设计原则：
- 确定性参数提取，不用 LLM 猜测
- 参数不完整必须追问，不能瞎猜
- 自动补全合理的默认值
- 支持上下文合并
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from geoagent.layers.architecture import Scenario, PipelineStatus
from geoagent.layers.layer2_intent import IntentResult, IntentClassifier


# =============================================================================
# 澄清问题
# =============================================================================

@dataclass
class ClarificationQuestion:
    """澄清问题定义"""
    field: str
    question: str
    options: Optional[List[str]] = None
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ClarificationResult:
    """澄清结果"""
    needs_clarification: bool
    questions: List[ClarificationQuestion] = field(default_factory=list)
    auto_filled: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 编排结果
# =============================================================================

@dataclass
class OrchestrationResult:
    """场景编排结果"""
    status: PipelineStatus
    scenario: Scenario
    needs_clarification: bool = False
    questions: List[ClarificationQuestion] = field(default_factory=list)
    auto_filled: Dict[str, Any] = field(default_factory=dict)
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    intent_result: Optional[IntentResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "scenario": self.scenario.value,
            "needs_clarification": self.needs_clarification,
            "questions": [
                {"field": q.field, "question": q.question, "options": q.options}
                for q in self.questions
            ],
            "auto_filled": self.auto_filled,
            "extracted_params": self.extracted_params,
            "error": self.error,
        }


# =============================================================================
# 参数提取器
# =============================================================================

class ParameterExtractor:
    """
    从自然语言中提取关键参数

    支持：
    - 地址/位置提取
    - 距离/范围提取
    - 模式提取（步行/驾车/骑行）
    - 时间提取
    - 文件名提取
    """

    # 距离提取正则
    DISTANCE_PATTERNS = [
        (r"(\d+(?:\.\d+)?)\s*(?:公里|km|kilometers?)", "kilometers"),
        (r"(\d+(?:\.\d+)?)\s*(?:米|meters?|m(?![a-z]))", "meters"),
        (r"(\d+(?:\.\d+)?)\s*(?:千米|km)", "kilometers"),
        (r"方圆\s*(\d+(?:\.\d+)?)\s*(?:公里|km)", "kilometers"),
        (r"方圆\s*(\d+(?:\.\d+)?)\s*(?:米|m)", "meters"),
    ]

    # 时间提取正则（分钟）
    TIME_PATTERNS = [
        (r"(\d+)\s*分钟", "minutes"),
        (r"(\d+)\s*min", "minutes"),
        (r"(\d+)\s*小时", "hours"),
    ]

    # 模式提取关键词
    MODE_KEYWORDS = {
        "walking": ["步行", "走路", "徒步", "walk", "walking"],
        "driving": ["驾车", "开车", "自驾", "drive", "driving", "开车"],
        "transit": ["公交", "地铁", "公共交通", "transit", "bus"],
        "cycling": ["骑行", "骑车", "bike", "cycling"],
    }

    def extract_distance(self, query: str) -> Optional[Dict[str, Any]]:
        """从查询中提取距离参数"""
        for pattern, unit in self.DISTANCE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return {
                    "distance": value,
                    "unit": unit,
                }
        return None

    def extract_time(self, query: str) -> Optional[Dict[str, Any]]:
        """从查询中提取时间参数"""
        for pattern, unit in self.TIME_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if unit == "hours":
                    value *= 60  # 转换为分钟
                return {
                    "time_threshold": value,
                    "time_unit": "minutes",
                }
        return None

    def extract_mode(self, query: str) -> Optional[str]:
        """从查询中提取交通模式"""
        query_lower = query.lower()
        for mode, keywords in self.MODE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    return mode
        return None

    def extract_locations(self, query: str) -> Dict[str, Optional[str]]:
        """从查询中提取起点和终点"""
        result = {"start": None, "end": None}

        patterns = [
            r"从\s*(.+?)\s*(?:到|至)\s*(.+)",
            r"(?:from|origin|起点)\s*[:：]?\s*(.+?)\s*(?:to|end|终点|destination|目的地)\s*[:：]?\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                result["start"] = match.group(1).strip()
                result["end"] = match.group(2).strip()
                break

        return result

    def extract_file_references(self, query: str) -> List[str]:
        """从查询中提取文件引用"""
        extensions = [".shp", ".geojson", ".json", ".gpkg", ".tif", ".tiff", ".csv"]
        references = []

        query_lower = query.lower()
        for ext in extensions:
            if ext in query_lower:
                pattern = rf"[\'\"\\]?([^\'\"\\\n]*{re.escape(ext)})[\'\"\\]?"
                matches = re.findall(pattern, query_lower)
                references.extend(matches)

        return list(set(references))

    def extract_value_field(self, query: str) -> Optional[str]:
        """从查询中提取数值字段名"""
        patterns = [
            r"(?:字段|field|属性)[:：]?\s*[\'\"\\]?(\w+)[\'\"\\]?",
            r"(?:按|根据|用)\s*[\'\"\\]?(\w+)[\'\"\\]?\s*(?:分析|计算|插值)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def extract_entity_name(self, query: str) -> Optional[str]:
        """从查询中提取实体名称（学校、地铁站等）"""
        patterns = [
            r"(?:在|对|以|给)\s*([^\s,，。、]+?)\s*(?:周边|附近|方圆|做|进行)",
            r"([^\s,，。、]+?)\s*(?:周边|附近|方圆|周围)",
            r"(?:分析|缓冲|找)\s*([^\s,，。、]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        return None

    def extract_overlay_operation(self, query: str) -> str:
        """从查询中提取叠加操作类型"""
        q = query.lower()
        if "交集" in query or "相交" in query or "intersect" in q:
            return "intersect"
        if "并集" in query or "合并" in query or "union" in q:
            return "union"
        if "裁剪" in query or "clip" in q:
            return "clip"
        if "差集" in query or "擦除" in query or "difference" in q:
            return "difference"
        return "intersect"

    def extract_interpolation_method(self, query: str) -> str:
        """从查询中提取插值方法"""
        q = query.lower()
        if "kriging" in q or "克里金" in query:
            return "kriging"
        if "nearest" in q or "最近邻" in query:
            return "nearest"
        if "idw" in q or "反距离" in query:
            return "idw"
        return "idw"

    def extract_all(self, query: str, scenario: Scenario) -> Dict[str, Any]:
        """
        从查询中提取所有相关参数

        覆盖所有 7 个场景。

        Args:
            query: 用户查询
            scenario: 场景类型

        Returns:
            提取的参数字典
        """
        params: Dict[str, Any] = {}

        # 通用提取（所有场景共用）
        params["distance_info"] = self.extract_distance(query)
        params["time_info"] = self.extract_time(query)
        params["mode"] = self.extract_mode(query)
        params["locations"] = self.extract_locations(query)
        params["files"] = self.extract_file_references(query)
        params["value_field"] = self.extract_value_field(query)

        # 通用地点
        params["start"] = params["locations"]["start"]
        params["end"] = params["locations"]["end"]
        params["location"] = params["start"]

        # ── route 场景 ───────────────────────────────────────────────
        if scenario == Scenario.ROUTE:
            params["start"] = params["locations"]["start"]
            params["end"] = params["locations"]["end"]

        # ── buffer 场景 ─────────────────────────────────────────────
        elif scenario == Scenario.BUFFER:
            files = params["files"]
            if files:
                params["input_layer"] = files[0]
            if params["distance_info"]:
                params["distance"] = params["distance_info"]["distance"]
                params["unit"] = params["distance_info"]["unit"]
            entity = self.extract_entity_name(query)
            if entity and not params.get("input_layer"):
                params["input_layer"] = entity

        # ── overlay 场景 ─────────────────────────────────────────────
        elif scenario == Scenario.OVERLAY:
            files = params["files"]
            if len(files) >= 2:
                params["layer1"] = files[0]
                params["layer2"] = files[1]
            elif len(files) == 1:
                params["layer1"] = files[0]
            params["operation"] = self.extract_overlay_operation(query)

        # ── interpolation 场景 ───────────────────────────────────────
        elif scenario == Scenario.INTERPOLATION:
            files = params["files"]
            if files:
                params["input_points"] = files[0]
            params["method"] = self.extract_interpolation_method(query)

        # ── viewshed 场景 ───────────────────────────────────────────
        elif scenario == Scenario.VIEWSHED:
            params["location"] = params["locations"]["start"] or self.extract_entity_name(query)
            files = params["files"]
            if files:
                params["dem_file"] = files[0]

        # ── statistics 场景 ─────────────────────────────────────────
        elif scenario == Scenario.STATISTICS:
            files = params["files"]
            if files:
                params["input_file"] = files[0]

        # ── raster 场景 ─────────────────────────────────────────────
        elif scenario == Scenario.RASTER:
            files = params["files"]
            if files:
                params["input_file"] = files[0]

        return params


# =============================================================================
# 澄清引擎
# =============================================================================

CLARIFICATION_TEMPLATES: Dict[Scenario, Dict[str, Dict[str, Any]]] = {
    Scenario.ROUTE: {
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
            "default": "walking",
        },
    },
    Scenario.BUFFER: {
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
    Scenario.OVERLAY: {
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
    Scenario.INTERPOLATION: {
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
            "default": "idw",
        },
    },
    Scenario.VIEWSHED: {
        "location": {
            "question": "请问观察点位置在哪里？",
            "options": None,
            "required": True,
            "examples": ["某楼顶", "某山顶", "某观景点"],
        },
        "dem_file": {
            "question": "请提供 DEM 高程数据文件？",
            "options": None,
            "required": True,
            "examples": ["dem.tif", "高程数据.geojson"],
        },
    },
    Scenario.STATISTICS: {
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
    },
    Scenario.RASTER: {
        "input_file": {
            "question": "请提供遥感影像文件？",
            "options": None,
            "required": True,
            "examples": ["Landsat.tif", "Sentinel2.tif", "卫星影像"],
        },
        "index_type": {
            "question": "请问要计算什么指数？",
            "options": ["NDVI（植被指数）", "NDWI（水体指数）", "EVI（增强植被指数）"],
            "required": False,
            "default": "ndvi",
        },
    },
}


class ClarificationEngine:
    """澄清引擎：检查参数是否完整，生成追问问题"""

    def __init__(self):
        self.templates = CLARIFICATION_TEMPLATES

    def check_params(self, scenario: Scenario, extracted_params: Dict[str, Any]) -> ClarificationResult:
        """检查参数是否完整"""
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


# =============================================================================
# 场景编排器
# =============================================================================

class ScenarioOrchestrator:
    """
    场景编排器

    核心流程：
    1. 接收意图分类结果
    2. 参数提取（确定性正则）
    3. 追问判断（参数不完整时生成追问）
    4. 构建编排结果

    使用方式：
        orchestrator = ScenarioOrchestrator()
        result = orchestrator.orchestrate("芜湖南站到方特的步行路径")
        if not result.needs_clarification:
            # 参数完整，进入第4层
            dsl = result.task
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.clarification_engine = ClarificationEngine()
        self.parameter_extractor = ParameterExtractor()
        self._scenario_defaults = self._init_defaults()

    def _init_defaults(self) -> Dict[Scenario, Dict[str, Any]]:
        """初始化场景默认值"""
        return {
            Scenario.ROUTE: {
                "mode": "walking",
            },
            Scenario.BUFFER: {
                "unit": "meters",
            },
            Scenario.OVERLAY: {
                "operation": "intersect",
            },
            Scenario.INTERPOLATION: {
                "method": "idw",
                "power": 2.0,
                "resolution": 100,
            },
            Scenario.VIEWSHED: {
                "observer_height": 1.7,
                "max_distance": 5000,
            },
            Scenario.STATISTICS: {
                "analysis_type": "auto",
            },
            Scenario.RASTER: {
                "index_type": "ndvi",
            },
        }

    def orchestrate(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        intent_result: Optional[IntentResult] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> OrchestrationResult:
        """
        编排用户输入

        Args:
            text: 用户输入的自然语言
            context: 上下文信息（如地图点击位置、已选图层等）
            intent_result: 预计算的意图结果（可选）
            event_callback: 事件回调函数

        Returns:
            OrchestrationResult 对象
        """
        # 1. 意图分类
        if intent_result is None:
            intent_result = self.intent_classifier.classify(text)

        scenario = intent_result.primary

        if event_callback:
            event_callback("intent_classified", {
                "scenario": scenario.value,
                "confidence": intent_result.confidence,
                "matched_keywords": intent_result.matched_keywords,
            })

        # 2. 提取参数
        extracted_params = self.parameter_extractor.extract_all(text, scenario)

        if event_callback:
            event_callback("params_extracted", {
                "extracted_params": {k: v for k, v in extracted_params.items() if k not in ("locations",)},
                "scenario": scenario.value,
            })

        # 合并上下文
        if context:
            extracted_params.update(context)

        # 合并自动填充的默认值
        defaults = self._scenario_defaults.get(scenario, {})
        for key, value in defaults.items():
            if key not in extracted_params or not extracted_params[key]:
                extracted_params[key] = value

        # 3. 检查缺失参数
        clarification = self.clarification_engine.check_params(scenario, extracted_params)

        if clarification.needs_clarification:
            return OrchestrationResult(
                status=PipelineStatus.CLARIFICATION_NEEDED,
                scenario=scenario,
                needs_clarification=True,
                questions=clarification.questions,
                auto_filled=clarification.auto_filled,
                extracted_params=extracted_params,
                intent_result=intent_result,
            )

        # 4. 参数完整，返回就绪结果
        return OrchestrationResult(
            status=PipelineStatus.ORCHESTRATED,
            scenario=scenario,
            needs_clarification=False,
            extracted_params=extracted_params,
            intent_result=intent_result,
        )


# =============================================================================
# 便捷函数
# =============================================================================

_orchestrator: Optional[ScenarioOrchestrator] = None


def get_orchestrator() -> ScenarioOrchestrator:
    """获取编排器单例"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ScenarioOrchestrator()
    return _orchestrator


def orchestrate(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    intent_result: Optional[IntentResult] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> OrchestrationResult:
    """
    便捷函数：编排用户输入

    这是第3层的标准出口函数。
    """
    orchestrator = get_orchestrator()
    return orchestrator.orchestrate(text, context, intent_result, event_callback)


__all__ = [
    "ClarificationQuestion",
    "ClarificationResult",
    "OrchestrationResult",
    "ParameterExtractor",
    "ClarificationEngine",
    "ScenarioOrchestrator",
    "CLARIFICATION_TEMPLATES",
    "get_orchestrator",
    "orchestrate",
]
