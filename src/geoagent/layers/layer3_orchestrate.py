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
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


from geoagent.layers.architecture import Scenario, PipelineStatus


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
# 澄清模板
# =============================================================================

CLARIFICATION_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "route": {
        "start": {"question": "请问起点位置是哪里？", "options": None, "required": True, "examples": ["芜湖南站", "市中心"]},
        "end": {"question": "请问终点位置是哪里？", "options": None, "required": True, "examples": ["方特欢乐世界", "高铁站"]},
        "mode": {"question": "请问出行方式是？", "options": ["步行", "驾车", "公交", "骑行"], "required": False, "default": "walking"},
    },
    "buffer": {
        "input_layer": {"question": "请问要对哪个图层做缓冲分析？", "options": None, "required": True, "examples": ["学校", "地铁站"]},
        "distance": {"question": "请问缓冲半径是多少米？", "options": ["500米", "1公里", "2公里", "5公里"], "required": True, "examples": ["500", "1000"]},
        "unit": {"question": "请问距离单位是？", "options": ["米", "公里"], "required": False, "default": "meters"},
    },
    "overlay": {
        "layer1": {"question": "请选择第一个图层？", "options": None, "required": True, "examples": ["土地利用", "行政区划"]},
        "layer2": {"question": "请选择第二个图层？", "options": None, "required": True, "examples": ["洪涝区", "保护区"]},
        "operation": {"question": "请问要进行什么叠加操作？", "options": ["交集", "并集", "裁剪", "差集"], "required": False, "default": "intersect"},
    },
    "accessibility": {
        "location": {"question": "请问分析的中心位置是哪里？", "options": None, "required": True, "examples": ["芜湖南站", "市中心"]},
        "mode": {"question": "请问用什么交通方式？", "options": ["步行", "驾车", "骑行"], "required": False, "default": "walking"},
        "time_threshold": {"question": "请问分析多长时间的可达范围？", "options": ["5分钟", "10分钟", "15分钟", "30分钟"], "required": True, "examples": ["5", "10", "15"]},
    },
}


# =============================================================================
# 澄清引擎
# =============================================================================

class ClarificationEngine:
    """检查参数是否完整，生成追问问题"""

    def __init__(self):
        self.templates = CLARIFICATION_TEMPLATES

    def check_params(self, scenario: Scenario, extracted_params: Dict[str, Any]) -> ClarificationResult:
        """
        检查参数是否完整，生成追问问题
        """
        scenario_str = scenario.value if hasattr(scenario, "value") else str(scenario)
        template = self.templates.get(scenario_str, {})
        if not template:
            return ClarificationResult(needs_clarification=False)

        questions = []
        auto_filled = {}

        for field_name, spec in template.items():
            if field_name not in extracted_params or not extracted_params[field_name]:
                if spec.get("required", True):
                    questions.append(ClarificationQuestion(
                        field=field_name,
                        question=spec["question"],
                        options=spec.get("options"),
                        required=True,
                    ))
                else:
                    default = spec.get("default")
                    if default:
                        auto_filled[field_name] = default

        return ClarificationResult(
            needs_clarification=len(questions) > 0,
            questions=questions,
            auto_filled=auto_filled,
        )


# =============================================================================
# 参数提取器
# =============================================================================

class ParameterExtractor:
    """从自然语言中提取关键参数"""

    DISTANCE_PATTERNS = [
        (r"方圆\s*(\d+(?:\.\d+)?)\s*(?:公里|km|kilometers?)\b", "kilometers"),
        (r"(\d+(?:\.\d+)?)\s*(?:公里|km|kilometers?|千米)\b", "kilometers"),
        (r"方圆\s*(\d+(?:\.\d+)?)\s*(?:米|meters?|m)\b", "meters"),
        (r"(\d+(?:\.\d+)?)\s*(?:米|meters?|m)\b", "meters"),
        (r"(\d+(?:\.\d+)?)\s*km\b", "kilometers"),
        (r"(\d+(?:\.\d+)?)\s*m\b(?![a-z])", "meters"),
        (r"半径\s*(\d+(?:\.\d+)?)\s*(?:公里|km)\b", "kilometers"),
        (r"半径\s*(\d+(?:\.\d+)?)\s*(?:米|m)\b", "meters"),
    ]

    TIME_PATTERNS = [
        (r"(\d+(?:\.\d+)?)\s*分钟\b", "minutes"),
        (r"(\d+(?:\.\d+)?)\s*min(?:ute)?s?\b", "minutes"),
        (r"(\d+(?:\.\d+)?)\s*小时\b", "hours"),
        (r"(\d+)\s*小时\s*(\d+)\s*分钟", "compound_hours"),
    ]

    MODE_KEYWORDS = {
        "walking": ["步行", "走路", "徒步", "walk", "walking", "人行"],
        "driving": ["驾车", "开车", "自驾", "drive", "driving", "行驶"],
        "transit": ["公交", "地铁", "公共交通", "transit", "bus", "metro"],
        "cycling": ["骑行", "骑车", "bike", "cycling", "自行车", "单车"],
    }

    GIS_EXTENSIONS = [".shp", ".geojson", ".json", ".gpkg", ".tif", ".tiff", ".kml", ".gml", ".xyz"]

    def extract_distance(self, query: str) -> Optional[Dict[str, Any]]:
        """从查询中提取距离参数"""
        for pattern, unit in self.DISTANCE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return {"distance": float(match.group(1)), "unit": unit}
        return None

    def extract_time(self, query: str) -> Optional[Dict[str, Any]]:
        """从查询中提取时间参数"""
        compound_match = re.search(r"(\d+)\s*小时\s*(\d+)\s*分钟", query)
        if compound_match:
            total = int(compound_match.group(1)) * 60 + int(compound_match.group(2))
            return {"time_threshold": float(total), "time_unit": "minutes"}
        for pattern, unit in self.TIME_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if unit == "hours":
                    value *= 60
                return {"time_threshold": value, "time_unit": "minutes"}
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
            r"从\s*(.+?)\s*(?:到|至|→|->|=>)\s*(.+?)(?:\s*的|\s*路|\s*线路|\s*导航|$)",
            r"(?:^|[^从])\s*(.+?)\s*(?:到|至)\s*(.+?)(?:\s*的|\s*路|\s*线路|\s*导航|\s*步行|\s*驾车|\s*骑行|\s*公交|$)",
            r"起点\s*(.+?)\s*终点\s*(.+?)(?:\s*$|\s*的)",
            r"(?:from|origin|起点)\s*[:：]?\s*(.+?)\s*(?:to|end|终点|destination|目的地|→|->)\s*[:：]?\s*(.+?)(?:\s*$|\s*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match and match.group(1).strip() != match.group(2).strip():
                result["start"] = match.group(1).strip()
                result["end"] = match.group(2).strip()
                break
        return result

    def extract_file_references(self, query: str) -> List[str]:
        """从查询中提取文件引用"""
        references = []
        query_lower = query.lower()
        for ext in self.GIS_EXTENSIONS:
            if ext in query_lower:
                pattern = rf"[\'\"\\]?([^\'\"\\\n]*{re.escape(ext)})[\'\"\\]?"
                matches = re.findall(pattern, query_lower)
                references.extend(matches)
        return list(set(references))

    def extract_value_field(self, query: str) -> Optional[str]:
        """从查询中提取数值字段名"""
        patterns = [
            r"(?:字段|field|属性|指标|列)\s*[:：]?\s*[\'\"\\]?([\w\u4e00-\u9fff]+)[\'\"\\]?",
            r"(?:按|根据|用|以)\s*[\'\"\\]?([\w\u4e00-\u9fff]+)[\'\"\\]?\s*(?:分析|计算|插值|统计)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def extract_coordinates(self, query: str) -> Optional[str]:
        """从查询中提取坐标"""
        patterns = [
            (r"([+-]?\d+\.?\d*)\s*[,，]\s*([+-]?\d+\.?\d*)", "standard"),
        ]
        for pattern, style in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                lon, lat = match.group(1), match.group(2)
                try:
                    lon_f, lat_f = float(lon), float(lat)
                    if -180 <= lon_f <= 180 and -90 <= lat_f <= 90:
                        return f"{lon_f},{lat_f}"
                except ValueError:
                    continue
        return None

    def extract_all(self, query: str, scenario: Any) -> Dict[str, Any]:
        """从查询中提取所有相关参数"""
        scenario_str = scenario.value if hasattr(scenario, "value") else str(scenario)
        scenario_str = scenario_str.lower().strip()

        params: Dict[str, Any] = {}
        params["distance_info"] = self.extract_distance(query)
        params["time_info"] = self.extract_time(query)
        params["mode"] = self.extract_mode(query)
        params["locations"] = self.extract_locations(query)
        params["files"] = self.extract_file_references(query)
        params["value_field"] = self.extract_value_field(query)
        params["coordinates"] = self.extract_coordinates(query)
        params["start"] = params["locations"].get("start")
        params["end"] = params["locations"].get("end")
        params["location"] = params["start"]

        if scenario_str == "buffer":
            files = params["files"]
            if files:
                params["input_layer"] = files[0]
            if params["distance_info"]:
                params["distance"] = params["distance_info"]["distance"]
                params["unit"] = params["distance_info"]["unit"]

        elif scenario_str == "overlay":
            files = params["files"]
            if len(files) >= 2:
                params["layer1"] = files[0]
                params["layer2"] = files[1]
            elif len(files) == 1:
                params["layer1"] = files[0]

        elif scenario_str == "accessibility":
            params["location"] = params["locations"].get("start")
            if params["time_info"]:
                params["time_threshold"] = params["time_info"]["time_threshold"]

        elif scenario_str == "viewshed":
            params["location"] = params["locations"].get("start") or params["coordinates"]

        return params


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
    intent_result: Optional[Any] = None

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
    """

    def __init__(self):
        from geoagent.layers.layer2_intent import IntentClassifier
        self.intent_classifier = IntentClassifier()
        self.clarification_engine = ClarificationEngine()
        self.parameter_extractor = ParameterExtractor()
        self._scenario_defaults = self._init_defaults()

    def _init_defaults(self) -> Dict[Scenario, Dict[str, Any]]:
        """初始化场景默认值"""
        return {
            Scenario.ROUTE: {"mode": "walking"},
            Scenario.BUFFER: {"unit": "meters"},
            Scenario.OVERLAY: {"operation": "intersect"},
            Scenario.INTERPOLATION: {"method": "idw", "power": 2.0, "resolution": 100},
            Scenario.VIEWSHED: {"observer_height": 1.7, "max_distance": 5000},
            Scenario.STATISTICS: {"analysis_type": "auto"},
            Scenario.RASTER: {"index_type": "ndvi"},
        }

    def orchestrate(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        intent_result: Optional[Any] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> OrchestrationResult:
        """
        编排用户输入
        """
        if intent_result is None:
            intent_result = self.intent_classifier.classify(text)

        scenario = intent_result.primary

        if event_callback:
            event_callback("intent_classified", {
                "scenario": scenario.value,
                "confidence": intent_result.confidence,
                "matched_keywords": intent_result.matched_keywords,
            })

        extracted_params = self.parameter_extractor.extract_all(text, scenario)

        if event_callback:
            event_callback("params_extracted", {
                "extracted_params": {k: v for k, v in extracted_params.items() if k not in ("locations",)},
                "scenario": scenario.value,
            })

        if context:
            extracted_params.update(context)

        defaults = self._scenario_defaults.get(scenario, {})
        for key, value in defaults.items():
            if key not in extracted_params or not extracted_params[key]:
                extracted_params[key] = value

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
    intent_result: Optional[Any] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> OrchestrationResult:
    """便捷函数：编排用户输入"""
    orchestrator = get_orchestrator()
    return orchestrator.orchestrate(text, context, intent_result, event_callback)


__all__ = [
    "ClarificationQuestion",
    "ClarificationResult",
    "ClarificationEngine",
    "ParameterExtractor",
    "OrchestrationResult",
    "ScenarioOrchestrator",
    "get_orchestrator",
    "orchestrate",
]
