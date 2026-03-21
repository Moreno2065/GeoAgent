"""
Scenario Orchestrator - 场景编排器
====================================
第三层架构：决定请求属于哪类任务、是否需要追问、选择分析模板。

核心职责：
1. 意图分类 → 确定场景
2. 参数提取 → 从自然语言中提取关键参数
3. 追问判断 → 参数不完整时生成追问
4. 模板选择 → 选择对应的分析模板
5. 任务构建 → 构建标准化任务对象
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from geoagent.dsl.protocol import (
    GeoDSL,
    ScenarioType,
    OutputSpec,
    ClarificationQuestion,
    OrchestrationStatus,
)
from geoagent.compiler.intent_classifier import (
    IntentClassifier,
    IntentResult,
    ClarificationEngine,
    SCENARIO_SUBTYPES,
)
from geoagent.compiler.task_schema import TaskType


# =============================================================================
# 编排结果
# =============================================================================

@dataclass
class OrchestrationResult:
    """场景编排结果"""
    status: OrchestrationStatus
    scenario: Optional[str] = None
    task: Optional[GeoDSL] = None
    questions: List[ClarificationQuestion] = field(default_factory=list)
    auto_filled: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    intent_result: Optional[IntentResult] = None

    def __post_init__(self):
        # 确保 status 是 OrchestrationStatus 枚举
        if isinstance(self.status, str):
            self.status = OrchestrationStatus(self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "scenario": self.scenario,
            "task": self.task.model_dump() if self.task else None,
            "questions": [q.model_dump() for q in self.questions],
            "auto_filled": self.auto_filled,
            "error": self.error,
        }

    @property
    def needs_clarification(self) -> bool:
        return self.status == OrchestrationStatus.CLARIFICATION_NEEDED

    @property
    def is_ready(self) -> bool:
        return self.status == OrchestrationStatus.READY


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

    # 地点连接词
    LOCATION_CONNECTORS = [
        "到", "至", "→", "->", "从", "from", "to",
    ]

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

        # 尝试匹配 "从X到Y" 模式
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
        # 匹配常见的 GIS 文件扩展名
        extensions = [".shp", ".geojson", ".json", ".gpkg", ".tif", ".tiff", ".csv"]
        references = []

        query_lower = query.lower()
        for ext in extensions:
            if ext in query_lower:
                # 提取文件名
                pattern = rf"[\'\"\\]?([^\'\"\\\n]*{re.escape(ext)})[\'\"\\]?"
                matches = re.findall(pattern, query_lower)
                references.extend(matches)

        return list(set(references))

    def extract_value_field(self, query: str) -> Optional[str]:
        """从查询中提取数值字段名"""
        # 匹配常见的字段引用模式
        patterns = [
            r"(?:字段|field|属性)[:：]?\s*[\'\"\\]?(\w+)[\'\"\\]?",
            r"(?:按|根据|用)\s*[\'\"\\]?(\w+)[\'\"\\]?\s*(?:分析|计算|插值)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def extract_all(self, query: str, scenario: str) -> Dict[str, Any]:
        """
        从查询中提取所有相关参数

        覆盖所有 11 个场景。

        Args:
            query: 用户查询
            scenario: 场景类型

        Returns:
            提取的参数字典
        """
        params = {}

        # ── 通用提取（所有场景共用）───────────────────────────────────
        params["distance_info"] = self.extract_distance(query)
        params["time_info"] = self.extract_time(query)
        params["mode"] = self.extract_mode(query)
        params["locations"] = self.extract_locations(query)
        params["files"] = self.extract_file_references(query)
        params["value_field"] = self.extract_value_field(query)

        # 通用地点（起点/位置）
        params["start"] = params["locations"]["start"]
        params["end"] = params["locations"]["end"]
        params["location"] = params["start"]  # 中心点

        # ── route 场景 ───────────────────────────────────────────────
        if scenario == "route":
            params["start"] = params["locations"]["start"]
            params["end"] = params["locations"]["end"]
            # mode 已通过通用提取获取

        # ── buffer 场景 ─────────────────────────────────────────────
        elif scenario == "buffer":
            # 从文件名引用中提取图层名
            files = params["files"]
            if files:
                params["input_layer"] = files[0]
            # 距离已通过通用提取获取
            if params["distance_info"]:
                params["distance"] = params["distance_info"]["distance"]
                params["unit"] = params["distance_info"]["unit"]
            # 尝试从 query 中提取图层名（学校/地铁站/河流等）
            entity = self._extract_entity_name(query)
            if entity and not params.get("input_layer"):
                params["input_layer"] = entity

        # ── overlay 场景 ─────────────────────────────────────────────
        elif scenario == "overlay":
            files = params["files"]
            if len(files) >= 2:
                params["layer1"] = files[0]
                params["layer2"] = files[1]
            elif len(files) == 1:
                params["layer1"] = files[0]
            # 操作类型
            params["operation"] = self._extract_overlay_operation(query)

        # ── interpolation 场景 ───────────────────────────────────────
        elif scenario == "interpolation":
            files = params["files"]
            if files:
                params["input_points"] = files[0]
            # 字段名已通过通用提取获取
            # 方法
            params["method"] = self._extract_interpolation_method(query)
            # 分辨率
            resolution = self._extract_resolution(query)
            if resolution:
                params["output_resolution"] = resolution

        # ── accessibility 场景 ───────────────────────────────────────
        elif scenario == "accessibility":
            params["location"] = params["locations"]["start"]
            if params["time_info"]:
                params["time_threshold"] = params["time_info"]["time_threshold"]

        # ── suitability 场景 ────────────────────────────────────────
        elif scenario == "suitability":
            files = params["files"]
            if files:
                params["criteria_layers"] = files
            # 从 query 中提取区域/分析范围
            params["area"] = self._extract_area(query)
            # 权重（暂时不支持从 NL 提取）

        # ── viewshed 场景 ────────────────────────────────────────────
        elif scenario == "viewshed":
            params["location"] = params["locations"]["start"]
            if not params["location"]:
                # 尝试从 query 中提取坐标
                params["location"] = self._extract_coordinates(query)
            # 观察高度
            params["observer_height"] = self._extract_height(query, "observer")
            files = params["files"]
            if files:
                params["dem_file"] = files[0]
            # 目标半径
            if params["distance_info"]:
                params["max_distance"] = params["distance_info"]["distance"] * 1000 if params["distance_info"]["unit"] == "kilometers" else params["distance_info"]["distance"]

        # ── shadow_analysis 场景 ──────────────────────────────────────
        elif scenario == "shadow_analysis":
            files = params["files"]
            if files:
                params["buildings"] = files[0]
            # 时间
            params["time"] = self._extract_datetime(query)

        # ── ndvi 场景 ───────────────────────────────────────────────
        elif scenario == "ndvi":
            files = params["files"]
            if files:
                params["input_file"] = files[0]
            # 传感器类型（默认 auto）

        # ── hotspot 场景 ─────────────────────────────────────────────
        elif scenario == "hotspot":
            files = params["files"]
            if files:
                params["input_file"] = files[0]
            # 字段名已通过通用提取获取

        # ── visualization 场景 ───────────────────────────────────────
        elif scenario == "visualization":
            files = params["files"]
            if files:
                params["input_files"] = files

        return params

    def _extract_entity_name(self, query: str) -> Optional[str]:
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

    def _extract_overlay_operation(self, query: str) -> str:
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
        return "intersect"  # 默认

    def _extract_interpolation_method(self, query: str) -> str:
        """从查询中提取插值方法"""
        q = query.lower()
        if "kriging" in q or "克里金" in query:
            return "kriging"
        if "nearest" in q or "最近邻" in query:
            return "nearest_neighbor"
        if "idw" in q or "反距离" in query:
            return "IDW"
        return "IDW"  # 默认

    def _extract_resolution(self, query: str) -> Optional[float]:
        """从查询中提取分辨率"""
        patterns = [
            (r"(\d+(?:\.\d+)?)\s*(?:米|meters?|m)(?:分辨率|精度)?", 1.0),
            (r"(\d+(?:\.\d+)?)\s*(?:公里|km)(?:分辨率|精度)?", 1000.0),
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier
        return None

    def _extract_area(self, query: str) -> str:
        """从查询中提取分析区域"""
        patterns = [
            r"(?:在|分析|选址)\s*([^\s,，。]+?)\s*(?:区域|范围|内|做|进行|选址|分析)",
            r"(?:芜湖|合肥|南京|北京|上海|广州|深圳)(?:市|区|县)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_coordinates(self, query: str) -> Optional[str]:
        """从查询中提取坐标"""
        # 经度,纬度 格式
        pattern = r"(\d+\.?\d*)\s*[,，]\s*(\d+\.?\d*)"
        match = re.search(pattern, query)
        if match:
            return f"{match.group(1)},{match.group(2)}"
        return None

    def _extract_height(self, query: str, height_type: str = "observer") -> float:
        """从查询中提取高度"""
        patterns = [
            (r"高度\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            (r"(\d+(?:\.\d+)?)\s*米高", 1.0),
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier
        # 默认观察高度
        if height_type == "observer":
            return 1.7  # 人眼高度
        return 0.0

    def _extract_datetime(self, query: str) -> str:
        """从查询中提取日期时间"""
        # ISO8601 格式
        pattern = r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s*T?\s*\d{1,2}:\d{2}(?::\d{2})?)?)"
        match = re.search(pattern, query)
        if match:
            dt = match.group(1).replace("/", "-")
            if "T" not in dt and ":" in dt:
                dt = dt.replace(" ", "T")
            return dt
        # 尝试中文日期
        patterns = [
            r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})时",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                year, month, day, hour = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:00"
        # 默认当前时间
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%dT12:00")


# =============================================================================
# 场景编排器
# =============================================================================

class ScenarioOrchestrator:
    """
    场景编排器

    核心流程：
    1. 意图分类 → 确定场景类型
    2. 参数提取 → 从 NL 中提取关键参数
    3. 追问判断 → 参数不完整时生成追问
    4. 任务构建 → 构建标准 DSL 任务

    使用方式：
        orchestrator = ScenarioOrchestrator()
        result = orchestrator.orchestrate("芜湖南站到方特的步行路径")
        if result.is_ready:
            task = result.task  # GeoDSL 对象
        elif result.needs_clarification:
            for q in result.questions:
                print(q.question)
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.clarification_engine = ClarificationEngine()
        self.parameter_extractor = ParameterExtractor()
        self._scenario_defaults = self._init_defaults()

    def _init_defaults(self) -> Dict[str, Dict[str, Any]]:
        """初始化场景默认值"""
        return {
            "route": {
                "mode": "walking",
                "outputs": {"map": True, "summary": True},
            },
            "buffer": {
                "unit": "meters",
                "outputs": {"map": True, "summary": True},
            },
            "overlay": {
                "operation": "intersect",
                "outputs": {"map": True, "summary": True},
            },
            "interpolation": {
                "method": "IDW",
                "outputs": {"map": True, "summary": True},
            },
            "accessibility": {
                "mode": "walking",
                "time_unit": "minutes",
                "outputs": {"map": True, "summary": True},
            },
            "suitability": {
                "method": "weighted",
                "outputs": {"map": True, "summary": True},
            },
            "viewshed": {
                "observer_height": 1.7,
                "max_distance": 5000,
                "outputs": {"map": True, "summary": True},
            },
            "shadow_analysis": {
                "outputs": {"map": True, "summary": True},
            },
            "ndvi": {
                "sensor": "auto",
                "outputs": {"map": True, "summary": True},
            },
            "hotspot": {
                "analysis_type": "auto",
                "outputs": {"map": True, "summary": True},
            },
            "visualization": {
                "viz_type": "interactive_map",
                "outputs": {"map": True, "summary": False},
            },
            "general": {
                "outputs": {"map": True, "summary": True},
            },
        }

    def orchestrate(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> OrchestrationResult:
        """
        编排用户输入

        Args:
            user_input: 用户输入的自然语言
            context: 上下文信息（如地图点击位置、已选图层等）
            event_callback: 事件回调函数

        Returns:
            OrchestrationResult 对象
        """
        # 1. 意图分类
        intent_result = self.intent_classifier.classify(user_input)
        scenario = intent_result.primary

        if event_callback:
            event_callback("intent_classified", {
                "scenario": scenario,
                "confidence": intent_result.confidence,
                "matched_keywords": intent_result.matched_keywords,
                "all_intents": list(intent_result.all_intents),
            })

        # 2. 提取参数
        extracted_params = self.parameter_extractor.extract_all(user_input, scenario)

        if event_callback:
            event_callback("params_extracted", {
                "extracted_params": {k: v for k, v in extracted_params.items() if k not in ("locations",)},
                "scenario": scenario,
            })

        # 合并上下文
        if context:
            extracted_params.update(context)

        # 3. 检查缺失参数
        clarification = self.clarification_engine.check_params(scenario, extracted_params)

        if clarification.needs_clarification:
            return OrchestrationResult(
                status=OrchestrationStatus.CLARIFICATION_NEEDED,
                scenario=scenario,
                questions=clarification.questions,
                auto_filled=clarification.auto_filled,
                intent_result=intent_result,
            )

        # 4. 构建任务 DSL
        task = self._build_task(scenario, extracted_params)

        return OrchestrationResult(
            status=OrchestrationStatus.READY,
            scenario=scenario,
            task=task,
            intent_result=intent_result,
        )

    def _build_task(
        self,
        scenario: str,
        params: Dict[str, Any],
    ) -> GeoDSL:
        """构建 GeoDSL 任务"""
        defaults = self._scenario_defaults.get(scenario, {})
        outputs_config = defaults.pop("outputs", {"map": True, "summary": True})

        # 构建 inputs
        inputs = {}
        parameters = {}

        if scenario == "route":
            inputs["start"] = params.get("start", "")
            inputs["end"] = params.get("end", "")
            parameters["mode"] = params.get("mode", defaults.get("mode", "walking"))
            parameters["provider"] = "auto"

        elif scenario == "buffer":
            inputs["input_layer"] = params.get("input_layer", params.get("files", [""])[0] if params.get("files") else "")
            inputs["distance"] = params.get("distance", params.get("distance_info", {}).get("distance", 500) if params.get("distance_info") else 500)
            parameters["unit"] = params.get("unit", defaults.get("unit", "meters"))
            parameters["dissolve"] = False

        elif scenario == "overlay":
            files = params.get("files", [])
            inputs["layer1"] = files[0] if len(files) > 0 else params.get("layer1", "")
            inputs["layer2"] = files[1] if len(files) > 1 else params.get("layer2", "")
            parameters["operation"] = params.get("operation", defaults.get("operation", "intersect"))

        elif scenario == "interpolation":
            inputs["input_points"] = params.get("input_points", params.get("files", [""])[0] if params.get("files") else "")
            inputs["value_field"] = params.get("value_field", "")
            parameters["method"] = params.get("method", defaults.get("method", "IDW"))
            parameters["power"] = 2.0
            parameters["output_resolution"] = params.get("output_resolution", 100)

        elif scenario == "accessibility":
            inputs["location"] = params.get("start", params.get("location", ""))
            parameters["mode"] = params.get("mode", defaults.get("mode", "walking"))
            parameters["time_threshold"] = params.get("time_threshold", params.get("time_info", {}).get("time_threshold", 15) if params.get("time_info") else 15)
            parameters["grid_resolution"] = 50

        elif scenario == "suitability":
            inputs["criteria_layers"] = params.get("files", [])
            inputs["area"] = params.get("area", "")
            parameters["method"] = defaults.get("method", "weighted")
            parameters["weights"] = params.get("weights", {})

        elif scenario == "viewshed":
            inputs["location"] = params.get("location", "")
            inputs["observer_height"] = params.get("observer_height", defaults.get("observer_height", 1.7))
            inputs["dem_file"] = params.get("dem_file", params.get("files", [""])[0] if params.get("files") else "")
            parameters["max_distance"] = params.get("max_distance", defaults.get("max_distance", 5000))
            parameters["target_height"] = params.get("target_height", 0)

        elif scenario == "shadow_analysis":
            inputs["buildings"] = params.get("buildings", params.get("files", [""])[0] if params.get("files") else "")
            parameters["time"] = params.get("time", datetime.now().strftime("%Y-%m-%dT12:00"))

        elif scenario == "ndvi":
            inputs["input_file"] = params.get("input_file", params.get("files", [""])[0] if params.get("files") else "")
            parameters["sensor"] = defaults.get("sensor", "auto")

        elif scenario == "hotspot":
            inputs["input_file"] = params.get("input_file", params.get("files", [""])[0] if params.get("files") else "")
            inputs["value_field"] = params.get("value_field", "")
            parameters["analysis_type"] = defaults.get("analysis_type", "auto")

        elif scenario == "visualization":
            inputs["input_files"] = params.get("files", [])
            parameters["viz_type"] = defaults.get("viz_type", "interactive_map")

        else:  # general
            inputs["description"] = params.get("description", "")

        # 构建 outputs
        outputs = OutputSpec(**outputs_config)

        # 构建 GeoDSL
        try:
            scenario_enum = ScenarioType(scenario)
        except ValueError:
            scenario_enum = ScenarioType.GENERAL

        return GeoDSL(
            version="1.0",
            scenario=scenario_enum,
            task=scenario,
            inputs=inputs,
            parameters=parameters,
            outputs=outputs,
        )

    def orchestrate_with_answers(
        self,
        user_input: str,
        answers: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> OrchestrationResult:
        """
        使用追问答案进行编排

        Args:
            user_input: 原始用户输入
            answers: 追问的答案
            context: 上下文信息
            event_callback: 事件回调函数

        Returns:
            OrchestrationResult 对象
        """
        # 合并答案到上下文
        if context:
            merged_context = {**context, **answers}
        else:
            merged_context = answers

        return self.orchestrate(user_input, merged_context, event_callback)


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
    user_input: str,
    context: Optional[Dict[str, Any]] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> OrchestrationResult:
    """
    便捷函数：编排用户输入

    Args:
        user_input: 用户输入的自然语言
        context: 上下文信息
        event_callback: 事件回调函数

    Returns:
        OrchestrationResult 对象
    """
    orchestrator = get_orchestrator()
    return orchestrator.orchestrate(user_input, context, event_callback)


__all__ = [
    "ScenarioOrchestrator",
    "OrchestrationResult",
    "ParameterExtractor",
    "get_orchestrator",
    "orchestrate",
]
