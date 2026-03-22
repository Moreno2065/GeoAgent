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

    # ── GIS 文件扩展名（已扩展）────────────────────────────────────────
    GIS_EXTENSIONS = [
        ".shp", ".geojson", ".json", ".gpkg", ".tif", ".tiff",
        ".kml", ".gml", ".xyz", ".topojson", ".osm",
        ".csv", ".xlsx", ".xls", ".fgb", ".mvt",
    ]

    # ── 中文动词（用于过滤文件引用误匹配）────────────────────────────────
    CHINESE_VERBS = [
        "叠加", "解析", "读取", "写入", "保存", "转换",
        "打开", "导入", "导出", "裁剪", "合并", "分析", "处理",
        "叠加", "融合", "生成", "制作", "创建",
    ]

    # ── 距离提取模式──────────────────────────────────────────────────
    # 关键：\b 是 ASCII 单词边界，不识别中文字符！
    # 所有以中文结尾的模式必须移除 \b
    DISTANCE_PATTERNS = [
        # 公里/千米
        (r"(\d+(?:\.\d+)?)\s*(?:公里|km|kilometers?)", "kilometers"),
        (r"方圆\s*(\d+(?:\.\d+)?)\s*(?:公里|km)", "kilometers"),
        (r"(\d+(?:\.\d+)?)\s*(?:公里|km)\s*(?:范围|内|以外)", "kilometers"),
        # 米
        (r"(\d+(?:\.\d+)?)\s*(?:米|meters?)", "meters"),
        (r"(\d+(?:\.\d+)?)\s*m\b(?![a-zA-Z])", "meters"),  # buffer 200m (英文)
        (r"方圆\s*(\d+(?:\.\d+)?)\s*(?:米|m)", "meters"),
        # 独立数字（语境判断：周边/缓冲/方圆 + 数字）
        (r"(?:周边|方圆|半径|范围|距离|缓冲)\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)?", "meters"),
    ]

    # ── 时间提取模式──────────────────────────────────────────────────
    TIME_PATTERNS = [
        (r"(\d+(?:\.\d+)?)\s*分钟", "minutes"),
        (r"(\d+(?:\.\d+)?)\s*min\b", "minutes"),
        (r"(\d+(?:\.\d+)?)\s*小时", "hours"),
        (r"(\d+(?:\.\d+)?)\s*h\b", "hours"),
        (r"(\d+(?:\.\d+)?)\s*天", "days"),
    ]

    # ── 交通模式关键词────────────────────────────────────────────────
    MODE_KEYWORDS = {
        "walking": ["步行", "走路", "徒步", "walk", "walking"],
        "driving": ["驾车", "开车", "自驾", "drive", "driving"],
        "transit": ["公交", "地铁", "公共交通", "transit", "bus"],
        "cycling": ["骑行", "骑车", "bike", "cycling"],
    }

    def extract_distance(self, query: str) -> Optional[Dict[str, Any]]:
        """从查询中提取距离参数"""
        for pattern, unit in self.DISTANCE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if unit == "kilometers":
                    value *= 1000  # 转换为米
                    unit = "meters"
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
                    value *= 60
                elif unit == "days":
                    value *= 1440
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

    def extract_city(self, query: str) -> str:
        """提取自然语言中的城市"""
        match = re.search(r"(?:在)?([\u4e00-\u9fa5]{2,5}市)", query)
        if match:
            return match.group(1)
        return ""

    def extract_numeric_param(self, query: str, unit: str = None) -> Optional[float]:
        """从查询中提取数值参数（通用）"""
        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:米|m|km|seconds?|分钟|min|小时|h|days?|天)",
            r"(\d+(?:\.\d+)?)\s*m\s*(?![a-zA-Z])",  # e.g. 500m (ASCII)
            r"(\d+(?:\.\d+)?)",  # 纯数字备用
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return float(match.group(1))
        return None

    def extract_locations(self, query: str) -> Dict[str, Optional[str]]:
        """从查询中提取起点和终点"""
        result = {"start": None, "end": None}

        # 模式按优先级排序（最宽松的放最后兜底）
        # 关键：\s* 两端都允许零个空格，因为中文经常没有空格分隔
        patterns = [
            # 1. 标准中文：从X到Y / 从X至Y（到某地的路径）
            (r"\u4ece\s*(.+?)\s*(?:\u5230|\u81f3|\u2192|->|=>)\s*(.+?)(?:\s*\u7684|\s*\u8def|\s*\u7ebf\u8def|\s*\u5bfc\u822a|$)", None),
            # 2. 直接分隔符（无"从"前缀）：芜湖南站到方特欢乐世界的步行路径
            # .*? 非贪婪捕获终点，后接 "的..." 或行尾；否则回退到贪婪
            (r"^(.*?)\s*(?:\u5230|\u81f3)\s*(.*?)(?:\s*\u7684.*|$)", None),
            # 3. 英文：from X to Y / origin:end
            (r"(?:from|origin|起点)\s*[:\uff1a]?\s*(.+?)\s*(?:to|end|终点|destination|目的地)\s*[:\uff1a]?\s*(.+?)(?:\s*$|\s*\?|\s*\u7684)", None),
        ]

        for pattern, _ in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match and match.lastindex is not None and match.lastindex >= 2:
                start = match.group(1).strip()
                end = match.group(2).strip()
                # 过滤掉太短的或明显不是地点的
                if len(start) >= 2 and len(end) >= 2:
                    result["start"] = start
                    result["end"] = end
                    break

        return result

    def extract_file_references(self, query: str) -> List[str]:
        """从查询中提取 GIS 文件引用（修复中文文件名识别）"""
        references = []
        query_lower = query.lower()

        for ext in self.GIS_EXTENSIONS:
            if ext.lower() not in query_lower:
                continue

            # 【修复点 🚀】加入中文字符范围 \u4e00-\u9fa5
            # 允许文件名包含：中文字符、字母、数字、下划线、连字符、点
            pattern = rf"([\u4e00-\u9fa5a-zA-Z0-9_\-\.]+?{re.escape(ext)})"

            matches = re.findall(pattern, query_lower)
            for m in matches:
                stripped = m.strip()
                if 3 < len(stripped) < 200:
                    references.append(stripped)

        # 去重
        seen = set()
        unique = []
        for r in sorted(references, key=len):
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique

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

    def extract_coordinates(self, query: str) -> Optional[str]:
        """从查询中提取坐标"""
        patterns = [
            # 标准格式：lon, lat / lon，lat
            (r"([+-]?\d+\.?\d*)\s*[,，]\s*([+-]?\d+\.?\d*)", "standard"),
            # 带括号：(118.5, 31.2)
            (r"\(\s*([+-]?\d+\.?\d*)\s*[,，]\s*([+-]?\d+\.?\d*)\s*\)", "bracketed"),
            # 经纬度标注（逗号分隔）：经度X，纬度Y / lon X, lat Y
            (r"经[度]?\s*([+-]?\d+\.?\d*)[度]?\s*[，,]\s*纬[度]?\s*([+-]?\d+\.?\d*)[度]?", "labeled_cn_comma"),
            (r"lon[gitude]?\s*([+-]?\d+\.?\d*)\s*[，,]\s*lat[gitude]?\s*([+-]?\d+\.?\d*)", "labeled_en_comma"),
            # 经纬度标注（空格分隔）：经度X 纬度Y / 经 X 纬 Y
            (r"经[度]?\s*([+-]?\d+\.?\d*)[度]?\s+纬[度]?\s*([+-]?\d+\.?\d*)[度]?", "labeled_cn_space"),
            (r"lon[gitude]?\s*([+-]?\d+\.?\d*)\s+lat[gitude]?\s*([+-]?\d+\.?\d*)", "labeled_en_space"),
        ]
        for pattern, style in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                lon = match.group(1)
                lat = match.group(2)
                # 验证合理性（经度 -180~180，纬度 -90~90）
                try:
                    lon_f = float(lon)
                    lat_f = float(lat)
                    if -180 <= lon_f <= 180 and -90 <= lat_f <= 90:
                        return f"{lon_f},{lat_f}"
                except ValueError:
                    continue
        return None

    def extract_resolution(self, query: str) -> Optional[float]:
        """从查询中提取栅格分辨率"""
        patterns = [
            # 分辨率X米/精度X米
            (r"(\d+(?:\.\d+)?)\s*(?:米|m)\s*(?:分辨率|精度)", 1.0),
            # 栅格大小/像元大小
            (r"(\d+(?:\.\d+)?)\s*(?:米|m)\s*(?:栅格|像元|像元大小|格子)", 1.0),
            # 分辨率500m
            (r"(\d+(?:\.\d+)?)\s*m\s*(?:分辨率|精度)", 1.0),
            # 公里分辨率
            (r"(\d+(?:\.\d+)?)\s*km\s*(?:分辨率|精度)", 1000.0),
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier
        return None

    def extract_height(self, query: str, height_type: str = "observer") -> float:
        """从查询中提取高度"""
        patterns = [
            # 中文：高度/海拔/楼高 + 数字 + 单位（无 \b 因为中文字符后无边界）
            (r"高度\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            (r"(\d+(?:\.\d+)?)\s*米\s*(?:高|高程)?", 1.0),
            (r"海拔\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            (r"楼高\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            # 英文：height/elevation/altitude + 数字 + m/meters
            (r"(?:height|elevation|altitude)\s*[:\s]*(\d+(?:\.\d+)?)\s*(?:m|meters?)\b", 1.0),
            (r"(?:building\s*)?height\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)\b", 1.0),
            # 独立数字模式：仅数字+米（在特定语境词附近）
            (r"(?:楼|建筑|物体|塔)\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier
        # 有语境词的回退值
        if height_type == "observer":
            return 1.7
        return 0.0

    def extract_datetime(self, query: str) -> Optional[str]:
        """从查询中提取日期时间"""
        # ISO8601 格式
        pattern = r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s*T?\s*\d{1,2}:\d{2}(?::\d{2})?)?)"
        match = re.search(pattern, query)
        if match:
            dt = match.group(1).replace("/", "-")
            if "T" not in dt and ":" in dt:
                dt = dt.replace(" ", "T")
            return dt
        # 相对时间
        relative = {
            "now": "now",
            "today": "today",
            "tomorrow": "tomorrow",
        }
        for key, val in relative.items():
            if key in query.lower():
                return val
        return None

    def extract_all(self, query: str, scenario: "str | Scenario | None" = None) -> Dict[str, Any]:
        """
        从查询中提取所有相关参数

        覆盖所有 11 个场景。

        Args:
            query: 用户查询
            scenario: 场景类型（字符串或 Scenario 枚举）

        Returns:
            提取的参数字典
        """
        # 支持传入 Scenario 枚举或字符串
        if scenario is not None:
            _scenario_str = scenario.value if hasattr(scenario, "value") else str(scenario)
        else:
            _scenario_str = "general"

        params = {}

        # ── 通用提取（所有场景共用）───────────────────────────────────
        params["distance_info"] = self.extract_distance(query)
        params["time_info"] = self.extract_time(query)
        params["mode"] = self.extract_mode(query)
        params["locations"] = self.extract_locations(query)
        params["files"] = self.extract_file_references(query)
        params["value_field"] = self.extract_value_field(query)
        params["coordinates"] = self.extract_coordinates(query)
        params["resolution"] = self.extract_resolution(query)
        params["datetime"] = self.extract_datetime(query)
        params["city"] = self.extract_city(query)

        # 通用地点（起点/位置）【安全访问】
        locations = params.get("locations") or {}
        params["start"] = locations.get("start")
        params["end"] = locations.get("end")
        params["location"] = params.get("start")  # 中心点

        # ── route 场景 ───────────────────────────────────────────────
        if _scenario_str == "route":
            params["start"] = locations.get("start")
            params["end"] = locations.get("end")
            # mode 已通过通用提取获取

        # ── buffer 场景 ─────────────────────────────────────────────
        elif _scenario_str == "buffer":
            # 从文件名引用中提取图层名
            files = params.get("files", [])
            if files:
                params["input_layer"] = files[0]
            # 距离已通过通用提取获取【安全访问】
            distance_info = params.get("distance_info")
            if distance_info:
                params["distance"] = distance_info.get("distance")
                params["unit"] = distance_info.get("unit")
            # 尝试从 query 中提取图层名（学校/地铁站/河流等）
            entity = self._extract_entity_name(query)
            if entity and not params.get("input_layer"):
                params["input_layer"] = entity

        # ── overlay 场景 ─────────────────────────────────────────────
        elif _scenario_str == "overlay":
            files = params.get("files", [])
            if len(files) >= 2:
                params["layer1"] = files[0]
                params["layer2"] = files[1]
            elif len(files) == 1:
                params["layer1"] = files[0]
            # 操作类型
            params["operation"] = self._extract_overlay_operation(query)

        # ── interpolation 场景 ───────────────────────────────────────
        elif _scenario_str == "interpolation":
            files = params.get("files", [])
            if files:
                params["input_points"] = files[0]
            # 字段名已通过通用提取获取
            # 方法
            params["method"] = self._extract_interpolation_method(query)
            # 分辨率
            resolution = self.extract_resolution(query)
            if resolution:
                params["output_resolution"] = resolution

        # ── accessibility 场景 ───────────────────────────────────────
        elif _scenario_str == "accessibility":
            params["location"] = locations.get("start")
            time_info = params.get("time_info")
            if time_info:
                params["time_threshold"] = time_info.get("time_threshold")

        # ── suitability 场景 ────────────────────────────────────────
        elif _scenario_str == "suitability":
            files = params.get("files", [])
            if files:
                params["criteria_layers"] = files
            # 从 query 中提取区域/分析范围
            params["area"] = self._extract_area(query)
            # 权重（暂时不支持从 NL 提取）

        # ── viewshed 场景 ────────────────────────────────────────────
        elif _scenario_str == "viewshed":
            params["location"] = locations.get("start")
            if not params.get("location"):
                # 尝试从 query 中提取坐标
                params["location"] = self.extract_coordinates(query)
            # 观察高度
            params["observer_height"] = self.extract_height(query, "observer")
            files = params.get("files", [])
            if files:
                params["dem_file"] = files[0]
            # 目标半径【安全访问】
            distance_info = params.get("distance_info")
            if distance_info:
                distance = distance_info.get("distance", 0)
                unit = distance_info.get("unit", "meters")
                params["max_distance"] = distance * 1000 if unit == "kilometers" else distance

        # ── shadow_analysis 场景 ──────────────────────────────────────
        elif _scenario_str == "shadow_analysis":
            files = params.get("files", [])
            if files:
                params["buildings"] = files[0]
            # 时间
            params["time"] = self.extract_datetime(query)

        # ── ndvi 场景 ───────────────────────────────────────────────
        elif _scenario_str == "ndvi":
            files = params.get("files", [])
            if files:
                params["input_file"] = files[0]
            # 传感器类型（默认 auto）

        # ── hotspot 场景 ─────────────────────────────────────────────
        elif _scenario_str == "hotspot":
            files = params.get("files", [])
            if files:
                params["input_file"] = files[0]
            # 字段名已通过通用提取获取

        # ── visualization 场景 ───────────────────────────────────────
        elif _scenario_str == "visualization":
            files = params.get("files", [])
            if files:
                params["input_files"] = files

        return params

    def _extract_entity_name(self, query: str) -> Optional[str]:
        """从查询中提取实体名称（修复"半径"误判）"""
        patterns = [
            r"(?:在|对|以|给)\s*([^\s,，。、]+?)\s*(?:周边|附近|方圆|做|进行)",
            r"([^\s,，。、]+?)\s*(?:周边|附近|方圆|周围)",
            # 【修复点 🚀】使用负向零宽断言 (?!...)，排除掉"半径"、"距离"等干扰词
            r"(?:分析|缓冲|找)\s*(?!半径|距离|范围|大小)([^\s,，。、]+)",
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
        scenario_str = intent_result.primary

        # ==========================================
        # 🚨 哈基米强插机制：最高指令直接接管！ (Priority Override)
        # ==========================================
        # 无论 L2 的词袋模型猜出了什么（哪怕是 0.99 的 overlay），
        # 只要用户的原话中出现了明确的"沙盒/代码"显式调用指令，直接夺权！
        sandbox_dictators = ["用代码", "沙盒", "写一段代码", "写脚本", "写python"]
        if any(trigger in user_input.lower() for trigger in sandbox_dictators):
            print(f"⚡ [最高权限] 检测到用户显式要求编程，强制剥夺 '{scenario_str}' 的执行权，路由至 code_sandbox！")
            scenario_str = "code_sandbox"

        # ==========================================
        # 🚨 哈基米防幻觉安检门：暴力拦截无效输入
        # ==========================================
        is_garbage_input = user_input.strip().isdigit() or len(user_input.strip()) < 2
        if scenario_str == "general" and is_garbage_input:
            # 物理切断流水线，不给大模型任何发癫的机会
            return OrchestrationResult(
                status=OrchestrationStatus.READY,
                scenario="unknown",
                error="本哈基米是一个严肃的🌍空间智能引擎！请发送明确的地理指令（例如：帮我查一下从芜湖南站到方特的路径）。",
                intent_result=intent_result
            )
        # ==========================================

        if event_callback:
            event_callback("intent_classified", {
                "scenario": scenario_str,
                "confidence": intent_result.confidence,
                "matched_keywords": intent_result.matched_keywords,
                "all_intents": list(intent_result.all_intents),
            })

        # 2. 提取参数
        extracted_params = self.parameter_extractor.extract_all(user_input, scenario_str)

        if event_callback:
            event_callback("params_extracted", {
                "extracted_params": {k: v for k, v in extracted_params.items() if k not in ("locations",)},
                "scenario": scenario_str,
            })

        # 合并上下文
        if context:
            extracted_params.update(context)

        # 3. 检查缺失参数
        clarification = self.clarification_engine.check_params(scenario_str, extracted_params)

        if clarification.needs_clarification:
            return OrchestrationResult(
                status=OrchestrationStatus.CLARIFICATION_NEEDED,
                scenario=scenario_str,
                questions=clarification.questions,
                auto_filled=clarification.auto_filled,
                intent_result=intent_result,
            )

        # 4. 构建任务 DSL
        task = self._build_task(scenario_str, extracted_params, user_input)

        return OrchestrationResult(
            status=OrchestrationStatus.READY,
            scenario=scenario_str,
            task=task,
            intent_result=intent_result,
        )

    def _build_task(
        self,
        scenario: str,
        params: Dict[str, Any],
        user_input: str = "",
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
            parameters["city"] = params.get("city", "")

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

        elif scenario == "code_sandbox":
            inputs["instruction"] = user_input  # 用户原始指令透传给 LLM
            parameters["timeout_seconds"] = 60.0
            parameters["mode"] = "exec"

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
