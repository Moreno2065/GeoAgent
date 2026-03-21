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
- 【防御性编程】所有字典访问使用 .get() 安全取值
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


from geoagent.layers.architecture import Scenario, PipelineStatus


# =============================================================================
# 安全工具函数
# =============================================================================

def _safe_get_value(obj: Any, default: str = "") -> str:
    """安全获取枚举值，处理 str/Enum/None 混合类型"""
    if obj is None:
        return default
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


# Alias for backward compatibility
_get_enum_value = _safe_get_value


def _safe_dict_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """安全地从字典中获取值"""
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


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
# 参数提取器（已合并 compiler/orchestrator.py 高级逻辑）
# =============================================================================

class ParameterExtractor:
    """
    从自然语言中提取关键参数【防御性版本】

    支持：
    - 地址/位置提取
    - 距离/范围提取
    - 模式提取（步行/驾车/骑行）
    - 时间提取
    - 文件名提取（含中文动词过滤）
    - 坐标提取（多种格式）
    - 分辨率/高度/日期时间提取

    【安全特性】：所有字典访问使用 .get() 链式调用，永不触发 KeyError
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
        patterns = [
            # 1. 标准中文：从X到Y / 从X至Y
            (r"\u4ece\s*(.+?)\s*(?:\u5230|\u81f3|\u2192|->|=>)\s*(.+?)(?:\s*\u7684|\s*\u8def|\s*\u7ebf\u8def|\s*\u5bfc\u822a|$)", None),
            # 2. 直接分隔符（无"从"前缀）
            (r"^(.*?)\s*(?:\u5230|\u81f3)\s*(.*?)(?:\s*\u7684.*|$)", None),
            # 3. 英文：from X to Y / origin:end
            (r"(?:from|origin|起点)\s*[:\uff1a]?\s*(.+?)\s*(?:to|end|终点|destination|目的地)\s*[:\uff1a]?\s*(.+?)(?:\s*$|\s*\?|\s*\u7684)", None),
        ]

        for pattern, _ in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match and match.lastindex is not None and match.lastindex >= 2:
                start = match.group(1).strip()
                end = match.group(2).strip()
                if len(start) >= 2 and len(end) >= 2:
                    result["start"] = start
                    result["end"] = end
                    break

        return result

    def extract_file_references(self, query: str) -> List[str]:
        """🚀 终极文件提取器：自动剥离介词，精准提取文件名"""
        references = []
        
        # ── 模式1：标准 GIS 扩展名匹配 ────────────────────────────────
        for ext in self.GIS_EXTENSIONS:
            if ext.lower() not in query.lower():
                continue
            
            # 核心魔法：(?:在|对|以|给|把|将|和|与)? 自动过滤掉前面的中文介词
            # [^\s,，。、的把和与对在给] 保证提取出来的纯粹是文件名
            pattern = rf"(?:在|对|以|给|把|将|和|与)?([^\s,，。、的把和与对在给]+?{re.escape(ext)})"
            matches = re.findall(pattern, query, re.IGNORECASE)
            
            for m in matches:
                if len(m) > 3:  # 过滤掉太短的错误匹配
                    references.append(m.strip())
        
        # ── 模式2：中文名 + 文件类型词（shp/SHP/图层/数据）──────────────
        # 处理 "土地利用点 shp" / "道路线 shp" / "河流数据 shp" 等情况
        gis_type_patterns = [
            r"(?:给|对|以|把|为)\s*([^\s,，。、]+?(?:点|线|面|区|数据|图层|文件))\s*(?:shp|SHP|\.shp)?",  # 介词 + 名称 + shp
            r"([^\s,，。、]+?(?:点|线|面|区|数据|图层|文件))\s*(?:shp|SHP)",  # 名称 + shp（不带扩展名）
        ]
        
        for pattern in gis_type_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for m in matches:
                # 清理可能的垃圾词
                cleaned = m.strip()
                
                # 🛡️ 过滤掉开头的介词（如"给土地利用点" -> "土地利用点"）
                prepositions = ['给', '对', '以', '把', '为']
                for prep in prepositions:
                    if cleaned.startswith(prep):
                        cleaned = cleaned[len(prep):].strip()
                
                if any(bad in cleaned.lower() for bad in ["缓冲", "半径", "距离", "增加", "生成", "分析"]):
                    continue
                if len(cleaned) >= 3:
                    references.append(cleaned)
        
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

    def extract_coordinates(self, query: str) -> Optional[str]:
        """从查询中提取坐标"""
        patterns = [
            # 标准格式：lon, lat / lon，lat
            (r"([+-]?\d+\.?\d*)\s*[,，]\s*([+-]?\d+\.?\d*)", "standard"),
            # 带括号：(118.5, 31.2)
            (r"\(\s*([+-]?\d+\.?\d*)\s*[,，]\s*([+-]?\d+\.?\d*)\s*\)", "bracketed"),
            # 经纬度标注
            (r"经[度]?\s*([+-]?\d+\.?\d*)[度]?\s*[，,]\s*纬[度]?\s*([+-]?\d+\.?\d*)[度]?", "labeled_cn_comma"),
            (r"lon[gitude]?\s*([+-]?\d+\.?\d*)\s*[，,]\s*lat[gitude]?\s*([+-]?\d+\.?\d*)", "labeled_en_comma"),
        ]
        for pattern, style in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                lon = match.group(1)
                lat = match.group(2)
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
            # 中文：高度/海拔/楼高 + 数字 + 单位
            (r"高度\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            (r"(\d+(?:\.\d+)?)\s*米\s*(?:高|高程)?", 1.0),
            (r"海拔\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            (r"楼高\s*(\d+(?:\.\d+)?)\s*(?:米|m)", 1.0),
            # 英文：height/elevation/altitude + 数字 + m/meters
            (r"(?:height|elevation|altitude)\s*[:\s]*(\d+(?:\.\d+)?)\s*(?:m|meters?)\b", 1.0),
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

    def _extract_entity_name(self, query: str) -> Optional[str]:
        """🛡️ 实体兜底提取：增加硬过滤"""
        patterns = [
            r"(?:在|对|以|给)\s*([^\s,，。、]+?)\s*(?:周边|附近|方圆|做|进行|缓冲|分析)",
            r"([^\s,，。、]+?)\s*(?:周边|附近|方圆|周围)",
            r"(?:分析|缓冲|找)\s*(?!半径|距离|米|m|buffer)([^\s,，。、]+)",
            # 🆕 新增：匹配 "xxx shp" / "xxx.shp" / "土地利用 shp" 等 GIS 文件描述
            r"(?:给|对|以|把)\s*([^\s,，。、]+?)\s*(?:shp|SHP|\.shp)\s*(?:这个)?(?:文件|图层|数据)?",
            r"([^\s,，。、]+?)\s*(?:shp|SHP)(?![a-zA-Z0-9])",  # 不跟字母数字的 shp
            # 🆕 新增：匹配 "土地利用点 shp" 这类中文名 + 文件类型词
            r"([^\s,，。、]+?(?:点|线|面|区|数据|图层|文件))\s*(?:shp|SHP|\.shp)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entity = match.group(1).strip()
                
                # 🛡️ 清理开头的介词（如"给土地利用点" -> "土地利用点"）
                prepositions = ['给', '对', '以', '把', '为']
                for prep in prepositions:
                    if entity.startswith(prep):
                        entity = entity[len(prep):].strip()
                
                # 🛡️ 终极硬过滤：如果提取出来的词包含以下任何垃圾词，直接扔掉！
                if any(bad in entity.lower() for bad in ["半径", "距离", "米", "m", "buffer", "缓冲", "分析"]):
                    continue
                    
                # 🛡️ 过滤掉太短的匹配
                if len(entity) < 2:
                    continue
                    
                return entity
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

    def extract_all(self, query: str, scenario: Any) -> Dict[str, Any]:
        """
        从查询中提取所有相关参数【防御性安全版本】

        覆盖所有 11 个场景。
        【安全特性】：所有字典访问使用 .get() 链式调用

        Args:
            query: 用户查询
            scenario: 场景类型（字符串或 Scenario 枚举）

        Returns:
            提取的参数字典
        """
        # 支持传入 Scenario 枚举或字符串
        _scenario_str = _safe_get_value(scenario, "general").lower().strip()

        params: Dict[str, Any] = {}

        # ── 通用提取（所有场景共用）───────────────────────────────────
        params["distance_info"] = self.extract_distance(query)
        params["time_info"] = self.extract_time(query)
        params["mode"] = self.extract_mode(query)
        params["locations"] = self.extract_locations(query) or {}
        params["files"] = self.extract_file_references(query)
        params["value_field"] = self.extract_value_field(query)
        params["coordinates"] = self.extract_coordinates(query)
        params["resolution"] = self.extract_resolution(query)
        params["datetime"] = self.extract_datetime(query)

        # ── 【安全】通用地点（使用 .get() 防止 KeyError）───────────────
        locations = params.get("locations", {}) or {}
        params["start"] = locations.get("start")
        params["end"] = locations.get("end")
        params["location"] = params.get("start")  # 中心点

        # ── route 场景 ───────────────────────────────────────────────
        if _scenario_str == "route":
            params["start"] = locations.get("start")
            params["end"] = locations.get("end")

        # ── buffer 场景 ─────────────────────────────────────────────
        elif _scenario_str == "buffer":
            files = params.get("files", [])
            if files:
                params["input_layer"] = files[0]
            # 【安全】距离信息使用链式 .get()
            distance_info = params.get("distance_info")
            if distance_info:
                params["distance"] = distance_info.get("distance")
                params["unit"] = distance_info.get("unit", "meters")
            # 尝试从 query 中提取图层名
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
            params["operation"] = self._extract_overlay_operation(query)

        # ── interpolation 场景 ───────────────────────────────────────
        elif _scenario_str == "interpolation":
            files = params.get("files", [])
            if files:
                params["input_points"] = files[0]
            resolution = params.get("resolution")
            if resolution:
                params["output_resolution"] = resolution
            params["method"] = self._extract_interpolation_method(query)

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
            params["area"] = self._extract_area(query)

        # ── viewshed 场景 ────────────────────────────────────────────
        elif _scenario_str == "viewshed":
            params["location"] = locations.get("start") or params.get("coordinates")
            params["observer_height"] = self.extract_height(query, "observer")
            files = params.get("files", [])
            if files:
                params["dem_file"] = files[0]
            # 【安全】距离信息
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
            params["time"] = params.get("datetime")

        # ── ndvi 场景 ───────────────────────────────────────────────
        elif _scenario_str == "ndvi":
            files = params.get("files", [])
            if files:
                params["input_file"] = files[0]

        # ── hotspot 场景 ─────────────────────────────────────────────
        elif _scenario_str == "hotspot":
            files = params.get("files", [])
            if files:
                params["input_file"] = files[0]

        # ── visualization 场景 ───────────────────────────────────────
        elif _scenario_str == "visualization":
            files = params.get("files", [])
            if files:
                params["input_files"] = files

        return params

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
        scenario_str = _safe_get_value(self.scenario)
        status_str = _safe_get_value(self.status)
        return {
            "status": status_str,
            "scenario": scenario_str,
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
            Scenario.ROUTE: {
                "mode": "walking",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.BUFFER: {
                "unit": "meters",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.OVERLAY: {
                "operation": "intersect",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.INTERPOLATION: {
                "method": "IDW",
                "power": 2.0,
                "resolution": 100,
                "outputs": {"map": True, "summary": True},
            },
            Scenario.VIEWSHED: {
                "observer_height": 1.7,
                "max_distance": 5000,
                "outputs": {"map": True, "summary": True},
            },
            Scenario.STATISTICS: {
                "analysis_type": "auto",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.RASTER: {
                "index_type": "ndvi",
                "sensor": "auto",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.ACCESSIBILITY: {
                "mode": "walking",
                "time_unit": "minutes",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.SHADOW_ANALYSIS: {
                "outputs": {"map": True, "summary": True},
            },
            Scenario.HOTSPOT: {
                "analysis_type": "auto",
                "outputs": {"map": True, "summary": True},
            },
            Scenario.VISUALIZATION: {
                "viz_type": "interactive_map",
                "outputs": {"map": True, "summary": False},
            },
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
                "scenario": _safe_get_value(scenario),
                "confidence": intent_result.confidence,
                "matched_keywords": intent_result.matched_keywords,
            })

        extracted_params = self.parameter_extractor.extract_all(text, scenario)

        if event_callback:
            event_callback("params_extracted", {
                "extracted_params": {k: v for k, v in extracted_params.items() if k not in ("locations",)},
                "scenario": _safe_get_value(scenario),
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
