"""
工具调用验证器 - 防幻觉终极版
================================
验证 LLM 是否真正调用了工具，防止"说谎"（幻觉）。

【核心功能】
- 检测 LLM 回复中声称的文件是否真实存在
- 检测捏造的数据（数量、坐标、面积等）
- 验证声称的工具调用是否有对应的实际调用
- 强制报告系统验证的真实文件路径

【验证流程】
1. 从 LLM 回复中提取声称的文件路径
2. 验证这些文件是否真实存在于文件系统
3. 如果发现声称的文件不存在，立即拦截并报告错误
4. 只允许报告经过系统确认存在的文件

【严厉措施】
- 严格模式：任何幻觉行为立即报错
- 输出拦截：只传递经过验证的文件路径
- 详细日志：记录所有验证过程便于审计
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

# =============================================================================
# 常量定义
# =============================================================================

# 捏造数据模式
FABRICATION_PATTERNS = [
    # 数量捏造：声称"约XX个"、"大约XX个"等模糊数量
    (r"[约大]约?\s*(\d+)\s*[个条座]", "vague_count"),
    # 坐标捏造：包含经纬度但不来自工具返回
    (r"\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\)", "bare_coordinates"),
    # 位置捏造："位于"、"坐落在"等
    (r"(位于|坐落在|处于|分布在).{0,50}(经纬度|坐标|位置)", "vague_location"),
    # 文件捏造：声称"已生成"、"已创建"、"已保存"但未验证
    (r"(已生成|已创建|已保存|已完成).{0,30}\.(html|json|shp|geojson|zip|png)", "unverified_file_claim"),
    # 路径捏造：未经确认的路径引用
    (r"(路径|文件)[是为]?\s*[:：]?\s*(/[a-zA-Z0-9_/.-]+\.[a-zA-Z]+)", "unverified_path"),
]

# 声称工具调用的模式
TOOL_CALL_CLAIM_PATTERNS = [
    r"已调用\s*\w+",
    r"已获取\s*\w+\s*数据",
    r"共[发现有]\s*\d+\s*[个条]",
    r"调用了\s*\w+\s*接口",
    r"通过\s*\w+\s*获取",
    r"已生成\s*\w+\s*文件",
    r"已创建\s*\w+\s*结果",
]

# 文件扩展名模式
FILE_EXTENSIONS = {
    ".html", ".json", ".geojson", ".shp", ".zip", ".png", 
    ".jpg", ".jpeg", ".pdf", ".csv", ".gpkg", ".tiff", ".tif"
}

# 允许报告的文件路径前缀（经过验证的路径）
VERIFIED_FILE_PREFIXES: Set[str] = set()

# OSM API 声称模式（用于检测 LLM 声称调用了 OpenStreetMap）
OSM_API_CLAIM_PATTERNS = [
    r"调用\s*OSM",
    r"调用\s*OpenStreetMap",
    r"通过\s*OSM",
    r"通过\s*OpenStreetMap",
    r"Overpass\s*API",
    r"overpass[_-]?api",
    r"使用\s*OSM\s*(获取|搜索|查询)",
    r"OSM\s*接口",
    r"openstreetmap\s*接口",
    r"查询\s*OSM",
    r"获取\s*OSM\s*数据",
]

# OSM POI 类型关键词
OSM_POI_KEYWORDS = [
    "星巴克", "Starbucks", "starbucks",
    "地铁站", "subway", "metro", "地铁",
    "医院", "hospital",
    "学校", "school",
    "银行", "bank",
    "超市", "supermarket",
    "药店", "pharmacy",
    "公园", "park",
    "餐厅", "restaurant",
    "酒店", "hotel",
]


@dataclass
class ValidationIssue:
    """验证问题"""
    severity: str  # "error" | "warning" | "info"
    issue_type: str
    description: str
    suggested_fix: str = ""
    evidence: str = ""  # 证据


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verified_files: List[str] = field(default_factory=list)
    blocked_files: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def add_issue(
        self,
        severity: str,
        issue_type: str,
        description: str,
        suggested_fix: str = "",
        evidence: str = "",
    ):
        self.issues.append(ValidationIssue(
            severity=severity,
            issue_type=issue_type,
            description=description,
            suggested_fix=suggested_fix,
            evidence=evidence,
        ))
        if severity == "error":
            self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "timestamp": self.timestamp,
            "issues": [
                {
                    "severity": i.severity,
                    "type": i.issue_type,
                    "description": i.description,
                    "suggested_fix": i.suggested_fix,
                    "evidence": i.evidence,
                }
                for i in self.issues
            ],
            "warnings": self.warnings,
            "verified_files": self.verified_files,
            "blocked_files": self.blocked_files,
        }


class ToolCallValidator:
    """
    工具调用验证器 - 防幻觉终极版

    【验证原则】
    1. 先验证后报告：所有声称的文件必须先通过文件系统验证
    2. 只信任系统返回：文件路径必须来自 ExecutorResult 或工具返回
    3. 零容忍：任何捏造行为立即报错
    4. 透明记录：所有验证过程都有日志可查
    """

    def __init__(self, strict_mode: bool = True):
        """
        初始化验证器

        Args:
            strict_mode: 严格模式，检测到问题直接报错
        """
        self._strict_mode = strict_mode
        self._compile_patterns()
        self._verified_output_files: Set[str] = set()  # 经过验证的输出文件
        self._validation_log: List[Dict[str, Any]] = []

    def _compile_patterns(self):
        """编译正则表达式"""
        self._fabrication_patterns = []
        for pattern, pattern_type in FABRICATION_PATTERNS:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._fabrication_patterns.append((compiled, pattern_type))
            except re.error:
                pass

        self._tool_claim_patterns = []
        for pattern in TOOL_CALL_CLAIM_PATTERNS:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._tool_claim_patterns.append(compiled)
            except re.error:
                pass

    def set_verified_files(self, files: List[str]):
        """
        设置经过验证的输出文件列表

        这些文件是系统执行器实际生成的文件，可以安全地报告给用户。

        Args:
            files: 经过验证的文件路径列表
        """
        self._verified_output_files = set()
        for f in files:
            if f and os.path.exists(f):
                self._verified_output_files.add(os.path.abspath(f))
                self._log_validation("file_verified", {"file": f, "status": "exists"})
            else:
                self._log_validation("file_verified", {"file": f, "status": "not_found"})

    def add_verified_file(self, file_path: str):
        """添加单个经过验证的文件"""
        if file_path and os.path.exists(file_path):
            abs_path = os.path.abspath(file_path)
            self._verified_output_files.add(abs_path)
            self._log_validation("file_verified", {"file": abs_path, "status": "exists"})

    def _log_validation(self, event_type: str, data: Dict[str, Any]):
        """记录验证日志"""
        self._validation_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
        })

    def validate(
        self,
        llm_response: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        output_files: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        验证 LLM 回复

        Args:
            llm_response: LLM 的回复文本
            tool_calls: 实际调用的工具列表
            output_files: 声称生成的输出文件列表（经过验证）
            context: 额外上下文信息

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # 如果提供了 output_files，将其标记为已验证
        if output_files:
            for f in output_files:
                if os.path.exists(f):
                    abs_f = os.path.abspath(f)
                    result.verified_files.append(abs_f)
                    self._verified_output_files.add(abs_f)

        # 1. 【最严格】检查是否捏造了文件
        self._check_file_fabrication(llm_response, result)

        # 2. 检查是否声称调用了工具但实际没有
        self._check_tool_call_claims(llm_response, tool_calls, result)

        # 3. 【新增】检查 OSM/OpenStreetMap API 调用声称
        self._check_osm_api_claims(llm_response, tool_calls, result)

        # 4. 检查是否捏造了数据
        self._check_data_fabrication(llm_response, tool_calls, result)

        # 5. 【核心】检查输出文件是否存在
        self._check_output_files(llm_response, result)

        # 6. 检查数据一致性
        self._check_data_consistency(llm_response, tool_calls, result)

        # 7. 路径规范化验证
        self._check_path_references(llm_response, result)

        return result

    def _check_file_fabrication(
        self,
        response: str,
        result: ValidationResult,
    ):
        """
        【核心检查】检查是否捏造了文件

        这是最严格的检查，任何未经确认的文件声称都会被拦截。
        """
        # 文件声称模式
        file_claim_patterns = [
            # "已生成xxx.html"、"已保存xxx.json"等
            r"(?:已生成|已创建|已保存|已完成|已输出|已写入)(?:到|至|于)?\s*([^\s]+\.(?:html|json|geojson|shp|zip|png|jpg))",
            # "文件路径是: xxx"或"路径: xxx"
            r"(?:文件路径|路径|文件)[是为]?\s*[:：]?\s*([^\s]+\.(?:html|json|geojson|shp|zip|png|jpg))",
            # "请查看xxx"（包含文件扩展名）
            r"请查看\s*([^\s]+\.(?:html|json|geojson|shp|zip|png|jpg))",
            # "保存在xxx"
            r"保存(?:在|于)\s*([^\s]+\.(?:html|json|geojson|shp|zip|png|jpg))",
            # 直接引用的文件路径
            r"(workspace[/\\][^\s]+\.(?:html|json|geojson|shp|zip|png|jpg))",
            r"([a-zA-Z]:\\[^\s]+\.(?:html|json|geojson|shp|zip|png|jpg))",
            r"(/[^\s]+\.(?:html|json|geojson|shp|zip|png|jpg))",
        ]

        mentioned_files = set()
        for pattern in file_claim_patterns:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                file_path = match.group(1) if match.lastindex else match.group(0)
                mentioned_files.add(file_path.strip())

        # 检查每个声称的文件
        for file_path in mentioned_files:
            # 标准化路径
            normalized = self._normalize_path(file_path)
            
            # 检查文件是否存在
            exists = os.path.exists(normalized)
            
            # 检查是否在已验证文件列表中
            abs_normalized = os.path.abspath(normalized)
            is_verified = abs_normalized in self._verified_output_files or normalized in self._verified_output_files

            self._log_validation("file_check", {
                "claimed_file": file_path,
                "normalized": normalized,
                "exists": exists,
                "is_verified": is_verified,
            })

            if not exists:
                # 文件不存在，这是一个严重错误
                result.add_issue(
                    severity="error",
                    issue_type="fabricated_file",
                    description=f"声称创建了文件但该文件不存在: {file_path}",
                    suggested_fix="你只能报告系统 ExecutorResult 中实际返回的文件路径。不要捏造任何文件路径！",
                    evidence=f"文件系统检查结果: {normalized} 不存在",
                )
                result.blocked_files.append(file_path)
            elif not is_verified:
                # 文件存在但未经过验证，发出警告
                result.add_warning(f"文件存在但未经系统验证: {file_path}")
                # 在严格模式下，这也算错误
                if self._strict_mode:
                    result.add_issue(
                        severity="warning",
                        issue_type="unverified_file",
                        description=f"文件存在但未经过 ExecutorResult 确认: {file_path}",
                        suggested_fix="建议只报告 ExecutorResult 返回的 output_file 字段",
                    )
            else:
                # 文件存在且已验证
                result.verified_files.append(abs_normalized)

    def _normalize_path(self, path: str) -> str:
        """标准化文件路径"""
        # 移除引号
        path = path.strip('"\'')
        # 转换相对路径为绝对路径
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    def _check_tool_call_claims(
        self,
        response: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        result: ValidationResult,
    ):
        """检查是否声称调用了工具"""
        # 查找声称调用工具的文本
        claims = []
        for pattern in self._tool_claim_patterns:
            matches = pattern.findall(response)
            claims.extend(matches)

        if not claims:
            return

        # 如果声称调用了工具但实际没有
        if tool_calls is None or len(tool_calls) == 0:
            for claim in claims:
                result.add_issue(
                    severity="error",
                    issue_type="fabricated_tool_call",
                    description=f"声称调用了工具但实际没有调用：'{claim}'",
                    suggested_fix="必须先调用工具才能报告结果",
                )

    def _check_osm_api_claims(
        self,
        response: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        result: ValidationResult,
    ):
        """
        检查是否声称调用了 OSM/OpenStreetMap API

        这是专门针对 OSM POI 搜索的验证，防止 LLM 捏造 OSM 数据。
        """
        import re as re_module

        # 检测 OSM API 声称
        osm_claims = []
        for pattern in OSM_API_CLAIM_PATTERNS:
            matches = re_module.findall(pattern, response, re_module.IGNORECASE)
            osm_claims.extend(matches)

        if not osm_claims:
            return

        # 检查是否有对应的工具调用
        has_osm_call = False
        if tool_calls:
            for call in tool_calls:
                tool_name = str(call.get("name", "")).lower()
                tool_params = str(call.get("parameters", {})).lower()

                # 检查是否是 OSM 相关工具
                if any(keyword in tool_name or keyword in tool_params for keyword in [
                    "osm", "overpass", "openstreetmap", "poi_search"
                ]):
                    has_osm_call = True
                    break

        if not has_osm_call:
            result.add_issue(
                severity="error",
                issue_type="fabricated_osm_call",
                description=f"声称调用了 OSM/OpenStreetMap 接口但实际没有调用：'{', '.join(osm_claims)}'",
                suggested_fix="必须调用 Overpass API 或 osm_plugin 才能获取 OSM 数据。正确调用方式：\n"
                           f"  1. 使用 OverpassExecutor（推荐）：task={{'task_type': 'poi_search', 'poi_types': ['starbucks', 'subway'], 'center_point': '广州体育中心', 'radius': 3000}}\n"
                           f"  2. 使用 osm_plugin：action='overpass_poi', location='广州体育中心', poi_types=['starbucks', 'subway']",
            )

    def _check_data_fabrication(
        self,
        response: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        result: ValidationResult,
    ):
        """检查是否捏造数据"""
        # 检查模糊数量（"约XX个"）
        vague_count_pattern = re.compile(r"[约大]约?\s*(\d+)\s*[个条座]")
        for match in vague_count_pattern.finditer(response):
            count_str = match.group(1)
            result.add_issue(
                severity="warning",
                issue_type="vague_count",
                description=f"使用模糊数量 '{count_str}'，应使用精确数量",
                suggested_fix="从工具返回的精确数量",
            )

        # 检查未经确认的坐标
        coord_pattern = re.compile(r"经?纬?度?[为是]?\s*(\d+\.\d+)[,，]\s*(\d+\.\d+)")
        if tool_calls is None or len(tool_calls) == 0:
            for match in coord_pattern.finditer(response):
                lat, lon = match.groups()
                result.add_issue(
                    severity="warning",
                    issue_type="unverified_coordinates",
                    description=f"回复中包含坐标 ({lat}, {lon})，但没有对应的工具调用验证",
                    suggested_fix="如果坐标来自工具调用，请在回复中说明工具名称",
                )

    def _check_output_files(
        self,
        response: str,
        result: ValidationResult,
    ):
        """检查输出文件是否存在"""
        # 从回复中提取文件路径
        file_path_patterns = [
            r"[:：]\s*(/[a-zA-Z0-9_/.-]+\.[a-zA-Z]+)",
            r"[:：]\s*([a-zA-Z]:\\[a-zA-Z0-9_/.-]+\.[a-zA-Z]+)",
            r"保存[至到]\s*[：:]\s*([^\s]+\.(?:html|json|geojson|shp|zip|png|jpg))",
        ]

        mentioned_files = set()
        for pattern in file_path_patterns:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                mentioned_files.add(match.group(1))

        if not mentioned_files:
            return

        # 检查文件是否存在
        for mentioned in mentioned_files:
            normalized = self._normalize_path(mentioned)
            abs_normalized = os.path.abspath(normalized)
            
            if not os.path.exists(normalized):
                result.add_issue(
                    severity="error",
                    issue_type="missing_output_file",
                    description=f"声称生成了文件但文件不存在：{mentioned}",
                    suggested_fix="确保文件已成功生成并使用正确的路径",
                )
                result.blocked_files.append(mentioned)
            elif abs_normalized not in self._verified_output_files:
                # 文件存在但不在已验证列表
                result.add_warning(f"文件存在但未经系统确认: {mentioned}")
                result.verified_files.append(abs_normalized)
            else:
                result.verified_files.append(abs_normalized)

    def _check_data_consistency(
        self,
        response: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        result: ValidationResult,
    ):
        """检查数据一致性"""
        if not tool_calls:
            return

        # 从工具返回中提取数量
        tool_counts = self._extract_counts_from_tools(tool_calls)

        # 检查回复中的数量是否一致
        response_counts = self._extract_counts_from_response(response)

        for count_type, count_value in response_counts.items():
            if count_type in tool_counts:
                tool_count = tool_counts[count_type]
                if count_value != tool_count:
                    result.add_issue(
                        severity="error",
                        issue_type="count_mismatch",
                        description=f"回复中数量 ({count_value}) 与工具返回 ({tool_count}) 不一致",
                        suggested_fix="必须使用工具返回的确切数量",
                    )

    def _extract_counts_from_tools(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """从工具调用中提取数量"""
        counts = {}

        for call in tool_calls:
            tool_name = call.get("name", "")
            result = call.get("result", "")

            # POI 搜索结果数量
            if "poi" in tool_name.lower():
                match = re.search(r"共.*?(\d+)\s*[个条]", str(result))
                if match:
                    counts["poi"] = int(match.group(1))

            # 通用数量提取
            all_matches = re.findall(r"(\d+)\s*[个条座]", str(result))
            if all_matches:
                counts[f"general_{tool_name}"] = int(all_matches[0])

        return counts

    def _extract_counts_from_response(self, response: str) -> Dict[str, int]:
        """从回复中提取数量"""
        counts = {}

        # 星巴克/POI 数量
        match = re.search(r"(星巴克|POI|点位).*?(\d+)\s*[个条]", response, re.IGNORECASE)
        if match:
            counts["poi"] = int(match.group(2))

        return counts

    def _check_path_references(
        self,
        response: str,
        result: ValidationResult,
    ):
        """检查路径引用是否有效"""
        # Windows 和 Unix 路径模式
        path_patterns = [
            r"([a-zA-Z]:\\[^\s]+)",  # Windows 绝对路径
            r"(/[^\s/]+\.[a-zA-Z]+)",  # Unix 路径
            r"(workspace[/\\][^\s]+\.[a-zA-Z]+)",  # workspace 相对路径
        ]

        for pattern in path_patterns:
            for match in re.finditer(pattern, response):
                path = match.group(1)
                # 检查文件扩展名是否是我们关心的
                ext = os.path.splitext(path)[1].lower()
                if ext in FILE_EXTENSIONS:
                    normalized = self._normalize_path(path)
                    if not os.path.exists(normalized):
                        result.add_issue(
                            severity="error",
                            issue_type="invalid_path_reference",
                            description=f"引用了不存在的文件路径: {path}",
                            suggested_fix="只引用系统 ExecutorResult 返回的已验证文件",
                        )

    def generate_enforcement_message(self, result: ValidationResult) -> str:
        """生成强制执行消息"""
        if result.is_valid:
            return ""

        messages = ["[严重警告] 检测到以下幻觉行为：\n"]

        for issue in result.issues:
            if issue.severity == "error":
                messages.append(f"[X] {issue.description}")
                if issue.suggested_fix:
                    messages.append(f"   -> 修复：{issue.suggested_fix}")
                if issue.evidence:
                    messages.append(f"   -> 证据：{issue.evidence}")

        messages.append("\n[警告] 你的回复包含捏造信息，已被系统拦截！")
        messages.append("请重新生成回复，只报告 ExecutorResult 中确认存在的文件。")

        return "\n".join(messages)

    def get_validation_report(self) -> str:
        """获取验证报告"""
        lines = ["=" * 60]
        lines.append("工具调用验证报告")
        lines.append("=" * 60)
        lines.append(f"时间: {self.timestamp}")
        lines.append(f"验证日志条目数: {len(self._validation_log)}")
        lines.append(f"已验证文件数: {len(self._verified_output_files)}")
        
        if self._validation_log:
            lines.append("\n验证日志:")
            for log in self._validation_log[-10:]:  # 最近10条
                lines.append(f"  [{log['timestamp']}] {log['event_type']}: {log['data']}")
        
        return "\n".join(lines)


# =============================================================================
# 便捷函数
# =============================================================================

_default_validator: Optional[ToolCallValidator] = None


def get_validator(strict_mode: bool = True) -> ToolCallValidator:
    """获取验证器单例"""
    global _default_validator
    if _default_validator is None:
        _default_validator = ToolCallValidator(strict_mode=strict_mode)
    return _default_validator


def validate_tool_calls(
    response: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    output_files: Optional[List[str]] = None,
) -> ValidationResult:
    """
    便捷函数：验证工具调用

    使用方式：
        result = validate_tool_calls(
            response="已获取数据，共45个点位",
            tool_calls=[{"name": "amap", "result": "..."}],
            output_files=["/path/to/file.html"],
        )
        if not result.is_valid:
            print(result.generate_enforcement_message())
    """
    validator = get_validator()
    if output_files:
        validator.set_verified_files(output_files)
    return validator.validate(response, tool_calls, output_files)


def validate_and_sanitize_response(
    response: str,
    verified_output_files: Optional[List[str]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, ValidationResult]:
    """
    验证并清理回复

    如果发现幻觉内容，返回：
    - 清理后的回复（移除了捏造的文件引用）
    - 验证结果

    Args:
        response: LLM 原始回复
        verified_output_files: 经过验证的文件列表
        tool_calls: 工具调用记录

    Returns:
        (清理后的回复, 验证结果)
    """
    validator = ToolCallValidator(strict_mode=True)
    if verified_output_files:
        validator.set_verified_files(verified_output_files)

    result = validator.validate(response, tool_calls, verified_output_files)

    # 如果没有问题，直接返回原回复
    if result.is_valid:
        return response, result

    # 清理回复中的捏造内容
    sanitized = response
    
    # 移除未经确认的文件引用
    for blocked_file in result.blocked_files:
        # 移除对不存在文件的引用行
        blocked_patterns = [
            rf"(?:已生成|已创建|已保存|已完成).{{0,30}}{re.escape(blocked_file)}",
            rf"(?:文件路径|路径|文件)[是为]?\s*[:：]?\s*{re.escape(blocked_file)}",
        ]
        for pattern in blocked_patterns:
            sanitized = re.sub(pattern, "[文件引用已移除]", sanitized, flags=re.IGNORECASE)

    return sanitized, result


__all__ = [
    "ToolCallValidator",
    "ValidationIssue",
    "ValidationResult",
    "get_validator",
    "validate_tool_calls",
    "validate_and_sanitize_response",
]
