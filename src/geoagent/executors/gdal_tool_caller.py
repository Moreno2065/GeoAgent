"""
GDAL 工具调用器 - LLM 任务 JSON → GDAL 执行
============================================
严格按照架构指导设计：

1. 接收 LLM 输出的任务 JSON
2. Schema 校验（参数白名单）
3. 调用 GDALEngine 执行
4. 返回标准化结果

设计原则：
- LLM 只能通过白名单中的工具操作
- 参数必须通过 Schema 校验
- 统一的错误处理和日志
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from geoagent.executors.gdal_engine import (
    GDAL_TOOL_WHITELIST,
    GDAL_TOOL_DEFINITIONS,
    GDALEngine,
    GDALResult,
    get_gdal_engine,
)


# =============================================================================
# 工具调用结果
# =============================================================================

@dataclass
class ToolCallResult:
    """LLM 工具调用结果"""
    success: bool
    tool_name: str
    arguments: Dict[str, Any]
    output_path: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "output_path": self.output_path,
            "data": self.data,
            "error": self.error,
            "validation_errors": self.validation_errors,
            "warnings": self.warnings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =============================================================================
# 参数 Schema 校验器
# =============================================================================

class SchemaValidator:
    """
    JSON Schema 参数校验器

    确保 LLM 输出的参数符合工具定义。
    """

    # 预定义工具的参数 Schema
    TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
        tool["name"]: tool["parameters"]
        for tool in GDAL_TOOL_DEFINITIONS
    }

    def validate(self, tool_name: str, arguments: Dict[str, Any]) -> List[str]:
        """
        校验参数是否符合 Schema

        Args:
            tool_name: 工具名称
            arguments: 参数字典

        Returns:
            错误列表（空表示校验通过）
        """
        errors = []

        # 检查工具是否存在
        if tool_name not in GDAL_TOOL_WHITELIST:
            errors.append(f"未知工具: {tool_name}，允许的工具: {sorted(GDAL_TOOL_WHITELIST)}")
            return errors

        # 获取 Schema
        schema = self.TOOL_SCHEMAS.get(tool_name)
        if not schema:
            errors.append(f"工具 {tool_name} 没有定义 Schema")
            return errors

        # 检查必需参数
        required = schema.get("required", [])
        for param in required:
            if param not in arguments or arguments[param] is None:
                errors.append(f"缺少必需参数: {param}")

        # 检查参数类型
        properties = schema.get("properties", {})
        for param_name, param_value in arguments.items():
            if param_name not in properties:
                errors.append(f"未知参数: {param_name}，允许的参数: {list(properties.keys())}")
                continue

            param_schema = properties[param_name]
            param_type = param_schema.get("type")

            # 类型校验
            if param_type == "string" and not isinstance(param_value, str):
                errors.append(f"参数 {param_name} 必须是字符串，实际: {type(param_value).__name__}")
            elif param_type == "number" and not isinstance(param_value, (int, float)):
                errors.append(f"参数 {param_name} 必须是数字，实际: {type(param_value).__name__}")
            elif param_type == "boolean" and not isinstance(param_value, bool):
                errors.append(f"参数 {param_name} 必须是布尔值，实际: {type(param_value).__name__}")
            elif param_type == "array" and not isinstance(param_value, list):
                errors.append(f"参数 {param_name} 必须是数组，实际: {type(param_value).__name__}")
            elif param_type == "object" and not isinstance(param_value, dict):
                errors.append(f"参数 {param_name} 必须是对象，实际: {type(param_value).__name__}")

            # 枚举值校验
            if "enum" in param_schema:
                if param_value not in param_schema["enum"]:
                    errors.append(
                        f"参数 {param_name} 的值 '{param_value}' 不在允许范围内，"
                        f"允许值: {param_schema['enum']}"
                    )

        return errors

    def get_required_params(self, tool_name: str) -> List[str]:
        """获取工具的必需参数"""
        schema = self.TOOL_SCHEMAS.get(tool_name, {})
        return schema.get("required", [])

    def get_optional_params(self, tool_name: str) -> Dict[str, Any]:
        """获取工具的可选参数及其默认值"""
        schema = self.TOOL_SCHEMAS.get(tool_name, {})
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        return {
            name: props.get("default")
            for name, props in properties.items()
            if name not in required
        }


# =============================================================================
# GDAL 工具调用器
# =============================================================================

class GDALToolCaller:
    """
    GDAL 工具调用器

    核心职责：
    1. 接收 LLM 输出的任务 JSON（可以是函数调用格式或直接工具调用格式）
    2. Schema 校验参数
    3. 调用 GDALEngine 执行
    4. 返回标准化结果

    支持的输入格式：

    格式 1 - 直接工具调用：
    {
        "tool": "raster_clip",
        "arguments": {
            "input_path": "data/dem.tif",
            "mask_path": "data/area.geojson",
            "output_path": "output/clipped.tif"
        }
    }

    格式 2 - 兼容旧接口（task 字段）：
    {
        "task": "raster_clip",
        "input_path": "data/dem.tif",
        "mask_path": "data/area.geojson",
        "output_path": "output/clipped.tif"
    }

    格式 3 - 符合 LangChain/MCP 等格式：
    {
        "name": "raster_clip",
        "arguments": {
            "input_path": "data/dem.tif",
            "mask_path": "data/area.geojson",
            "output_path": "output/clipped.tif"
        }
    }

    使用方式：
        caller = GDALToolCaller()
        result = caller.call({
            "tool": "raster_clip",
            "arguments": {
                "input_path": "data/dem.tif",
                "mask_path": "data/area.geojson",
                "output_path": "output/clipped.tif"
            }
        })
    """

    def __init__(self, engine: Optional[GDALEngine] = None):
        """
        初始化 GDAL 工具调用器

        Args:
            engine: GDAL 引擎实例（可选，默认使用全局单例）
        """
        self.engine = engine or get_gdal_engine()
        self.validator = SchemaValidator()

    def call(self, task: Dict[str, Any]) -> ToolCallResult:
        """
        执行工具调用

        Args:
            task: 任务字典（支持多种格式）

        Returns:
            ToolCallResult 执行结果
        """
        # 解析任务格式
        tool_name, arguments = self._parse_task(task)

        if not tool_name:
            return ToolCallResult(
                success=False,
                tool_name="",
                arguments=task,
                error="无法从任务中识别工具名称",
                validation_errors=["任务必须包含 'tool' 或 'task' 字段"],
            )

        # 白名单校验
        if tool_name not in GDAL_TOOL_WHITELIST:
            return ToolCallResult(
                success=False,
                tool_name=tool_name,
                arguments=arguments,
                error=f"未知工具: {tool_name}",
                validation_errors=[f"工具 {tool_name} 不在白名单中，允许的工具: {sorted(GDAL_TOOL_WHITELIST)}"],
            )

        # Schema 校验
        validation_errors = self.validator.validate(tool_name, arguments)
        if validation_errors:
            return ToolCallResult(
                success=False,
                tool_name=tool_name,
                arguments=arguments,
                error="参数校验失败",
                validation_errors=validation_errors,
            )

        # 执行工具
        try:
            gdal_result = self.engine.execute(tool_name, arguments)
            return self._convert_result(gdal_result, arguments)
        except Exception as e:
            return ToolCallResult(
                success=False,
                tool_name=tool_name,
                arguments=arguments,
                error=f"执行异常: {str(e)}",
                validation_errors=[traceback.format_exc()],
            )

    def _parse_task(self, task: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
        """
        解析任务字典，提取工具名称和参数

        Args:
            task: 任务字典

        Returns:
            (tool_name, arguments) 元组
        """
        # 格式 1: {"tool": "xxx", "arguments": {...}}
        if "tool" in task:
            tool_name = task["tool"]
            arguments = task.get("arguments", {})
            # arguments 可能是空字典，这时取其他字段作为参数
            if not arguments:
                arguments = {k: v for k, v in task.items() if k != "tool"}
            return tool_name, arguments

        # 格式 3: {"name": "xxx", "arguments": {...}}
        if "name" in task:
            tool_name = task["name"]
            arguments = task.get("arguments", {})
            if not arguments:
                arguments = {k: v for k, v in task.items() if k != "name"}
            return tool_name, arguments

        # 格式 2: {"task": "xxx", ...其他参数}
        if "task" in task:
            tool_name = task["task"]
            arguments = {k: v for k, v in task.items() if k != "task"}
            return tool_name, arguments

        # 无法识别
        return None, {}

    def _convert_result(self, gdal_result: GDALResult, arguments: Dict[str, Any]) -> ToolCallResult:
        """将 GDALResult 转换为 ToolCallResult"""
        return ToolCallResult(
            success=gdal_result.success,
            tool_name=gdal_result.tool_name,
            arguments=arguments,
            output_path=gdal_result.output_path,
            data=gdal_result.data,
            error=gdal_result.error,
            validation_errors=[],
            warnings=gdal_result.warnings,
        )

    def call_batch(self, tasks: List[Dict[str, Any]]) -> List[ToolCallResult]:
        """
        批量执行工具调用

        Args:
            tasks: 任务列表

        Returns:
            结果列表
        """
        return [self.call(task) for task in tasks]

    def validate_only(self, task: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        仅校验参数，不执行

        Args:
            task: 任务字典

        Returns:
            (is_valid, errors) 元组
        """
        tool_name, arguments = self._parse_task(task)

        if not tool_name:
            return False, ["无法识别工具名称"]

        if tool_name not in GDAL_TOOL_WHITELIST:
            return False, [f"工具 {tool_name} 不在白名单中"]

        errors = self.validator.validate(tool_name, arguments)
        return len(errors) == 0, errors

    # ── 工具元信息 ──────────────────────────────────────────────────────────────

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取所有工具定义（用于 LLM 的 tool calling）"""
        return GDAL_TOOL_DEFINITIONS

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return sorted(GDAL_TOOL_WHITELIST)

    def describe_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具描述"""
        for tool in GDAL_TOOL_DEFINITIONS:
            if tool["name"] == tool_name:
                return tool
        return None


# =============================================================================
# 全局单例
# =============================================================================

_tool_caller: Optional[GDALToolCaller] = None


def get_tool_caller() -> GDALToolCaller:
    """获取 GDAL 工具调用器单例"""
    global _tool_caller
    if _tool_caller is None:
        _tool_caller = GDALToolCaller()
    return _tool_caller


def call_gdal_tool(task: Dict[str, Any]) -> ToolCallResult:
    """
    便捷函数：执行 GDAL 工具调用

    Args:
        task: 任务字典

    Returns:
        ToolCallResult 执行结果
    """
    caller = get_tool_caller()
    return caller.call(task)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "ToolCallResult",
    "SchemaValidator",
    "GDALToolCaller",
    "get_tool_caller",
    "call_gdal_tool",
]
