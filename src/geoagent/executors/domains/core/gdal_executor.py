"""
GDALExecutor - GDAL 工具执行器
================================
将 GDAL 工具调用集成到 Layer 5 Executor 架构。

核心职责：
1. 接收任务字典（兼容 Layer 5 接口）
2. Pydantic Schema 验证
3. 调用 GDALToolCaller 执行
4. 转换为 ExecutorResult 返回

设计原则：
- 与其他 Executor（BufferExecutor, OverlayExecutor）保持一致
- 支持 engine 降级和错误处理
- 统一 ExecutorResult 格式
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult
from geoagent.executors.gdal_engine import (
    GDAL_TOOL_WHITELIST,
    GDALEngine,
    GDALResult,
    get_gdal_engine,
)
from geoagent.executors.gdal_tool_caller import (
    GDALToolCaller,
    ToolCallResult,
    get_tool_caller,
)
from geoagent.executors.gdal_schema import (
    GDALSchemaValidator,
    validate_gdal_task,
)


class GDALExecutor(BaseExecutor):
    """
    GDAL 工具执行器

    路由策略：
    - 根据 task 字段识别工具类型
    - 支持的 task 值：
      * raster_clip, raster_reproject, raster_translate, raster_resample
      * vector_reproject, vector_buffer, vector_clip, vector_intersect

    引擎：GDAL/OGR/OSR

    示例：
        executor = GDALExecutor()
        result = executor.run({
            "task": "raster_clip",
            "input_path": "data/dem.tif",
            "mask_path": "data/area.geojson",
            "output_path": "output/clipped.tif"
        })
    """

    task_type = "gdal"
    supported_engines = {"gdal"}

    # task 名称白名单
    SUPPORTED_TASKS: Set[str] = GDAL_TOOL_WHITELIST

    def __init__(self):
        self._engine: Optional[GDALEngine] = None
        self._tool_caller: Optional[GDALToolCaller] = None
        self._schema_validator: Optional[GDALSchemaValidator] = None

    @property
    def engine(self) -> GDALEngine:
        """懒加载 GDAL 引擎"""
        if self._engine is None:
            self._engine = get_gdal_engine()
        return self._engine

    @property
    def tool_caller(self) -> GDALToolCaller:
        """懒加载工具调用器"""
        if self._tool_caller is None:
            self._tool_caller = GDALToolCaller(self.engine)
        return self._tool_caller

    @property
    def schema_validator(self) -> GDALSchemaValidator:
        """懒加载 Schema 验证器"""
        if self._schema_validator is None:
            self._schema_validator = GDALSchemaValidator()
        return self._schema_validator

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行 GDAL 工具

        Args:
            task: 任务参数字典，支持以下格式：
                - {"task": "raster_clip", "input_path": "...", ...}
                - {"tool": "raster_clip", "arguments": {...}}
                - {"name": "raster_clip", "arguments": {...}}

        Returns:
            ExecutorResult 统一结果格式
        """
        # 1. 提取 task 名称
        task_name = task.get("task") or task.get("tool") or task.get("name") or ""

        if not task_name:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="任务必须包含 'task' 或 'tool' 字段",
                engine="gdal",
            )

        # 2. 白名单校验
        if task_name not in self.SUPPORTED_TASKS:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"不支持的工具: {task_name}，支持的工具: {sorted(self.SUPPORTED_TASKS)}",
                engine="gdal",
            )

        # 3. Pydantic Schema 验证
        is_valid, validated_model, validation_errors = validate_gdal_task(task)
        if not is_valid:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"参数校验失败: {', '.join(validation_errors[:3])}",
                engine="gdal",
                error_detail="\n".join(validation_errors),
            )

        # 4. 转换为执行参数字典
        validated_dict = validated_model.to_dict()

        # 5. 执行工具
        tool_result = self.tool_caller.call(validated_dict)

        # 6. 转换结果
        return self._convert_result(tool_result, task_name)

    def _convert_result(self, result: ToolCallResult, task_name: str) -> ExecutorResult:
        """将 ToolCallResult 转换为 ExecutorResult"""
        if result.success:
            extra_data = result.data if result.data else {}
            return ExecutorResult.ok(
                task_type=self.task_type,
                engine="gdal",
                data={
                    "tool_name": result.tool_name,
                    "output_path": result.output_path,
                    **extra_data,
                },
                warnings=result.warnings,
                meta={
                    "arguments": result.arguments,
                    "task_name": task_name,
                },
            )
        else:
            error_msg = result.error or "未知错误"
            if result.validation_errors:
                error_msg += f"\n校验错误: {', '.join(result.validation_errors[:2])}"

            return ExecutorResult.err(
                task_type=self.task_type,
                error=error_msg,
                engine="gdal",
                error_detail="\n".join(result.validation_errors) if result.validation_errors else None,
            )

    def get_available_tools(self) -> list[str]:
        """获取可用的工具列表"""
        return self.tool_caller.get_tool_names()

    def get_tool_definition(self, tool_name: str) -> Optional[dict]:
        """获取工具定义"""
        return self.tool_caller.describe_tool(tool_name)


# =============================================================================
# 全局便捷函数
# =============================================================================

_gdal_executor: Optional[GDALExecutor] = None


def get_gdal_executor() -> GDALExecutor:
    """获取 GDAL Executor 单例"""
    global _gdal_executor
    if _gdal_executor is None:
        _gdal_executor = GDALExecutor()
    return _gdal_executor


def execute_gdal_task(task: Dict[str, Any]) -> ExecutorResult:
    """
    便捷函数：执行 GDAL 任务

    Args:
        task: 任务参数字典

    Returns:
        ExecutorResult
    """
    executor = get_gdal_executor()
    return executor.run(task)


__all__ = [
    "GDALExecutor",
    "get_gdal_executor",
    "execute_gdal_task",
]
