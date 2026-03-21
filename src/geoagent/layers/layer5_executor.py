"""
第5层：执行引擎层（Task Router + Executors）
=============================================
核心职责：
1. 确定性执行：读 DSL → 查路由表 → 调用固定函数 → 生成结果
2. 不让 LLM 直接碰执行逻辑
3. 所有库（ArcPy, GeoPandas, NetworkX, Amap, PostGIS）都是被调用者

设计原则：
- 后端代码路由，不依赖 LLM
- TaskRouter 是唯一的调度入口
- 统一 ExecutorResult 格式
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
from enum import Enum

from geoagent.layers.architecture import Scenario, Engine


# =============================================================================
# 执行结果标准化
# =============================================================================

@dataclass
class ExecutorResult:
    """
    Executor 返回的标准化结果

    所有 GIS Executor 必须返回此格式的结果，
    确保前端可以统一展示。
    """
    success: bool
    scenario: str
    task: str
    engine: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None
    warnings: list = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "scenario": self.scenario,
            "task": self.task,
            "engine": self.engine,
            "data": self.data,
            "error": self.error,
            "error_detail": self.error_detail,
            "warnings": self.warnings,
            "meta": self.meta,
        }

    @classmethod
    def ok(
        cls,
        scenario: str,
        task: str,
        engine: str,
        data: Dict[str, Any],
        **kwargs
    ) -> "ExecutorResult":
        return cls(
            success=True,
            scenario=scenario,
            task=task,
            engine=engine,
            data=data,
            **kwargs
        )

    @classmethod
    def err(
        cls,
        scenario: str,
        task: str,
        error: str,
        engine: str = "unknown",
        error_detail: Optional[str] = None,
        **kwargs
    ) -> "ExecutorResult":
        return cls(
            success=False,
            scenario=scenario,
            task=task,
            engine=engine,
            error=error,
            error_detail=error_detail or traceback.format_exc(),
            **kwargs
        )


# =============================================================================
# Scenario → Executor 映射表
# =============================================================================

SCENARIO_EXECUTOR_MAP: Dict[Scenario, str] = {
    Scenario.ROUTE: "route",
    Scenario.BUFFER: "buffer",
    Scenario.OVERLAY: "overlay",
    Scenario.INTERPOLATION: "idw",
    Scenario.VIEWSHED: "shadow",
    Scenario.STATISTICS: "hotspot",
    Scenario.RASTER: "ndvi",
}


# =============================================================================
# TaskRouter
# =============================================================================

class TaskRouter:
    """
    统一任务路由器

    核心职责：
    1. 根据 scenario 确定使用哪个 Executor
    2. 调用 Executor.run(task) 执行任务
    3. 统一返回 ExecutorResult 格式
    4. 处理降级和错误

    设计原则：
    - 后端代码路由，不依赖 LLM
    - 所有库都是"被调用者"
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def _get_executor(self, executor_key: str):
        """获取 Executor 实例（延迟加载）"""
        if executor_key in self._cache:
            return self._cache[executor_key]

        # 延迟导入
        executor_map = {
            "route": ("geoagent.executors.route_executor", "RouteExecutor"),
            "buffer": ("geoagent.executors.buffer_executor", "BufferExecutor"),
            "overlay": ("geoagent.executors.overlay_executor", "OverlayExecutor"),
            "idw": ("geoagent.executors.idw_executor", "IDWExecutor"),
            "shadow": ("geoagent.executors.shadow_executor", "ShadowExecutor"),
            "hotspot": ("geoagent.executors.hotspot_executor", "HotspotExecutor"),
            "ndvi": ("geoagent.executors.ndvi_executor", "NdviExecutor"),
        }

        if executor_key not in executor_map:
            return None

        module_name, class_name = executor_map[executor_key]
        try:
            module = __import__(module_name, fromlist=[class_name])
            executor_cls = getattr(module, class_name)
            executor = executor_cls()
            self._cache[executor_key] = executor
            return executor
        except (ImportError, AttributeError):
            return None

    def route(self, scenario: Scenario, task_dict: Dict[str, Any]) -> ExecutorResult:
        """
        执行任务路由

        Args:
            scenario: 场景类型
            task_dict: 任务参数字典

        Returns:
            ExecutorResult 统一结果格式
        """
        _scenario = scenario.value if hasattr(scenario, 'value') else scenario
        executor_key = SCENARIO_EXECUTOR_MAP.get(scenario, "general")
        executor = self._get_executor(executor_key)

        if executor is None:
            return ExecutorResult.err(
                scenario=_scenario,
                task=task_dict.get("task", _scenario),
                error=f"无法加载 Executor: {executor_key}，场景 '{_scenario}' 暂不支持",
                engine="router"
            )
        try:
            result = executor.run(task_dict)
            return self._convert_result(result, _scenario, executor_key)
        except Exception as e:
            return ExecutorResult.err(
                scenario=_scenario,
                task=task_dict.get("task", _scenario),
                error=f"Executor {executor_key} 执行失败: {str(e)}",
                engine=executor_key,
                error_detail=traceback.format_exc(),
            )

    def _convert_result(self, result: Any, scenario: str, executor_key: str) -> ExecutorResult:
        """转换 Executor 结果为标准化格式"""
        if isinstance(result, ExecutorResult):
            return result

        if isinstance(result, dict):
            return ExecutorResult(
                success=result.get("success", True),
                scenario=scenario,
                task=result.get("task_type", scenario),
                engine=result.get("engine", executor_key),
                data=result.get("data"),
                error=result.get("error"),
                error_detail=result.get("error_detail"),
            )

        return ExecutorResult.ok(
            scenario=scenario,
            task=scenario,
            engine=executor_key,
            data={"raw_result": str(result)},
        )


# =============================================================================
# 全局 Router 实例
# =============================================================================

_router: Optional[TaskRouter] = None


def get_router() -> TaskRouter:
    """获取全局 Router 单例"""
    global _router
    if _router is None:
        _router = TaskRouter()
    return _router


# =============================================================================
# 核心执行函数
# =============================================================================

def execute_task(scenario: Scenario, task_dict: Dict[str, Any]) -> ExecutorResult:
    """
    统一任务执行入口

    确定性路由：scenario + task_dict → Executor.run()

    设计原则：
    - 后端代码路由，不依赖 LLM
    - 所有库（ArcPy/GeoPandas/Amap/NetworkX）都是"被调用者"
    - 统一返回 ExecutorResult

    Args:
        scenario: 场景类型
        task_dict: 任务参数字典

    Returns:
        ExecutorResult 统一结果
    """
    router = get_router()
    return router.route(scenario, task_dict)


# =============================================================================
# 便捷函数
# =============================================================================

def execute_route(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行路径规划任务"""
    return execute_task(Scenario.ROUTE, task_dict)


def execute_buffer(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行缓冲区分析任务"""
    return execute_task(Scenario.BUFFER, task_dict)


def execute_overlay(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行叠置分析任务"""
    return execute_task(Scenario.OVERLAY, task_dict)


def execute_interpolation(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行插值分析任务"""
    return execute_task(Scenario.INTERPOLATION, task_dict)


def execute_viewshed(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行视域分析任务"""
    return execute_task(Scenario.VIEWSHED, task_dict)


def execute_statistics(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行统计分析任务"""
    return execute_task(Scenario.STATISTICS, task_dict)


def execute_raster(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行栅格分析任务"""
    return execute_task(Scenario.RASTER, task_dict)


__all__ = [
    "ExecutorResult",
    "SCENARIO_EXECUTOR_MAP",
    "TaskRouter",
    "get_router",
    "execute_task",
    "execute_route",
    "execute_buffer",
    "execute_overlay",
    "execute_interpolation",
    "execute_viewshed",
    "execute_statistics",
    "execute_raster",
]
