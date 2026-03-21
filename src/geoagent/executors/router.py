"""
Executor Router - 任务路由中枢
=================================
统一的任务路由层：scenario/task → 对应的 Executor。

核心设计原则（来自用户架构指导）：
1. 所有库（ArcPy, GeoPandas, NetworkX, Amap, PostGIS）都是"被调用者"，不是"决策者"
2. TaskRouter 是唯一的调度入口，不让库之间互相调用
3. 统一数据格式：输入 → GeoJSON/GeoDataFrame → 输出
4. 标准化 ExecutorResult，所有 Executor 返回统一格式

文件结构：
  router.py      ← 本文件：TaskRouter + execute_task（核心入口）
  scenario.py    ← scenario → Executor 映射表
  [各 Executor 文件] ← 具体执行器
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult


# =============================================================================
# Executor 实例缓存（延迟导入，避免循环依赖）
# =============================================================================

_executor_cache: Dict[str, BaseExecutor] = {}


def _get_executor(executor_key: str) -> Optional[BaseExecutor]:
    """延迟加载 Executor 实例（单例缓存）"""
    if executor_key in _executor_cache:
        return _executor_cache[executor_key]

    cls_map: Dict[str, type[BaseExecutor]] = {
        "route":         lambda: __import__(
                              "geoagent.executors.route_executor",
                              fromlist=["RouteExecutor"]).RouteExecutor,
        "buffer":        lambda: __import__(
                              "geoagent.executors.buffer_executor",
                              fromlist=["BufferExecutor"]).BufferExecutor,
        "overlay":       lambda: __import__(
                              "geoagent.executors.overlay_executor",
                              fromlist=["OverlayExecutor"]).OverlayExecutor,
        "interpolation":  lambda: __import__(
                              "geoagent.executors.idw_executor",
                              fromlist=["IDWExecutor"]).IDWExecutor,
        "ndvi":          lambda: __import__(
                              "geoagent.executors.ndvi_executor",
                              fromlist=["NdviExecutor"]).NdviExecutor,
        "hotspot":        lambda: __import__(
                              "geoagent.executors.hotspot_executor",
                              fromlist=["HotspotExecutor"]).HotspotExecutor,
        "shadow_analysis": lambda: __import__(
                              "geoagent.executors.shadow_executor",
                              fromlist=["ShadowExecutor"]).ShadowExecutor,
        "viewshed":      lambda: __import__(
                              "geoagent.executors.shadow_executor",
                              fromlist=["ShadowExecutor"]).ShadowExecutor,
        "visualization":  lambda: __import__(
                              "geoagent.executors.viz_executor",
                              fromlist=["VisualizationExecutor"]).VisualizationExecutor,
        "accessibility":   lambda: __import__(
                              "geoagent.executors.route_executor",
                              fromlist=["RouteExecutor"]).RouteExecutor,
        "suitability":   lambda: __import__(
                              "geoagent.executors.general_executor",
                              fromlist=["GeneralExecutor"]).GeneralExecutor,
        "postgis":       lambda: __import__(
                              "geoagent.executors.postgis_executor",
                              fromlist=["PostGISExecutor"]).PostGISExecutor,
        "general":        lambda: __import__(
                              "geoagent.executors.general_executor",
                              fromlist=["GeneralExecutor"]).GeneralExecutor,
    }

    loader = cls_map.get(executor_key)
    if loader is None:
        return None

    try:
        executor_cls = loader()
        instance = executor_cls()
        _executor_cache[executor_key] = instance
        return instance
    except Exception:
        return None


# =============================================================================
# scenario → executor_key 映射（来自 dsl/protocol.py SCENARIO_EXECUTOR_MAP）
# =============================================================================

SCENARIO_EXECUTOR_KEY: Dict[str, str] = {
    "route":           "route",
    "buffer":          "buffer",
    "overlay":         "overlay",
    "interpolation":    "interpolation",
    "shadow_analysis":  "shadow_analysis",
    "viewshed":        "shadow_analysis",    # viewshed 使用 shadow_executor
    "ndvi":            "ndvi",
    "hotspot":         "hotspot",
    "visualization":    "visualization",
    "accessibility":    "route",              # accessibility 路由到 route executor
    "suitability":     "general",             # suitability 暂用 general executor
    "general":         "general",
}


# =============================================================================
# task 类型 → scenario 映射（兼容旧 task 类型命名）
# =============================================================================

# task_schema.py 中的 task 字段值（如 "buffer", "overlay" 等）
# 与 SCENARIO_EXECUTOR_KEY 的 key 完全一致，所以直接用
_TASK_TO_SCENARIO: Dict[str, str] = {
    "route":            "route",
    "buffer":           "buffer",
    "overlay":          "overlay",
    "interpolation":     "interpolation",
    "shadow_analysis":   "shadow_analysis",
    "viewshed":         "viewshed",
    "ndvi":             "ndvi",
    "hotspot":          "hotspot",
    "visualization":     "visualization",
    "accessibility":     "accessibility",
    "suitability":      "suitability",
    "general":          "general",
    # 兼容 geo_engine/router.py 中的 task 别名
    "proximity":        "buffer",
    "surface":          "interpolation",
    "spatial join":     "overlay",
    "spatial_join":     "overlay",
}


def _task_to_scenario(task_type: str) -> str:
    """将 task 类型字符串映射为 scenario（再映射为 executor_key）"""
    t = task_type.lower().strip()
    return _TASK_TO_SCENARIO.get(t, "general")


def _scenario_to_executor_key(scenario: str) -> str:
    """将 scenario 映射为 executor key"""
    s = scenario.lower().strip()
    return SCENARIO_EXECUTOR_KEY.get(s, "general")


def _resolve_executor_key(task_type: str) -> str:
    """快捷方法：task_type → executor_key 一步到位"""
    scenario = _task_to_scenario(task_type)
    return _scenario_to_executor_key(scenario)


# =============================================================================
# TaskRouter
# =============================================================================

class TaskRouter:
    """
    统一任务路由器

    核心职责：
    1. 根据 task_type / scenario 确定使用哪个 Executor
    2. 调用 Executor.run(task) 执行任务
    3. 统一返回 ExecutorResult 格式
    4. 处理降级和错误

    使用方式：
        router = TaskRouter()
        result = router.route({
            "task": "buffer",
            "input_layer": "roads.shp",
            "distance": 500,
            "unit": "meters",
        })
    """

    def __init__(self):
        self._cache: Dict[str, BaseExecutor] = {}

    def _load_executor(self, executor_key: str) -> Optional[BaseExecutor]:
        """加载 Executor（带缓存）"""
        if executor_key in self._cache:
            return self._cache[executor_key]

        executor = _get_executor(executor_key)
        if executor:
            self._cache[executor_key] = executor
        return executor

    def route(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行任务路由

        Args:
            task: 任务参数字典，至少包含 "task" 字段

        Returns:
            ExecutorResult 统一结果格式
        """
        task_type = task.get("task", "general")
        executor_key = _resolve_executor_key(task_type)

        executor = self._load_executor(executor_key)
        if executor is None:
            return ExecutorResult.err(
                task_type,
                f"无法加载 Executor: {executor_key}，任务类型 '{task_type}' 暂不支持",
                engine="router"
            )

        try:
            return executor.run(task)
        except Exception as e:
            return ExecutorResult.err(
                task_type,
                f"Executor {executor_key} 执行失败: {str(e)}",
                engine=executor_key,
                error_detail=traceback.format_exc(),
            )

    def route_by_scenario(self, scenario: str, task: Dict[str, Any]) -> ExecutorResult:
        """通过 scenario 名称路由（不依赖 task 字段）"""
        executor_key = _scenario_to_executor_key(scenario)
        executor = self._load_executor(executor_key)
        if executor is None:
            return ExecutorResult.err(
                scenario,
                f"无法加载 Executor: {executor_key}，场景 '{scenario}' 暂不支持",
                engine="router"
            )
        try:
            return executor.run(task)
        except Exception as e:
            return ExecutorResult.err(
                scenario,
                f"Executor {executor_key} 执行失败: {str(e)}",
                engine=executor_key,
                error_detail=traceback.format_exc(),
            )

    def list_executors(self) -> list[str]:
        """列出所有可用的 Executor key"""
        return list(SCENARIO_EXECUTOR_KEY.values())


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
# 核心执行函数（便捷入口）
# =============================================================================

def execute_task(task: Dict[str, Any]) -> ExecutorResult:
    """
    统一任务执行入口（核心 API）

    确定性路由：task["task"] → Executor.run()

    设计原则：
    - 后端代码路由，不依赖 LLM 的工具选择能力
    - 所有库（ArcPy/GeoPandas/Amap/NetworkX）都是"被调用者"
    - 统一返回 ExecutorResult

    Args:
        task: 任务字典，至少包含 "task" 字段

    Returns:
        ExecutorResult 统一结果
    """
    return get_router().route(task)


def execute_task_by_dict(task: Dict[str, Any]) -> ExecutorResult:
    """
    便捷函数：execute_task 的别名

    Args:
        task: 任务字典

    Returns:
        ExecutorResult
    """
    return execute_task(task)


def execute_scenario(scenario: str, params: Dict[str, Any]) -> ExecutorResult:
    """
    通过场景名称执行任务

    Args:
        scenario: 场景名称（如 "buffer", "route"）
        params: 任务参数字典

    Returns:
        ExecutorResult
    """
    task = {"task": scenario, **params}
    return get_router().route(task)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "TaskRouter",
    "execute_task",
    "execute_task_by_dict",
    "execute_scenario",
    "get_router",
    "SCENARIO_EXECUTOR_KEY",
    "_task_to_scenario",
    "_scenario_to_executor_key",
    "_resolve_executor_key",
]
