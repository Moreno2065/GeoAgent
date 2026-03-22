"""
第5层：执行引擎层（Task Router + Executors）
=============================================
核心职责：
1. 确定性执行：读 DSL → 查路由表 → 调用固定函数 → 生成结果
2. 不让 LLM 直接碰执行逻辑
3. 所有库（ArcPy, GeoPandas, NetworkX, Amap, PostGIS）都是被调用者

设计原则：
- 后端代码路由，不依赖 LLM
- TaskRouter 是唯一的调度入口（统一由 executors/router.py 提供）
- 统一 ExecutorResult 格式（统一由 executors/base.py 提供）
- 本模块是 layers 层对 executors 层的代理门面

重要重构说明（2026-03-21）：
- ExecutorResult 已统一到 geoagent.executors.base
- TaskRouter 已统一到 geoagent.executors.router
- 本模块不再重复定义，而是委托给 executors 层
- 保留 execute_* 便捷函数作为 layers 层的 API 入口
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List

# ── 统一委托给 executors 层 ────────────────────────────────────────────────
# 所有类型和核心逻辑都委托给 geoagent.executors
from geoagent.executors.base import BaseExecutor, ExecutorResult as _BaseExecutorResult
from geoagent.executors.router import (
    TaskRouter,
    execute_task as _execute_task,
    get_router as _get_router,
)


# =============================================================================
# ExecutorResult（委托给 executors.base，保持向后兼容）
# =============================================================================

ExecutorResult = _BaseExecutorResult


# =============================================================================
# TaskRouter（委托给 executors.router，保持向后兼容）
# =============================================================================

SCENARIO_EXECUTOR_MAP: Dict[str, str] = {
    "route": "route",
    "buffer": "buffer",
    "overlay": "overlay",
    "interpolation": "idw",
    "viewshed": "shadow",
    "statistics": "hotspot",
    "raster": "ndvi",
    "suitability": "suitability",  # MCDA 适宜性选址分析
    "code_sandbox": "code_sandbox",  # 受限代码执行（补丁层）
    # ── Amap 高德 Web 服务 ────────────────────────────────────────
    "geocode": "amap",
    "regeocode": "amap",
    "district": "amap",
    "static_map": "amap",
    "coord_convert": "amap",
    "grasp_road": "amap",
    "input_tips": "amap",
    "poi_search": "amap",
    "traffic_status": "amap",
    "traffic_events": "amap",
    "transit_info": "amap",
    "ip_location": "amap",
    "weather": "amap",
}


class TaskRouter(_get_router().__class__):
    """任务路由器（委托给 geoagent.executors.router.TaskRouter）"""

    def __init__(self):
        super().__init__()

    def route(self, scenario_or_str, task_dict: Dict[str, Any]) -> ExecutorResult:
        """执行任务路由"""
        from geoagent.layers.architecture import Scenario

        # 安全获取 scenario 字符串值
        if hasattr(scenario_or_str, 'value'):
            scenario_str = scenario_or_str.value
        elif isinstance(scenario_or_str, str):
            scenario_str = scenario_or_str
        else:
            scenario_str = str(scenario_or_str)

        executor_key = SCENARIO_EXECUTOR_MAP.get(scenario_str, "general")
        task_dict = dict(task_dict)
        task_dict["task"] = scenario_str

        return _execute_task(task_dict)


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

def execute_task(scenario: Any, task_dict: Dict[str, Any]) -> ExecutorResult:
    """
    统一任务执行入口

    确定性路由：scenario + task_dict → Executor.run()

    设计原则：
    - 后端代码路由，不依赖 LLM
    - 所有库（ArcPy/GeoPandas/Amap/NetworkX）都是"被调用者"
    - 统一返回 ExecutorResult

    Args:
        scenario: 场景类型（Scenario 枚举或字符串）
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
    return execute_task("route", task_dict)


def execute_buffer(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行缓冲区分析任务"""
    return execute_task("buffer", task_dict)


def execute_overlay(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行叠置分析任务"""
    return execute_task("overlay", task_dict)


def execute_interpolation(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行插值分析任务"""
    return execute_task("interpolation", task_dict)


def execute_viewshed(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行视域分析任务"""
    return execute_task("viewshed", task_dict)


def execute_statistics(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行统计分析任务"""
    return execute_task("statistics", task_dict)


def execute_raster(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行栅格分析任务"""
    return execute_task("raster", task_dict)


def execute_shadow(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行阴影分析任务"""
    return execute_task("shadow_analysis", task_dict)


def execute_ndvi(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行 NDVI 任务"""
    return execute_task("ndvi", task_dict)


def execute_hotspot(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行热点分析任务"""
    return execute_task("hotspot", task_dict)


def execute_visualization(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行可视化任务"""
    return execute_task("visualization", task_dict)


def execute_accessibility(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行可达性分析任务"""
    return execute_task("accessibility", task_dict)


def execute_suitability(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行选址分析任务"""
    return execute_task("suitability", task_dict)


def execute_general(task_dict: Dict[str, Any]) -> ExecutorResult:
    """执行通用任务"""
    return execute_task("general", task_dict)


def execute_code_sandbox(task_dict: Dict[str, Any]) -> ExecutorResult:
    """
    执行受限代码沙盒任务。

    由 LLM 生成自定义 Python 代码，在受控环境中执行。
    仅在标准 Executor 无法覆盖的任务时使用。

    Args:
        task_dict: 包含以下字段的字典：
            - code: LLM 生成的 Python 代码
            - context_data: 数据上下文（可选）
            - description: 任务描述（可选）
            - timeout_seconds: 超时时间（默认 60.0）
            - mode: "exec" 或 "eval"（默认 "exec"）

    Returns:
        ExecutorResult
    """
    from geoagent.executors.code_sandbox_executor import CodeSandboxExecutor
    executor = CodeSandboxExecutor()
    return executor.run(task_dict)


# =============================================================================
# Capability Registry 便捷函数（54 个标准化能力节点）
# =============================================================================

def execute_capability(capability_name: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> ExecutorResult:
    """
    通过 CapabilityRegistry 执行标准化能力

    Args:
        capability_name: 能力名称（如 "vector_buffer", "raster_ndvi" 等）
        inputs: 输入数据
        params: 分析参数

    Returns:
        ExecutorResult
    """
    from geoagent.geo_engine.capability import execute_capability as cap_exec
    return cap_exec(capability_name, inputs, params)


def execute_capability_from_task(task: Dict[str, Any]) -> ExecutorResult:
    """
    通过 Task DSL 执行能力

    Args:
        task: Task DSL 字典，包含 task/inputs/params 字段

    Returns:
        ExecutorResult
    """
    from geoagent.geo_engine.capability.router import execute_capability_task
    return execute_capability_task(task)


def list_all_capabilities() -> List[str]:
    """列出所有可用能力"""
    from geoagent.geo_engine.capability import list_capabilities
    return list_capabilities()


def list_capabilities_by_category(category: str) -> List[str]:
    """按类别列出能力"""
    from geoagent.geo_engine.capability import list_capabilities
    return list_capabilities(category=category)


def search_capabilities(query: str) -> List[str]:
    """搜索能力"""
    from geoagent.geo_engine.capability import search_capabilities as search
    return search(query)


def get_capability_info(name: str) -> Optional[Dict[str, Any]]:
    """获取能力详细信息"""
    from geoagent.geo_engine.capability import capability_info
    return capability_info(name)


__all__ = [
    "ExecutorResult",
    "BaseExecutor",
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
    "execute_shadow",
    "execute_ndvi",
    "execute_hotspot",
    "execute_visualization",
    "execute_accessibility",
    "execute_suitability",
    "execute_general",
    "execute_code_sandbox",
    # Capability Registry
    "execute_capability",
    "execute_capability_from_task",
    "list_all_capabilities",
    "list_capabilities_by_category",
    "search_capabilities",
    "get_capability_info",
]
