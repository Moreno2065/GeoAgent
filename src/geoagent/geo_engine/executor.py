"""
Task DSL Executor - 统一任务执行器
==================================
根据 Engine 名称路由到对应 Engine 执行任务。

核心设计原则：
    1. 后端代码路由，不依赖 LLM
    2. 每个 task = 一个 executor
    3. 无 ReAct 循环
    4. 统一的错误处理
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from geoagent.geo_engine.router import route_task, EngineName, validate_task_structure
from geoagent.geo_engine.data_utils import format_result


# =============================================================================
# 延迟导入 Engine（避免循环依赖）
# =============================================================================

_engine_cache: Dict[str, Any] = {}


def _get_engine(engine_name: str):
    """延迟加载 Engine 类"""
    if engine_name in _engine_cache:
        return _engine_cache[engine_name]

    if engine_name == EngineName.VECTOR:
        from geoagent.geo_engine.vector_engine import VectorEngine
        _engine_cache[engine_name] = VectorEngine
    elif engine_name == EngineName.RASTER:
        from geoagent.geo_engine.raster_engine import RasterEngine
        _engine_cache[engine_name] = RasterEngine
    elif engine_name == EngineName.NETWORK:
        from geoagent.geo_engine.network_engine import NetworkEngine
        _engine_cache[engine_name] = NetworkEngine
    elif engine_name == EngineName.ANALYSIS:
        from geoagent.geo_engine.analysis_engine import AnalysisEngine
        _engine_cache[engine_name] = AnalysisEngine
    elif engine_name == EngineName.IO:
        from geoagent.geo_engine.io_engine import IOEngine
        _engine_cache[engine_name] = IOEngine
    else:
        return None

    return _engine_cache[engine_name]


# =============================================================================
# 结果格式化辅助
# =============================================================================

def _ok(result: Dict[str, Any]) -> str:
    """格式化成功结果为 JSON 字符串"""
    return json.dumps({"success": True, **result}, ensure_ascii=False, indent=2)


def _err(msg: str, detail: str = "") -> str:
    """格式化错误结果为 JSON 字符串"""
    return json.dumps({
        "success": False,
        "error": msg,
        "detail": detail,
    }, ensure_ascii=False, indent=2)


# =============================================================================
# 执行器函数
# =============================================================================

def execute_task(task: Dict[str, Any]) -> str:
    """
    根据 Task DSL 确定性地执行任务

    这是确定性执行的核心入口，不依赖 LLM 的工具选择。

    Args:
        task: Task DSL 字典，格式：
            {
                "task": "route",           # 必填：任务大类
                "type": "shortest_path",   # 必填：操作类型
                "inputs": {...},            # 输入数据
                "params": {...},            # 参数
                "outputs": {...}           # 输出控制（可选）
            }

    Returns:
        JSON 格式的执行结果

    执行流程：
        1. 验证 task 结构
        2. 路由到对应 Engine
        3. 调用 Engine.run(task)
        4. 格式化并返回结果
    """
    # ── 验证结构 ────────────────────────────────────────────────────
    valid, error_msg = validate_task_structure(task)
    if not valid:
        return _err(f"Task DSL 结构无效: {error_msg}")

    # ── 路由到 Engine ──────────────────────────────────────────────
    engine_name = route_task(task)
    Engine = _get_engine(engine_name)

    if Engine is None:
        return _err(
            f"不支持的 Engine: {engine_name}",
            detail=f"可选 Engine: {list(_engine_cache.keys())}"
        )

    # ── 执行 ────────────────────────────────────────────────────────
    try:
        result = Engine.run(task)
        # 如果 Engine.run() 返回的是字典，格式化为 JSON
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return _err(f"任务执行异常 [{task.get('task')}/{task.get('type')}]: {str(e)}")


def execute_task_by_dict(data: Dict[str, Any]) -> str:
    """
    根据字典数据执行任务（便捷函数）

    Args:
        data: 包含 task 字段的字典

    Returns:
        JSON 格式的执行结果
    """
    return execute_task(data)


# =============================================================================
# 快捷执行函数（直接调用单个 Engine）
# =============================================================================

def execute_vector(task: Dict[str, Any]) -> str:
    """直接执行矢量任务"""
    from geoagent.geo_engine.vector_engine import VectorEngine
    try:
        result = VectorEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return _err(f"矢量任务执行失败: {str(e)}")


def execute_raster(task: Dict[str, Any]) -> str:
    """直接执行栅格任务"""
    from geoagent.geo_engine.raster_engine import RasterEngine
    try:
        result = RasterEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return _err(f"栅格任务执行失败: {str(e)}")


def execute_network(task: Dict[str, Any]) -> str:
    """直接执行路网任务"""
    from geoagent.geo_engine.network_engine import NetworkEngine
    try:
        result = NetworkEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return _err(f"路网任务执行失败: {str(e)}")


def execute_analysis(task: Dict[str, Any]) -> str:
    """直接执行分析任务"""
    from geoagent.geo_engine.analysis_engine import AnalysisEngine
    try:
        result = AnalysisEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return _err(f"分析任务执行失败: {str(e)}")


def execute_io(task: Dict[str, Any]) -> str:
    """直接执行 IO 任务"""
    from geoagent.geo_engine.io_engine import IOEngine
    try:
        result = IOEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return _err(f"IO 任务执行失败: {str(e)}")


# =============================================================================
# 新架构：委托给 Executor Layer（推荐）
# =============================================================================

def execute_task_via_executor_layer(task: Dict[str, Any]) -> str:
    """
    通过新 Executor Layer 执行任务（推荐方式）

    这是新架构的入口，将任务委托给 geoagent.executors.TaskRouter。
    所有库（ArcPy, GeoPandas, NetworkX, Amap）都是"被调用者"，
    通过统一的 Executor Layer 进行调度。

    Args:
        task: 任务字典，至少包含 "task" 字段

    Returns:
        JSON 格式的执行结果
    """
    try:
        from geoagent.executors.router import execute_task as executor_layer_run
        from geoagent.executors.base import ExecutorResult

        result: ExecutorResult = executor_layer_run(task)
        if result.success:
            return json.dumps({
                "success": True,
                "task": result.task_type,
                "engine": result.engine,
                "data": result.data,
                "metadata": result.metadata or {},
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": result.error or "执行失败",
                "detail": result.error_detail or "",
            }, ensure_ascii=False, indent=2)
    except ImportError:
        return _err("Executor Layer 不可用，请检查 geoagent.executors 模块")
    except Exception as e:
        return _err(f"Executor Layer 执行失败: {str(e)}")


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "execute_task",
    "execute_task_by_dict",
    "execute_vector",
    "execute_raster",
    "execute_network",
    "execute_analysis",
    "execute_io",
    # 新架构
    "execute_task_via_executor_layer",
]
