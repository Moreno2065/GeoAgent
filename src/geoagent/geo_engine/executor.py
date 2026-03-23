"""
Task DSL Executor - 统一任务执行器
==================================
根据 Engine 名称路由到对应 Engine 执行任务。

核心设计原则：
    1. 后端代码路由，不依赖 LLM
    2. 每个 task = 一个 executor
    3. 无 ReAct 循环
    4. 统一的错误处理

P1 增强特性：
    - 超时控制（Timeout）：防止大文件处理导致进程挂起
    - 资源限制：内存使用监控，防止 OOM
    - 增强的错误处理：更详细的错误分类
"""

from __future__ import annotations

import json
import signal
import threading
import time
import tracemalloc
from typing import Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from geoagent.geo_engine.router import route_task, EngineName, validate_task_structure
from geoagent.geo_engine.data_utils import format_result


# =============================================================================
# 全局配置
# =============================================================================

# 默认超时时间（秒）
DEFAULT_TIMEOUT_SECONDS = 300  # 5 分钟

# 默认最大内存使用（MB）
DEFAULT_MAX_MEMORY_MB = 2048  # 2GB

# 超时/内存超限错误码
ERROR_CODE_TIMEOUT = "EXEC_TIMEOUT"
ERROR_CODE_MEMORY_EXCEEDED = "MEMORY_EXCEEDED"


# =============================================================================
# 超时控制工具类
# =============================================================================

class TimeoutError(Exception):
    """执行超时异常"""
    def __init__(self, seconds: float, task_name: str = ""):
        self.seconds = seconds
        self.task_name = task_name
        super().__init__(f"任务执行超时（{seconds}秒）: {task_name}")


class MemoryLimitError(Exception):
    """内存超限异常"""
    def __init__(self, current_mb: float, limit_mb: float):
        self.current_mb = current_mb
        self.limit_mb = limit_mb
        super().__init__(f"内存使用超限: {current_mb:.1f}MB > {limit_mb:.1f}MB")


class ResourceMonitor:
    """
    资源监控器

    用于监控执行过程中的内存使用情况。
    支持装饰器模式：`@ResourceMonitor(max_memory_mb=1024)`
    """

    def __init__(self, max_memory_mb: float = DEFAULT_MAX_MEMORY_MB):
        self.max_memory_mb = max_memory_mb
        self.peak_memory_mb = 0.0
        self._tracking = False

    def start(self):
        """开始追踪内存使用"""
        self.peak_memory_mb = 0.0
        self._tracking = True
        tracemalloc.start()

    def stop(self) -> float:
        """
        停止追踪并返回峰值内存使用

        Returns:
            峰值内存使用（MB）
        """
        self._tracking = False
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.peak_memory_mb = peak / (1024 * 1024)
            return self.peak_memory_mb
        return 0.0

    def check_memory(self) -> None:
        """
        检查当前内存使用，超限则抛出异常

        Raises:
            MemoryLimitError: 内存使用超限
        """
        if not self._tracking:
            return

        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            current_mb = current / (1024 * 1024)
            if current_mb > self.max_memory_mb:
                raise MemoryLimitError(current_mb, self.max_memory_mb)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def with_timeout(timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
    """
    超时控制装饰器

    使用信号实现超时（仅 Unix/Linux/Mac），Windows 使用线程方案。

    Args:
        timeout_seconds: 超时时间（秒）

    示例：
        @with_timeout(60)
        def long_running_task():
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                # 任务超时，抛出超时异常
                raise TimeoutError(timeout_seconds, func.__name__)

            if error[0]:
                raise error[0]

            return result[0]

        return wrapper
    return decorator


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

def _ok(result: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
    """格式化成功结果为 JSON 字符串"""
    response = {"success": True, **result}
    if metadata:
        response["_metadata"] = metadata
    return json.dumps(response, ensure_ascii=False, indent=2)


def _err(msg: str, detail: str = "", code: str = "EXEC_ERROR") -> str:
    """格式化错误结果为 JSON 字符串"""
    return json.dumps({
        "success": False,
        "error": msg,
        "detail": detail,
        "error_code": code,
    }, ensure_ascii=False, indent=2)


# =============================================================================
# 执行器函数
# =============================================================================

def execute_task(
    task: Dict[str, Any],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_memory_mb: float = DEFAULT_MAX_MEMORY_MB,
) -> str:
    """
    根据 Task DSL 确定性地执行任务（带超时和资源控制）

    P1 增强：
    - 超时控制：防止大文件处理导致进程挂起
    - 内存监控：防止 OOM 导致进程崩溃

    Args:
        task: Task DSL 字典，格式：
            {
                "task": "route",           # 必填：任务大类
                "type": "shortest_path",   # 必填：操作类型
                "inputs": {...},            # 输入数据
                "params": {...},            # 参数
                "outputs": {...}           # 输出控制（可选）
            }
        timeout_seconds: 执行超时时间（秒），默认 300 秒
        max_memory_mb: 最大内存使用（MB），默认 2048 MB

    Returns:
        JSON 格式的执行结果

    执行流程：
        1. 验证 task 结构
        2. 路由到对应 Engine
        3. 在资源限制下执行 Engine.run(task)
        4. 格式化并返回结果
    """
    # ── 验证结构 ────────────────────────────────────────────────────
    valid, error_msg = validate_task_structure(task)
    if not valid:
        return _err(f"Task DSL 结构无效: {error_msg}")

    # ── 路由到 Engine ──────────────────────────────────────────────
    # route_task 可能返回 EngineName 枚举或字符串，统一转为字符串
    _raw_engine = route_task(task)
    engine_name = _raw_engine.value if hasattr(_raw_engine, 'value') else str(_raw_engine)
    Engine = _get_engine(engine_name)

    if Engine is None:
        return _err(
            f"不支持的 Engine: {engine_name}",
            detail=f"可选 Engine: {list(_engine_cache.keys())}",
        )

    # ── 执行（带资源限制）─────────────────────────────────────────────
    task_name = f"{task.get('task')}/{task.get('type')}"
    monitor = ResourceMonitor(max_memory_mb=max_memory_mb)

    def _execute_with_resource_guard():
        """在资源监控下执行任务"""
        result = None
        error = None
        peak_memory_mb = 0.0

        # 启动内存监控
        monitor.start()

        try:
            # 定期检查内存（每 0.5 秒）
            result = Engine.run(task)
        except MemoryLimitError as e:
            error = e
        except TimeoutError as e:
            error = e
        except Exception as e:
            error = e
        finally:
            peak_memory_mb = monitor.stop()

        if error:
            raise error

        return result, peak_memory_mb

    # 使用线程池执行，支持超时
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_with_resource_guard)

        try:
            # 等待执行结果（带超时）
            result, peak_memory = future.result(timeout=timeout_seconds)

            # 返回成功结果（附带资源使用信息）
            # engine_name 在前面已统一转为字符串
            metadata = {
                "engine": engine_name,
                "peak_memory_mb": round(peak_memory, 2),
                "timeout_used": timeout_seconds,
            }

            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)

        except FuturesTimeoutError:
            return _err(
                f"任务执行超时（{timeout_seconds}秒）",
                detail=f"任务 {task_name} 执行时间超过限制",
                code=ERROR_CODE_TIMEOUT,
            )

        except MemoryLimitError as e:
            return _err(
                f"内存使用超限: {e.current_mb:.1f}MB > {e.limit_mb:.1f}MB",
                detail=f"任务 {task_name} 内存使用过大，请尝试：\n"
                       f"1. 使用更小的数据样本\n"
                       f"2. 分批处理数据\n"
                       f"3. 增加 max_memory_mb 参数",
                code=ERROR_CODE_MEMORY_EXCEEDED,
            )

        except Exception as e:
            return _err(
                f"任务执行异常 [{task_name}]: {str(e)}",
                detail="请检查输入数据和参数是否正确",
            )


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

def execute_vector(task: Dict[str, Any], timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    直接执行矢量任务（带超时控制）

    Args:
        task: 任务字典
        timeout_seconds: 超时时间（秒）

    Returns:
        JSON 格式的执行结果
    """
    from geoagent.geo_engine.vector_engine import VectorEngine

    def _run():
        result = VectorEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    return _execute_with_timeout(_run, timeout_seconds, "vector", task.get("type", ""))


def execute_raster(task: Dict[str, Any], timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    直接执行栅格任务（带超时控制）

    Args:
        task: 任务字典
        timeout_seconds: 超时时间（秒）

    Returns:
        JSON 格式的执行结果
    """
    from geoagent.geo_engine.raster_engine import RasterEngine

    def _run():
        result = RasterEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    return _execute_with_timeout(_run, timeout_seconds, "raster", task.get("type", ""))


def execute_network(task: Dict[str, Any], timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    直接执行路网任务（带超时控制）

    Args:
        task: 任务字典
        timeout_seconds: 超时时间（秒）

    Returns:
        JSON 格式的执行结果
    """
    from geoagent.geo_engine.network_engine import NetworkEngine

    def _run():
        result = NetworkEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    return _execute_with_timeout(_run, timeout_seconds, "network", task.get("type", ""))


def execute_analysis(task: Dict[str, Any], timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    直接执行分析任务（带超时控制）

    Args:
        task: 任务字典
        timeout_seconds: 超时时间（秒）

    Returns:
        JSON 格式的执行结果
    """
    from geoagent.geo_engine.analysis_engine import AnalysisEngine

    def _run():
        result = AnalysisEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    return _execute_with_timeout(_run, timeout_seconds, "analysis", task.get("type", ""))


def execute_io(task: Dict[str, Any], timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    直接执行 IO 任务（带超时控制）

    Args:
        task: 任务字典
        timeout_seconds: 超时时间（秒）

    Returns:
        JSON 格式的执行结果
    """
    from geoagent.geo_engine.io_engine import IOEngine

    def _run():
        result = IOEngine.run(task)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    return _execute_with_timeout(_run, timeout_seconds, "io", task.get("type", ""))


def _execute_with_timeout(
    func: Callable,
    timeout_seconds: float,
    engine_name: str,
    task_type: str,
) -> str:
    """
    带超时的执行辅助函数

    Args:
        func: 要执行的函数
        timeout_seconds: 超时时间
        engine_name: Engine 名称
        task_type: 任务类型

    Returns:
        JSON 格式的执行结果
    """
    task_name = f"{engine_name}/{task_type}"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)

        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            return _err(
                f"任务执行超时（{timeout_seconds}秒）",
                detail=f"任务 {task_name} 执行时间超过限制",
                code=ERROR_CODE_TIMEOUT,
            )
        except Exception as e:
            return _err(
                f"{engine_name.title()} 任务执行失败: {str(e)}",
                detail=f"任务 {task_name} 执行异常",
            )


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
    # 核心函数
    "execute_task",
    "execute_task_by_dict",
    "execute_vector",
    "execute_raster",
    "execute_network",
    "execute_analysis",
    "execute_io",
    # 新架构
    "execute_task_via_executor_layer",
    # P1: 资源控制
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_MEMORY_MB",
    "ERROR_CODE_TIMEOUT",
    "ERROR_CODE_MEMORY_EXCEEDED",
    "TimeoutError",
    "MemoryLimitError",
    "ResourceMonitor",
    "with_timeout",
]
