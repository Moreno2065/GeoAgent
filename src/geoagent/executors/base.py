"""
BaseExecutor - 统一执行层基类
===============================
所有 Executor 的抽象基类，定义标准接口。

核心契约：
1. run(task: dict) -> ExecutorResult
2. 内部自主选择库（Amap/GeoPandas/NetworkX/ArcPy）
3. 输入/输出统一为 GeoJSON 或标准字典
4. 错误处理统一为 ExecutorResult.error
"""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


# =============================================================================
# 标准数据格式定义（所有 Executor 的输出格式）
# =============================================================================

@dataclass
class ExecutorResult:
    """
    统一执行结果格式

    所有 Executor 返回此格式，确保 TaskRouter 统一处理。
    """
    success: bool
    task_type: str  # "route" / "buffer" / "overlay" / ...
    engine: str  # "amap" / "geopandas" / "networkx" / "arcpy" / "scipy"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为 JSON 字符串（兼容旧接口）"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "task_type": self.task_type,
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
        task_type: str,
        engine: str,
        data: Dict[str, Any],
        **kwargs
    ) -> "ExecutorResult":
        return cls(success=True, task_type=task_type, engine=engine, data=data, **kwargs)

    @classmethod
    def err(
        cls,
        task_type: str,
        error: str,
        engine: str = "unknown",
        error_detail: Optional[str] = None,
        **kwargs
    ) -> "ExecutorResult":
        return cls(
            success=False,
            task_type=task_type,
            engine=engine,
            error=error,
            error_detail=error_detail or traceback.format_exc(),
            **kwargs
        )


# =============================================================================
# BaseExecutor 抽象基类
# =============================================================================

class BaseExecutor(ABC):
    """
    统一 Executor 基类

    所有具体的 Executor 必须继承此类并实现 run() 方法。

    设计原则：
    - 库隔离：每个 Executor 内部导入自己需要的库，不泄露到外部
    - 引擎选择：内部根据参数自动选择最优库（engine 参数或启发式）
    - 统一输出：始终返回 ExecutorResult
    - 可测试：run() 接收 dict，输入输出都是纯数据

    示例：
        class RouteExecutor(BaseExecutor):
            def run(self, task: dict) -> ExecutorResult:
                if task.get("provider") == "amap":
                    return self._run_amap(task)
                else:
                    return self._run_osmnx(task)
    """

    # 子类必须设置
    task_type: str = ""
    supported_engines: Set[str] = set()

    @abstractmethod
    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行任务

        Args:
            task: 任务参数字典（来自 Pydantic 模型解析后的字典）

        Returns:
            ExecutorResult 统一结果格式
        """
        ...

    # ── 通用辅助方法 ────────────────────────────────────────────────────────

    def _workspace_path(self, relative_path: str) -> str:
        """将相对路径转换为 workspace 下的绝对路径"""
        from pathlib import Path
        ws = Path(__file__).parent.parent.parent.parent / "workspace"
        return str(ws / relative_path)

    def _resolve_path(self, file_path: str) -> str:
        """解析文件路径（相对路径 → workspace/ 绝对路径）"""
        from pathlib import Path
        p = Path(file_path)
        if p.is_absolute():
            return str(p)
        return self._workspace_path(file_path)

    def _check_dependency(self, module_name: str) -> bool:
        """检查可选依赖是否可用"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def _warn(self, msg: str) -> None:
        """记录警告信息（在 ExecutorResult.warnings 中累积）"""
        pass  # 由调用方在 run() 中收集

    def get_engine_hint(self, task: Dict[str, Any]) -> Optional[str]:
        """
        从任务参数中提取 engine 提示

        子类可重写此方法实现自己的 engine 选择逻辑。
        """
        return task.get("engine") or task.get("provider") or None


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "BaseExecutor",
    "ExecutorResult",
]
