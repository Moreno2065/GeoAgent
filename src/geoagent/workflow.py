"""
GeoAgent Workflow 模块
=====================
三层收敛编译器 Workflow 封装。

本模块仅包含编译器相关的 Workflow 封装。
LangGraph 相关代码已移除。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generator, Optional

from geoagent.compiler import GISCompiler


# =============================================================================
# 编译器 Workflow
# =============================================================================

def create_compiler_workflow(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    max_retries: int = 3,
) -> "SimpleCompilerWorkflow":
    """
    创建三层收敛编译器 Workflow

    Args:
        api_key: API 密钥
        model: 模型名称
        base_url: API 基础 URL
        max_retries: 最大重试次数

    Returns:
        SimpleCompilerWorkflow 实例
    """
    return SimpleCompilerWorkflow(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_retries=max_retries,
    )


class SimpleCompilerWorkflow:
    """
    简化的编译器 Workflow

    这是三层收敛架构的轻量级封装。
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
    ):
        self.compiler = GISCompiler(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
        )
        self.stats = {"total_requests": 0, "successful": 0, "failed": 0}

    def run(
        self,
        user_query: str,
        event_callback: Optional[Callable] = None,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        同步执行

        Args:
            user_query: 用户查询
            event_callback: 事件回调
            thread_id: 线程 ID

        Returns:
            执行结果
        """
        self.stats["total_requests"] += 1
        result = self.compiler.compile(user_query, event_callback)
        if result.get("success"):
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1
        return result

    def run_stream(
        self,
        user_query: str,
        event_callback: Optional[Callable] = None,
        thread_id: str = "default",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式执行

        Yields:
            各阶段的事件
        """
        for event in self.compiler.compile_stream(user_query, event_callback):
            yield event

    def reset(self) -> None:
        """重置统计"""
        self.compiler.reset_stats()
        self.stats = {"total_requests": 0, "successful": 0, "failed": 0}


__all__ = [
    "SimpleCompilerWorkflow",
    "create_compiler_workflow",
]
