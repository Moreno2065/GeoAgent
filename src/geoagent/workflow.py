"""
GeoAgent Workflow 模块
=====================
六层架构 Workflow 封装。

支持：
1. 六层 Pipeline（确定性优先）
2. 编译器 Workflow（向后兼容）
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generator, Optional

from geoagent.compiler import GISCompiler
from geoagent.layers.pipeline import SixLayerPipeline, PipelineConfig, run_pipeline


# =============================================================================
# 六层 Pipeline Workflow
# =============================================================================

def create_pipeline_workflow(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    max_retries: int = 3,
    enable_fallback: bool = True,
) -> "PipelineWorkflow":
    """
    创建六层 Pipeline Workflow

    Args:
        api_key: API 密钥
        model: 模型名称
        base_url: API 基础 URL
        max_retries: 最大重试次数
        enable_fallback: 是否启用 fallback

    Returns:
        PipelineWorkflow 实例
    """
    return PipelineWorkflow(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_retries=max_retries,
        enable_fallback=enable_fallback,
    )


class PipelineWorkflow:
    """
    六层 Pipeline Workflow

    封装六层架构的处理流水线，提供简洁的 run/run_stream 接口。

    使用方式：
        workflow = PipelineWorkflow()
        result = workflow.run("芜湖南站到方特的步行路径")
        if result.success:
            print(result.rendered_result)
        elif result.needs_clarification:
            for q in result.questions:
                print(q["question"])
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        enable_fallback: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback
        self.stats = {"total_requests": 0, "successful": 0, "failed": 0}

    def run(
        self,
        user_query: str,
        event_callback: Optional[Callable] = None,
        thread_id: str = "default",
    ) -> "PipelineResult":
        """
        同步执行

        Args:
            user_query: 用户查询
            event_callback: 事件回调
            thread_id: 线程 ID（保留，用于未来扩展）

        Returns:
            PipelineResult 对象
        """
        self.stats["total_requests"] += 1

        config = PipelineConfig(
            enable_clarification=True,
            enable_fallback=self.enable_fallback,
            max_retries=self.max_retries,
            event_callback=event_callback,
        )

        from geoagent.layers.pipeline import get_pipeline
        pipeline = get_pipeline(config)
        result = pipeline.run(user_query)

        if result.success:
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

        Args:
            user_query: 用户查询
            event_callback: 事件回调
            thread_id: 线程 ID

        Yields:
            各阶段的事件
        """
        self.stats["total_requests"] += 1

        config = PipelineConfig(event_callback=event_callback)
        pipeline = SixLayerPipeline(config)

        success = False
        for event in pipeline.run_stream(user_query):
            yield event
            if event.get("event") == "layer6_completed":
                success = event.get("status") == "completed"

        if success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1

    def reset(self) -> None:
        """重置统计"""
        self.stats = {"total_requests": 0, "successful": 0, "failed": 0}


# =============================================================================
# 编译器 Workflow（向后兼容）
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
    简化的编译器 Workflow（三层收敛架构）

    这是三层收敛架构的轻量级封装，保持向后兼容。
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


# =============================================================================
# 默认 Workflow 工厂（推荐使用六层架构）
# =============================================================================

def create_workflow(
    api_key: str = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    max_retries: int = 3,
    enable_fallback: bool = True,
    use_six_layer: bool = True,
) -> Any:
    """
    创建默认 Workflow

    Args:
        api_key: API 密钥
        model: 模型名称
        base_url: API 基础 URL
        max_retries: 最大重试次数
        enable_fallback: 是否启用 fallback
        use_six_layer: 是否使用六层架构（True=推荐，False=向后兼容）

    Returns:
        PipelineWorkflow 或 SimpleCompilerWorkflow 实例
    """
    if use_six_layer:
        return create_pipeline_workflow(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            enable_fallback=enable_fallback,
        )
    else:
        return create_compiler_workflow(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
        )


__all__ = [
    # 六层架构
    "PipelineWorkflow",
    "create_pipeline_workflow",
    # 三层收敛（向后兼容）
    "SimpleCompilerWorkflow",
    "create_compiler_workflow",
    # 默认工厂
    "create_workflow",
]
