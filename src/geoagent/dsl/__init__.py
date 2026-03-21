"""
GeoDSL - GeoAgent 统一任务描述语言协议

定义所有 GIS 任务的标准结构，确保：
1. LLM 只做翻译（自然语言 → DSL）
2. 执行层完全确定性
3. 结果标准化输出
"""

from .protocol import (
    ScenarioType,
    OutputSpec,
    GeoDSL,
    ExecutorResult,
    get_scenario_executor_map,
)

__all__ = [
    "ScenarioType",
    "OutputSpec",
    "GeoDSL",
    "ExecutorResult",
    "get_scenario_executor_map",
]
