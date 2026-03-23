"""
GeoDSL - GeoAgent 统一任务描述语言协议

定义所有 GIS 任务的标准结构，确保：
1. LLM 只做翻译（自然语言 → DSL）
2. 执行层完全确定性
3. 结果标准化输出

【重构说明】
- ScenarioType 现已废弃，请使用 layers.architecture.Scenario
- 此模块保留向后兼容导出
"""

from geoagent.layers.architecture import Scenario

# 向后兼容导出（已废弃）
from .protocol import (
    ScenarioType,  # 向后兼容别名，已废弃
    OutputSpec,
    GeoDSL,
    ExecutorResult,
    get_scenario_executor_map,
)

__all__ = [
    # 主要类型
    "Scenario",  # 唯一的 Scenario 枚举
    # 向后兼容（已废弃）
    "ScenarioType",  # 已废弃，请使用 Scenario
    "OutputSpec",
    "GeoDSL",
    "ExecutorResult",
    "get_scenario_executor_map",
]
