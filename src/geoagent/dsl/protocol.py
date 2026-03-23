"""
GeoDSL 协议定义
================
GeoAgent 统一任务描述语言（GeoDSL）的核心协议定义。

设计原则：
1. LLM 只做翻译（自然语言 → 结构化 DSL），不参与决策
2. 所有任务必须符合本协议定义的格式
3. 执行层完全确定性，不依赖 LLM 的后续判断

【重构说明】
- 此模块已重构为核心 DSL 定义和向后兼容导出层
- 主要模型定义已迁移到 layers/layer4_dsl.py
- ScenarioType 已废弃，请使用 layers.architecture.Scenario
- 场景到执行器的映射表统一在 executors/router.py 中
"""

from __future__ import annotations

from enum import Enum

# 导入唯一的 Scenario 枚举（从 architecture 模块）
from geoagent.layers.architecture import Scenario as _Scenario

# =============================================================================
# 向后兼容别名（废弃警告）
# =============================================================================

class ScenarioType(str, Enum):
    """
    GeoDSL 支持的场景类型枚举（已废弃，请使用 layers.architecture.Scenario）
    
    此枚举仅用于向后兼容，最终会被移除。
    唯一真实的场景枚举定义在 layers.architecture.Scenario
    """
    # 使用 _Scenario 的所有值创建别名
    ROUTE = _Scenario.ROUTE.value
    BUFFER = _Scenario.BUFFER.value
    OVERLAY = _Scenario.OVERLAY.value
    INTERPOLATION = _Scenario.INTERPOLATION.value
    SHADOW_ANALYSIS = _Scenario.SHADOW_ANALYSIS.value
    VIEWSHED = _Scenario.VIEWSHED.value
    NDVI = _Scenario.NDVI.value
    HOTSPOT = _Scenario.HOTSPOT.value
    VISUALIZATION = _Scenario.VISUALIZATION.value
    ACCESSIBILITY = _Scenario.ACCESSIBILITY.value
    SUITABILITY = _Scenario.SUITABILITY.value
    CODE_SANDBOX = _Scenario.CODE_SANDBOX.value
    MULTI_CRITERIA_SEARCH = _Scenario.MULTI_CRITERIA_SEARCH.value
    GENERAL = "general"  # GENERAL 在 _Scenario 中不存在，使用字符串值
    
    @classmethod
    def from_scenario(cls, scenario: "_Scenario") -> "ScenarioType":
        """从 layers.architecture.Scenario 转换"""
        return cls(scenario.value)
    
    def to_scenario(self) -> "_Scenario":
        """转换为 layers.architecture.Scenario"""
        try:
            return _Scenario(self.value)
        except ValueError:
            return _Scenario.ROUTE  # 默认使用 ROUTE


# =============================================================================
# 主要模型（从 layers.layer4_dsl 导入，消除重复定义）
# =============================================================================

# 从 layers.layer4_dsl 导入核心模型
from geoagent.layers.layer4_dsl import (
    OutputSpec,
    GeoDSL,
    VisualizationSpec,
    ViewSpec,
    LayerSpec,
    WorkflowStep,
    SCHEMA_REQUIRED_PARAMS,
    SchemaValidationError,
    SchemaValidator,
)

# 从 layers.layer3_orchestrate 导入编排相关模型
try:
    from geoagent.layers.layer3_orchestrate import ClarificationQuestion, OrchestrationStatus
except ImportError:
    # 如果导入失败，定义简化的版本
    from pydantic import BaseModel, Field
    
    class ClarificationQuestion(BaseModel):
        """澄清问题定义"""
        field: str = Field(description="需要澄清的字段名")
        question: str = Field(description="追问话术")
        options: list = Field(default_factory=list, description="可选答案列表")
        required: bool = Field(default=True, description="是否必填")
    
    class OrchestrationStatus(str, Enum):
        """编排状态枚举"""
        READY = "ready"
        CLARIFICATION_NEEDED = "clarification_needed"
        INVALID = "invalid"
        EXECUTING = "executing"
        COMPLETED = "completed"
        FAILED = "failed"

# 从 executors.base 导入执行结果模型
try:
    from geoagent.executors.base import ExecutorResult
except ImportError:
    # 如果导入失败，定义简化的版本
    from pydantic import BaseModel, Field
    from typing import Optional as _Optional
    
    class ExecutorResult(BaseModel):
        """执行结果"""
        success: bool = Field(description="执行是否成功")
        scenario: str = Field(description="场景类型")
        task: str = Field(description="任务类型")
        summary: str = Field(default="", description="业务结论摘要")
        detail: _Optional[str] = Field(default=None, description="技术详情")
        map_file: _Optional[str] = Field(default=None, description="地图文件路径")
        output_files: list = Field(default_factory=list, description="输出文件列表")
        metrics: dict = Field(default_factory=dict, description="数值指标")
        error: _Optional[str] = Field(default=None, description="错误信息")


# =============================================================================
# 场景到执行器的映射表（唯一真相来源）
# =============================================================================

def get_scenario_executor_map() -> dict[str, str]:
    """
    获取场景到执行器的映射表（唯一真相来源）
    
    统一从 geoagent.executors.router.SCENARIO_EXECUTOR_KEY 获取。
    此函数保留用于向后兼容。

    Returns:
        dict: scenario → executor key 映射表
    """
    try:
        from geoagent.executors.router import SCENARIO_EXECUTOR_KEY
        return dict(SCENARIO_EXECUTOR_KEY)
    except ImportError:
        # Fallback：硬编码值（不应该发生）
        return {
            "route": "route",
            "buffer": "buffer",
            "overlay": "overlay",
            "interpolation": "interpolation",
            "shadow_analysis": "shadow_analysis",
            "ndvi": "ndvi",
            "hotspot": "hotspot",
            "visualization": "visualization",
            "accessibility": "route",
            "suitability": "general",
            "general": "general",
        }


# 向后兼容别名
SCENARIO_EXECUTOR_MAP = get_scenario_executor_map()


# =============================================================================
# 工具函数
# =============================================================================

def validate_geo_dsl(data: dict) -> GeoDSL:
    """
    校验并解析 GeoDSL 数据

    Args:
        data: 待校验的字典数据

    Returns:
        GeoDSL: 校验通过的任务对象

    Raises:
        SchemaValidationError: 校验失败时抛出
    """
    return GeoDSL(**data)


def _dsl_to_task_type(scenario: ScenarioType) -> str:
    """将场景类型映射为任务类型"""
    return scenario.value


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # Scenario（唯一枚举）
    "ScenarioType",  # 向后兼容别名
    # 核心模型
    "OutputSpec",
    "GeoDSL",
    "VisualizationSpec",
    "ViewSpec",
    "LayerSpec",
    "WorkflowStep",
    # 编排相关
    "ClarificationQuestion",
    "OrchestrationStatus",
    # 执行相关
    "ExecutorResult",
    "SchemaValidationError",
    "SchemaValidator",
    "SCHEMA_REQUIRED_PARAMS",
    # 路由
    "get_scenario_executor_map",
    "SCENARIO_EXECUTOR_MAP",
    # 工具函数
    "validate_geo_dsl",
    "_dsl_to_task_type",
]
