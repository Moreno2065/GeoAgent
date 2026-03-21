"""
GeoAgent Compiler Package
========================
任务编译器模块：三层收敛架构

Architecture:
    User Input → Intent Classifier → Dynamic Schema Injection → Pydantic Validation → Execution
"""

from geoagent.compiler.task_schema import (
    # 枚举
    TaskType,
    RouteMode,
    BufferUnit,
    OverlayOperation,
    InterpolationMethod,
    HotspotNeighborStrategy,
    VisualizationType,
    # 基础模型
    BaseTask,
    # 任务模型
    RouteTask,
    BufferTask,
    OverlayTask,
    InterpolationTask,
    ShadowTask,
    NdviTask,
    HotspotTask,
    VisualizationTask,
    GeneralTask,
    # 工具函数
    TaskModel,
    TASK_MODEL_MAP,
    get_task_schema_json,
    get_all_task_schemas,
    get_task_description,
    parse_task_from_dict,
    parse_task_from_json,
)

from geoagent.compiler.intent_classifier import (
    IntentClassifier,
    classify_intent,
    classify_intent_simple,
    get_classifier,
    get_task_type_for_intent,
    IntentResult,
    INTENT_KEYWORDS,
    # 新增
    ClarificationEngine,
    CLARIFICATION_TEMPLATES,
    SCENARIO_SUBTYPES,
)

from geoagent.compiler.orchestrator import (
    ScenarioOrchestrator,
    OrchestrationResult,
    ParameterExtractor,
    orchestrate,
    get_orchestrator,
)

from geoagent.compiler.task_executor import (
    execute_task,
    execute_task_by_dict,
)

from geoagent.compiler.compiler import (
    GISCompiler,
)

__all__ = [
    # task_schema
    "TaskType", "RouteMode", "BufferUnit", "OverlayOperation",
    "InterpolationMethod", "HotspotNeighborStrategy", "VisualizationType",
    "BaseTask", "RouteTask", "BufferTask", "OverlayTask", "InterpolationTask",
    "ShadowTask", "NdviTask", "HotspotTask", "VisualizationTask", "GeneralTask",
    "TaskModel", "TASK_MODEL_MAP", "get_task_schema_json",
    "get_all_task_schemas", "get_task_description",
    "parse_task_from_dict", "parse_task_from_json",
    # intent_classifier
    "IntentClassifier", "classify_intent", "classify_intent_simple",
    "get_classifier", "get_task_type_for_intent", "IntentResult",
    "INTENT_KEYWORDS",
    "ClarificationEngine", "CLARIFICATION_TEMPLATES", "SCENARIO_SUBTYPES",
    # orchestrator
    "ScenarioOrchestrator", "OrchestrationResult", "ParameterExtractor",
    "orchestrate", "get_orchestrator",
    # task_executor
    "execute_task", "execute_task_by_dict",
    # compiler
    "GISCompiler",
]
