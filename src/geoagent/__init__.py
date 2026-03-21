"""
GeoAgent - 基于 DeepSeek API 的空间智能 GIS 分析 Agent
================================================================
三层收敛架构：用户输入 → 意图分类 → 动态Schema → Pydantic校验 → 确定性执行

新增 V2 六层架构（推荐）：
    User Input → Intent → Orchestrate → DSL → Execute → Render

V2 核心设计原则：
    1. LLM 只做"翻译"：NL → DSL，不做决策
    2. 所有执行确定性：后端代码路由，无 ReAct 循环
    3. 有限标准场景：route/buffer/overlay/interpolation/viewshed/statistics/raster
    4. 参数补全+澄清机制：参数不完整必须追问
    5. 结果面向业务结论：不是技术结果，是解释卡片

标准用法:
    from geoagent import GeoAgent, GeoAgentV2, create_agent, create_agent_v2

    # V2 推荐用法（六层架构）
    agent = create_agent_v2(api_key="sk-...")
    result = agent.run("芜湖南站到方特的步行路径")
    print(result.to_user_text())

    # V2 快捷方法
    result = agent.route("芜湖南站", "方特欢乐世界", mode="walking")
    result = agent.buffer("schools.shp", 500, unit="meters")
    result = agent.overlay("landuse.shp", "flood.shp", operation="intersect")

    # V2 Pipeline 直接使用
    from geoagent.pipeline import run_pipeline
    result = run_pipeline("芜湖南站到方特的步行路径")

    # V1 用法（向后兼容）
    agent = create_agent(api_key="sk-...")
    result = agent.compile("芜湖南站到方特的步行路径")
"""

from geoagent.version import __version__

# 核心
from geoagent.core import GeoAgent, create_agent

# ── V2 六层架构（推荐）──────────────────────────────────────────────
from geoagent.geoagent_v2 import GeoAgentV2, create_agent_v2

# Pipeline
from geoagent.pipeline import (
    GeoAgentPipeline,
    PipelineResult,
    run_pipeline,
    run_pipeline_mvp,
    get_pipeline,
)

# Layers
from geoagent.layers import (
    # architecture
    Scenario,
    RouteSubType,
    BufferSubType,
    OverlaySubType,
    PipelineStatus,
    ARCHITECTURE_VERSION,
    ARCHITECTURE_NAME,
    # layer 1
    UserInput,
    InputParser,
    parse_user_input,
    # layer 2
    IntentClassifier,
    IntentResult,
    classify_intent,
    # layer 3
    ScenarioOrchestrator,
    OrchestrationResult,
    ClarificationQuestion,
    # layer 4
    GeoDSL,
    SchemaValidator,
    SchemaValidationError,
    # layer 5
    TaskRouter,
    ExecutorResult,
    execute_task,
    # layer 6
    ResultRenderer,
    RenderResult,
    BusinessConclusion,
    ExplanationCard,
    render_result,
)

# 编译器模块
from geoagent.compiler import (
    GISCompiler,
    IntentClassifier,
    classify_intent,
    execute_task,
    TaskType,
    RouteTask, BufferTask, OverlayTask, InterpolationTask,
    ShadowTask, NdviTask, HotspotTask, VisualizationTask, GeneralTask,
)

# Workflow（编译器封装）
from geoagent.workflow import SimpleCompilerWorkflow, create_compiler_workflow

# GeoEngine 统一执行系统
from geoagent.geo_engine import (
    GeoEngine,
    create_geo_engine,
    get_geo_engine,
    route_task,
    ENGINE_MAP,
    TASK_EXECUTOR_KEY,
    route_to_executor,
    geo_execute_task,
    execute_task_via_executor_layer,
)
from geoagent.geo_engine.router import (
    EngineName, TASK_EXAMPLES, validate_task_structure,
)
from geoagent.geo_engine.executor import (
    execute_vector,
    execute_raster,
    execute_network,
    execute_analysis,
    execute_io,
)

# Executor Layer（新架构，优先使用）
from geoagent.executors import (
    BaseExecutor,
    ExecutorResult,
    TaskRouter,
    execute_task as executor_execute_task,
    execute_task_by_dict,
    execute_scenario,
    get_router,
    SCENARIO_EXECUTOR_KEY,
    ScenarioConfig,
    SCENARIO_CONFIGS,
    get_scenario_config,
    get_all_scenarios,
    get_executor_key,
    resolve_engine,
)

# 知识库（可选，优雅降级）
try:
    from geoagent.knowledge import (
        GISKnowledgeBase,
        get_knowledge_base,
        search_gis_knowledge,
    )
except ImportError:
    GISKnowledgeBase = None
    get_knowledge_base = None
    search_gis_knowledge = None

__all__ = [
    "__version__",

    # ── V2 六层架构（推荐）────────────────────────────────────────────
    "GeoAgentV2",
    "create_agent_v2",

    # Pipeline
    "GeoAgentPipeline",
    "PipelineResult",
    "run_pipeline",
    "run_pipeline_mvp",
    "get_pipeline",

    # Layers
    "Scenario",
    "RouteSubType",
    "BufferSubType",
    "OverlaySubType",
    "PipelineStatus",
    "ARCHITECTURE_VERSION",
    "ARCHITECTURE_NAME",
    "UserInput",
    "InputParser",
    "parse_user_input",
    "IntentClassifier",
    "IntentResult",
    "classify_intent",
    "ScenarioOrchestrator",
    "OrchestrationResult",
    "ClarificationQuestion",
    "GeoDSL",
    "SchemaValidator",
    "SchemaValidationError",
    "TaskRouter",
    "ExecutorResult",
    "execute_task",
    "ResultRenderer",
    "RenderResult",
    "BusinessConclusion",
    "ExplanationCard",
    "render_result",

    # 核心
    "GeoAgent",
    "create_agent",

    # 编译器
    "GISCompiler",
    "TaskType",
    "RouteTask", "BufferTask", "OverlayTask", "InterpolationTask",
    "ShadowTask", "NdviTask", "HotspotTask", "VisualizationTask", "GeneralTask",

    # Workflow
    "SimpleCompilerWorkflow",
    "create_compiler_workflow",

    # GeoEngine
    "GeoEngine",
    "create_geo_engine",
    "get_geo_engine",
    "route_task",
    "ENGINE_MAP",
    "TASK_EXECUTOR_KEY",
    "route_to_executor",
    "EngineName",
    "geo_execute_task",
    "execute_task_via_executor_layer",
    "TASK_EXAMPLES",
    "validate_task_structure",
    "execute_vector",
    "execute_raster",
    "execute_network",
    "execute_analysis",
    "execute_io",

    # Executor Layer
    "BaseExecutor",
    "executor_execute_task",
    "execute_task_by_dict",
    "execute_scenario",
    "get_router",
    "SCENARIO_EXECUTOR_KEY",
    "SCENARIO_CONFIGS",
    "get_scenario_config",
    "get_all_scenarios",
    "get_executor_key",
    "resolve_engine",

    # 知识库
    "GISKnowledgeBase",
    "get_knowledge_base",
    "search_gis_knowledge",
]
