"""
GeoAgent Layers Package
=====================
六层架构的各个层：
  Layer 1: Input Layer（用户输入层）
  Layer 2: Intent Classifier（意图识别层）
  Layer 3: Scenario Orchestrator（场景编排层）
  Layer 4: Task DSL Builder（任务编译层）
  Layer 5: Task Router + Executors（执行引擎层）
  Layer 6: Result Renderer（结果呈现层）—— 已由 renderer/result_renderer.py 提供
"""

from geoagent.layers.architecture import (
    Scenario,
    RouteSubType,
    BufferSubType,
    OverlaySubType,
    InterpolationSubType,
    ViewshedSubType,
    StatisticsSubType,
    PipelineStatus,
    Engine,
    ARCHITECTURE_VERSION,
    ARCHITECTURE_NAME,
    MVP_SCENARIOS,
    MVP_SCENARIO_EXECUTOR_MAP,
)

from geoagent.layers.layer1_input import (
    InputSource,
    UserInput,
    InputValidator,
    InputParser,
    get_parser,
    parse_user_input,
)

from geoagent.layers.layer2_intent import (
    INTENT_KEYWORDS,
    IntentResult,
    IntentClassifier,
    get_classifier,
    classify_intent,
    classify_intent_simple,
)

from geoagent.layers.layer3_orchestrate import (
    ClarificationQuestion,
    ClarificationResult,
    OrchestrationResult,
    ParameterExtractor,
    ClarificationEngine,
    ScenarioOrchestrator,
    get_orchestrator,
    orchestrate,
)

from geoagent.layers.layer4_dsl import (
    OutputSpec,
    GeoDSL,
    SCHEMA_REQUIRED_PARAMS,
    SchemaValidationError,
    SchemaValidator,
    DSLBuilder,
    get_validator,
    get_builder,
    validate_dsl,
    build_dsl,
)

from geoagent.layers.layer5_executor import (
    ExecutorResult,
    SCENARIO_EXECUTOR_MAP,
    TaskRouter,
    get_router,
    execute_task,
    execute_route,
    execute_buffer,
    execute_overlay,
    execute_interpolation,
    execute_viewshed,
    execute_statistics,
    execute_raster,
)

from geoagent.layers.pipeline import (
    PipelineResult,
    PipelineConfig,
    SixLayerPipeline,
    get_pipeline,
    run_pipeline,
    # P2: 状态机模式
    StateMachinePipeline,
    StateMachineResult,
    State,
    StateContext,
)

from geoagent.layers.async_pipeline import (
    AsyncSixLayerPipeline,
    AsyncPipelineConfig,
    AsyncPipelineResult,
    AsyncLLMClient,
    ConcurrentTaskRunner,
    create_async_pipeline,
)

from geoagent.layers.llm_router import (
    RouteDecision,
    LLMJudgement,
    LLMRouter,
    get_llm_router,
    llm_route,
    llm_judge,
)

__all__ = [
    # architecture
    "Scenario",
    "RouteSubType",
    "BufferSubType",
    "OverlaySubType",
    "InterpolationSubType",
    "ViewshedSubType",
    "StatisticsSubType",
    "PipelineStatus",
    "Engine",
    "ARCHITECTURE_VERSION",
    "ARCHITECTURE_NAME",
    "MVP_SCENARIOS",
    "MVP_SCENARIO_EXECUTOR_MAP",
    # layer 1
    "InputSource",
    "UserInput",
    "InputValidator",
    "InputParser",
    "get_parser",
    "parse_user_input",
    # layer 2
    "INTENT_KEYWORDS",
    "IntentResult",
    "IntentClassifier",
    "get_classifier",
    "classify_intent",
    "classify_intent_simple",
    # layer 3
    "ClarificationQuestion",
    "ClarificationResult",
    "OrchestrationResult",
    "ParameterExtractor",
    "ClarificationEngine",
    "ScenarioOrchestrator",
    "get_orchestrator",
    "orchestrate",
    # layer 4
    "OutputSpec",
    "GeoDSL",
    "SCHEMA_REQUIRED_PARAMS",
    "SchemaValidationError",
    "SchemaValidator",
    "DSLBuilder",
    "get_validator",
    "get_builder",
    "validate_dsl",
    "build_dsl",
    # layer 5
    "ExecutorResult",
    "SCENARIO_EXECUTOR_MAP",
    "TaskRouter",
    "get_router",
    "execute_task",
    "execute_route",
    "execute_buffer",
    "execute_overlay",
    "execute_interpolation",
    "execute_viewshed",
    "execute_statistics",
    "execute_raster",
    # pipeline
    "PipelineResult",
    "PipelineConfig",
    "SixLayerPipeline",
    "get_pipeline",
    "run_pipeline",
    # P2: 状态机模式
    "StateMachinePipeline",
    "StateMachineResult",
    "State",
    "StateContext",
    # P2: 异步 Pipeline
    "AsyncSixLayerPipeline",
    "AsyncPipelineConfig",
    "AsyncPipelineResult",
    "AsyncLLMClient",
    "ConcurrentTaskRunner",
    "create_async_pipeline",
    # LLM 路由
    "RouteDecision",
    "LLMJudgement",
    "LLMRouter",
    "get_llm_router",
    "llm_route",
    "llm_judge",
]
