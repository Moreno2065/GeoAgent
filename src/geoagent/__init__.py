"""
GeoAgent - 基于 DeepSeek API 的空间智能 GIS 分析 Agent
================================================================
三层收敛架构：用户输入 → 意图分类 → 动态Schema → Pydantic校验 → 确定性执行

新增 GeoEngine 统一执行系统：
    GeoEngine
    ├── vector_engine   (GeoPandas)
    ├── raster_engine   (Rasterio)
    ├── network_engine  (NetworkX / OSMnx)
    ├── analysis_engine (SciPy / PySAL)
    └── io_engine       (Fiona / STAC)

核心设计原则：
    1. LLM 不直接调用 geopandas，只调用 GeoEngine
    2. 所有模块间统一格式：矢量→GeoDataFrame，栅格→xarray/rasterio，输出→GeoJSON
    3. 每个 task = 一个 executor，无 ReAct 循环
    4. 确定性执行，后端代码路由

标准用法:
    from geoagent import GeoAgent, create_agent
    agent = create_agent(api_key="sk-...")

    # 三层收敛编译器（推荐）
    result = agent.compile("芜湖南站到方特的步行路径")

    # 直接使用 GeoEngine
    from geoagent import GeoEngine
    engine = GeoEngine()
    result = engine.execute({
        "task": "route",
        "type": "shortest_path",
        "inputs": {"start": "芜湖南站", "end": "方特欢乐世界"},
        "params": {"mode": "walking", "city": "芜湖"},
    })
"""

from geoagent.version import __version__

# 核心
from geoagent.core import GeoAgent, create_agent

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
    # 核心
    "GeoAgent",
    "create_agent",
    # 编译器
    "GISCompiler",
    "IntentClassifier",
    "classify_intent",
    "execute_task",
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
    # Executor Layer（新架构）
    "BaseExecutor",
    "ExecutorResult",
    "TaskRouter",
    "executor_execute_task",
    "execute_scenario",
    "get_router",
    "SCENARIO_EXECUTOR_KEY",
    "ScenarioConfig",
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
