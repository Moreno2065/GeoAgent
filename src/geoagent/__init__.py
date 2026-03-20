"""
GeoAgent - 基于 DeepSeek API 的空间智能 GIS 分析 Agent
================================================================
标准的 Python 包结构，支持:
- pip install -e . 开发模式安装
- python -m geoagent 运行
- geoagent CLI 命令
- 导入 geoagent.core / geoagent.gis_tools 等子模块

增强功能:
- LangChain Agent 集成 (geoagent.langchain_agent)
- 增强版 System Prompt (geoagent.system_prompts)
- 综合技术知识库 (2万+字)
- 高级 GIS/RS 工具集 (geoagent.gis_tools.advanced_tools)
"""

from geoagent.version import __version__

# 公开 API
from geoagent.core import GeoAgent, create_agent, GIS_EXPERT_SYSTEM_PROMPT

# LangGraph Workflow（可选，优雅降级）
try:
    from geoagent.workflow import GeoAgentWorkflow
    LANGGRAPH_AVAILABLE = True
except ImportError:
    GeoAgentWorkflow = None
    LANGGRAPH_AVAILABLE = False

# LangChain Agent（可选，优雅降级）
try:
    from geoagent.langchain_agent import (
        GISReActAgent,
        create_gis_react_agent,
        AgentConfig,
        AgentResponse,
        LANGCHAIN_AVAILABLE,
    )
except ImportError:
    GISReActAgent = None
    create_gis_react_agent = None
    AgentConfig = None
    AgentResponse = None
    LANGCHAIN_AVAILABLE = False

# System Prompts
from geoagent.system_prompts import (
    GIS_EXPERT_SYSTEM_PROMPT_V2,
    GIS_EXPERT_MINIMAL_PROMPT,
    LANGCHAIN_GIS_PROMPT,
    RAG_GIS_PROMPT,
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
    "GIS_EXPERT_SYSTEM_PROMPT",
    # LangGraph Workflow
    "GeoAgentWorkflow",
    "LANGGRAPH_AVAILABLE",
    # LangChain
    "GISReActAgent",
    "create_gis_react_agent",
    "AgentConfig",
    "AgentResponse",
    "LANGCHAIN_AVAILABLE",
    # System Prompt
    "GIS_EXPERT_SYSTEM_PROMPT_V2",
    "GIS_EXPERT_MINIMAL_PROMPT",
    "LANGCHAIN_GIS_PROMPT",
    "RAG_GIS_PROMPT",
    # 知识库
    "GISKnowledgeBase",
    "get_knowledge_base",
    "search_gis_knowledge",
]
