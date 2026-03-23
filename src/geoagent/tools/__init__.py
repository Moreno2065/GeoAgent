"""
GeoAgent 工具模块
统一导出工具执行器
"""

from geoagent.tools.registry import (
    execute_tool, execute_task, execute_task_from_dict,
    # 工具分类
    CORE_TOOLS, TOOL_CLUSTERS,
    get_tools_for_cluster, get_all_tools_in_clusters,
    get_all_registered_tools, is_core_tool, get_tool_cluster,
)
from geoagent.tools.tool_rag import (
    retrieve_gis_tools,
    get_retrieved_tool_schemas,
    format_retrieval_context,
)
from geoagent.tools.embedding_router import (
    EmbeddingRouter,
    EmbeddingProvider,
    EmbeddingCache,
    ConfidenceLevel,
    RoutingMatch,
    RoutingResult,
    get_embedding_router,
    clear_embedding_router,
)

__all__ = [
    # 旧版执行器
    "execute_tool",
    # 新版任务执行器
    "execute_task",
    "execute_task_from_dict",
    # 工具 RAG
    "retrieve_gis_tools",
    "get_retrieved_tool_schemas",
    "format_retrieval_context",
    # 工具分类
    "CORE_TOOLS",
    "TOOL_CLUSTERS",
    "get_tools_for_cluster",
    "get_all_tools_in_clusters",
    "get_all_registered_tools",
    "is_core_tool",
    "get_tool_cluster",
    # Embedding 语义路由
    "EmbeddingRouter",
    "EmbeddingProvider",
    "EmbeddingCache",
    "ConfidenceLevel",
    "RoutingMatch",
    "RoutingResult",
    "get_embedding_router",
    "clear_embedding_router",
]
