"""
GeoAgent 工具模块
统一导出工具执行器
"""

from geoagent.tools.registry import execute_tool, execute_task, execute_task_from_dict
from geoagent.tools.tool_rag import (
    retrieve_gis_tools,
    get_retrieved_tool_schemas,
    format_retrieval_context,
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
]
