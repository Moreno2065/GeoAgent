"""
核心/通用域 (Core Domain)
==========================
包含通用和基础设施 Executor。

模块：
  - general_executor.py     : 通用执行器
  - gdal_executor.py       : GDAL 执行器
  - postgis_executor.py    : PostGIS 执行器
  - code_sandbox_executor.py : 代码沙盒 (暂不可用，源文件编码损坏)
  - arcgis_executor.py     : ArcGIS 执行器
  - workflow_engine.py     : 工作流引擎
"""

from geoagent.executors.domains.core.general_executor import GeneralExecutor
from geoagent.executors.domains.core.gdal_executor import GDALExecutor
from geoagent.executors.domains.core.postgis_executor import PostGISExecutor
from geoagent.executors.domains.core.arcgis_executor import ArcGISExecutor
from geoagent.executors.domains.core.workflow_engine import WorkflowEngine

# CodeSandboxExecutor 暂不可用 (源文件编码损坏)
CodeSandboxExecutor = None

__all__ = [
    "GeneralExecutor",
    "GDALExecutor",
    "PostGISExecutor",
    "CodeSandboxExecutor",  # 暂不可用
    "ArcGISExecutor",
    "WorkflowEngine",
]
