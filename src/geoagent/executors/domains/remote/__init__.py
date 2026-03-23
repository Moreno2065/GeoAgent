"""
遥感分析域 (Remote Sensing Domain)
===================================
包含遥感影像分析 Executor。

模块：
  - ndvi_executor.py               : NDVI/NDWI 指数
  - remote_sensing_executor.py      : 综合遥感分析
"""

from geoagent.executors.domains.remote.ndvi_executor import NdviExecutor
from geoagent.executors.domains.remote.remote_sensing_executor import (
    RemoteSensingExecutor,
    RemoteSensingIndex,
    BandMapping,
)

__all__ = [
    "NdviExecutor",
    "RemoteSensingExecutor",
    "RemoteSensingIndex",
    "BandMapping",
]
