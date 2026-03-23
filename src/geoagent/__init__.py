"""
GeoAgent 七层空间Agent架构
========================

## 架构概述

GeoAgent采用七层架构设计，将空间智能能力系统化整合：

```
┌─────────────────────────────────────────────────────────────┐
│                    空间Agent 七层架构                        │
├─────────────────────────────────────────────────────────────┤
│  第1层：用户交互层     - 自然语言、多模态输入                  │
│  第2层：意图理解层     - 30+场景分类、实体识别                 │
│  第3层：知识融合层     - RAG检索、最佳实践                    │
│  第4层：任务规划层     - 工作流编排、依赖管理                   │
│  第5层：执行引擎层     - 矢量/栅格/遥感/网络                   │
│  第6层：验证安全层     - 防幻觉、CRS/OOM检查                  │
│  第7层：结果呈现层     - 地图、图表、自然语言                   │
└─────────────────────────────────────────────────────────────┘
```

## 核心能力

### 矢量分析 (VectorPro)
- 缓冲区分析
- 空间叠置
- 空间连接
- 融合/简化

### 栅格处理 (RasterLab)
- 裁剪/重投影/重采样
- 坡度坡向
- 分区统计

### 遥感智能 (SenseAI)
- NDVI/NDWI/EVI
- 变化检测
- STAC搜索

### 三维分析 (LiDAR3D)
- 视域分析
- 阴影分析
- 体积计算
- 流域分割

### 云端遥感 (CloudRS)
- COG读取
- 多景镶嵌

## 黄金规则

1. **CRS铁律**：任何叠置分析前必须检查CRS
2. **OOM防御**：大TIFF必须使用Window分块读取
3. **防幻觉**：不捏造文件/数据/坐标
"""

from __future__ import annotations

# 版本信息
__version__ = "2.0.0"
__author__ = "GeoAgent Team"

# 架构常量
ARCHITECTURE_VERSION = "2.0"
ARCHITECTURE_NAME = "Seven-Layer Spatial Agent Architecture"

# 导出主要模块
from geoagent.layers.architecture import (
    Scenario,
    PipelineStatus,
    SpatialOperation,
    Engine,
    ARCHITECTURE_VERSION,
    ARCHITECTURE_NAME,
)

from geoagent.pipeline import (
    GeoAgentPipeline,
    PipelineResult,
    PipelineContext,
)

from geoagent.executors.base import (
    BaseExecutor,
    ExecutorResult,
)

from geoagent.gis_tools.geotoolbox import (
    GeoToolbox,
    get_toolbox,
)

from geoagent.system_prompts import (
    GIS_EXPERT_SYSTEM_PROMPT_V2,
    ANTI_HALLUCINATION_SYSTEM_PROMPT,
    CRS_SPECIFICATION_PROMPT,
    OOM_DEFENSE_PROMPT,
    REMOTE_SENSING_PROMPT,
)

# 执行器 (从功能域导入)
from geoagent.executors import (
    RouteExecutor,
    BufferExecutor,
    OverlayExecutor,
    IDWExecutor,
    HotspotExecutor,
    SuitabilityExecutor,
    ShadowExecutor,
    LiDAR3DExecutor,
    calculate_sun_position,
    AmapExecutor,
    OverpassExecutor,
    OSMExecutor,
    STACSearchExecutor,
    NdviExecutor,
    RemoteSensingExecutor,
    RemoteSensingIndex,
    BandMapping,
    VisualizationExecutor,
    GeneralExecutor,
    GDALExecutor,
    PostGISExecutor,
    CodeSandboxExecutor,
    ArcGISExecutor,
    WorkflowEngine,
    MultiCriteriaSearchExecutor,
    HybridRetrieverExecutor,
)

# 核心入口类
from geoagent.core import (
    GeoAgent,
    GeoAgentV2,
    create_agent,
    create_agent_v2,
)

# 导出所有公开API
__all__ = [
    # 版本
    "__version__",
    "__author__",
    # 架构常量
    "ARCHITECTURE_VERSION",
    "ARCHITECTURE_NAME",
    # 核心类
    "Scenario",
    "PipelineStatus",
    "SpatialOperation",
    "Engine",
    "GeoAgent",
    "GeoAgentV2",
    "GeoAgentPipeline",
    "PipelineResult",
    "PipelineContext",
    "create_agent",
    "create_agent_v2",
    "BaseExecutor",
    "ExecutorResult",
    "GeoToolbox",
    "get_toolbox",
    # 提示词
    "GIS_EXPERT_SYSTEM_PROMPT_V2",
    "ANTI_HALLUCINATION_SYSTEM_PROMPT",
    "CRS_SPECIFICATION_PROMPT",
    "OOM_DEFENSE_PROMPT",
    "REMOTE_SENSING_PROMPT",
    # 执行器
    "RemoteSensingExecutor",
    "RemoteSensingIndex",
    "BandMapping",
    "LiDAR3DExecutor",
    "calculate_sun_position",
    "STACSearchExecutor",
]
