"""
GIS 任务 Schema 定义
====================
所有 GIS 分析任务的 Pydantic 模型定义，用于：
1. 约束 LLM 输出参数
2. Pydantic 强制校验
3. 确定性任务执行路由
"""

from __future__ import annotations

import json
from typing import Literal, Optional, Any, Dict, List, Union, get_args, get_origin
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# =============================================================================
# 任务类型枚举
# =============================================================================

class TaskType(str, Enum):
    """支持的 GIS 任务类型枚举"""
    ROUTE = "route"
    BUFFER = "buffer"
    OVERLAY = "overlay"
    INTERPOLATION = "interpolation"
    SHADOW_ANALYSIS = "shadow_analysis"
    NDVI = "ndvi"
    HOTSPOT = "hotspot"
    VISUALIZATION = "visualization"
    ACCESSIBILITY = "accessibility"
    SUITABILITY = "suitability"
    VIEWSHED = "viewshed"
    GENERAL = "general"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


# =============================================================================
# 基础任务模型
# =============================================================================

class BaseTask(BaseModel):
    """所有任务的基类"""
    task: str = Field(description="任务类型标识符")

    def model_dump_geojson(self) -> Dict[str, Any]:
        """导出为 GeoJSON 兼容的字典"""
        return self.model_dump(exclude_none=True)


# =============================================================================
# Route 任务（路径规划）
# =============================================================================

class RouteMode(str, Enum):
    WALKING = "walking"
    DRIVING = "driving"
    TRANSIT = "transit"


class RouteTask(BaseTask):
    """
    路径规划任务

    示例：
        NL: "芜湖南站到方特欢乐世界的步行路径"
        Task: RouteTask(task="route", mode="walking", start="芜湖南站", end="方特欢乐世界", city="芜湖")
    """
    task: Literal["route"] = "route"
    mode: Literal["walking", "driving", "transit"] = Field(
        description="出行方式：walking（步行）/ driving（驾车）/ transit（公交）"
    )
    start: str = Field(description="起点地址或名称")
    end: str = Field(description="终点地址或名称")
    city: Optional[str] = Field(default=None, description="城市名称（用于辅助定位）")
    provider: Literal["amap", "osm", "auto"] = Field(
        default="auto",
        description="数据源：amap（国内高德）/ osm（海外OSM）/ auto（自动选择）"
    )

    @field_validator("start", "end")
    @classmethod
    def validate_address(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("地址不能为空且长度至少2个字符")
        return v.strip()

    @field_validator("city")
    @classmethod
    def validate_city(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return v.strip()
        return v


# =============================================================================
# Buffer 任务（缓冲区分析）
# =============================================================================

class BufferUnit(str, Enum):
    METERS = "meters"
    KILOMETERS = "kilometers"
    DEGREES = "degrees"


class BufferTask(BaseTask):
    """
    缓冲区分析任务

    示例：
        NL: "生成天安门500米缓冲区"
        Task: BufferTask(task="buffer", input_layer="tiananmen.shp", distance=500, unit="meters")
    """
    task: Literal["buffer"] = "buffer"
    input_layer: str = Field(description="输入矢量文件路径（workspace/ 相对路径）")
    distance: float = Field(gt=0, description="缓冲距离（必须大于0）")
    unit: Literal["meters", "kilometers", "degrees"] = Field(
        default="meters",
        description="距离单位：meters（米）/ kilometers（千米）/ degrees（度，仅EPSG:4326时有效"
    )
    dissolve: bool = Field(default=False, description="是否融合所有结果要素")
    cap_style: Literal["round", "square", "flat"] = Field(
        default="round",
        description="端点样式：round（圆形，默认）/ square（方形）/ flat（平头）"
    )

    @field_validator("input_layer")
    @classmethod
    def validate_input_layer(cls, v: str) -> str:
        if not v or len(v.strip()) < 1:
            raise ValueError("输入图层不能为空")
        return v.strip()

    @field_validator("distance")
    @classmethod
    def validate_distance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("缓冲距离必须大于0")
        if v > 100000:
            raise ValueError("缓冲距离不能超过100公里（防止内存溢出）")
        return v


# =============================================================================
# Overlay 任务（空间叠置分析）
# =============================================================================

class OverlayOperation(str, Enum):
    INTERSECT = "intersect"
    UNION = "union"
    CLIP = "clip"
    DIFFERENCE = "difference"
    SYMMETRIC_DIFFERENCE = "symmetric_difference"


class OverlayTask(BaseTask):
    """
    空间叠置分析任务

    示例：
        NL: "计算土地利用和洪涝区叠加分析"
        Task: OverlayTask(task="overlay", operation="intersect", layer1="landuse.shp", layer2="flood_zone.shp")
    """
    task: Literal["overlay"] = "overlay"
    operation: Literal["intersect", "union", "clip", "difference", "symmetric_difference"] = Field(
        description="叠加操作类型：intersect（交集）/ union（合并）/ clip（裁剪）/ difference（差集）/ symmetric_difference（对称差）"
    )
    layer1: str = Field(description="第一个输入图层路径")
    layer2: str = Field(description="第二个输入图层路径")
    output_file: Optional[str] = Field(default=None, description="输出文件路径")

    @field_validator("layer1", "layer2")
    @classmethod
    def validate_layer(cls, v: str) -> str:
        if not v or len(v.strip()) < 1:
            raise ValueError("图层路径不能为空")
        return v.strip()


# =============================================================================
# Interpolation 任务（空间插值）
# =============================================================================

class InterpolationMethod(str, Enum):
    IDW = "IDW"
    KRIGING = "kriging"
    NEAREST_NEIGHBOR = "nearest_neighbor"


class InterpolationTask(BaseTask):
    """
    空间插值分析任务（IDW / Kriging）

    示例：
        NL: "用IDW方法对PM2.5监测站数据进行插值分析"
        Task: InterpolationTask(task="interpolation", method="IDW", input_points="stations.csv", value_field="PM25", output_resolution=100)
    """
    task: Literal["interpolation"] = "interpolation"
    method: Literal["IDW", "kriging", "nearest_neighbor"] = Field(
        description="插值方法：IDW（反距离加权）/ kriging（克里金插值）/ nearest_neighbor（最近邻）"
    )
    input_points: str = Field(description="输入点数据文件路径（CSV/GeoJSON/Shapefile）")
    value_field: str = Field(description="用于插值的数值字段名")
    output_resolution: Optional[float] = Field(default=None, ge=1, description="输出栅格分辨率（米）")
    output_file: Optional[str] = Field(default=None, description="输出栅格文件路径")
    power: float = Field(default=2.0, ge=1, le=10, description="IDW 幂次（默认2）")

    @field_validator("value_field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        if not v or len(v.strip()) < 1:
            raise ValueError("字段名不能为空")
        return v.strip()


# =============================================================================
# Shadow Analysis 任务（阴影分析）
# =============================================================================

class ShadowTask(BaseTask):
    """
    建筑物阴影分析任务

    示例：
        NL: "分析2026年3月21日下午3点的建筑物阴影"
        Task: ShadowTask(task="shadow_analysis", buildings="buildings.shp", time="2026-03-21T15:00")
    """
    task: Literal["shadow_analysis"] = "shadow_analysis"
    buildings: str = Field(description="建筑物矢量文件路径（必须包含高度字段）")
    time: str = Field(description="分析时间（ISO8601格式，如 2026-03-21T15:00）")
    sun_angle: Optional[float] = Field(default=None, ge=0, le=90, description="太阳高度角（度），不填则自动计算")
    azimuth: Optional[float] = Field(default=None, ge=0, lt=360, description="太阳方位角（度，从正北顺时针），不填则自动计算")
    output_file: Optional[str] = Field(default=None, description="输出阴影多边形文件路径")

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        from datetime import datetime
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"时间格式无效，请使用 ISO8601 格式，如 2026-03-21T15:00，当前值：{v}")
        return v


# =============================================================================
# NDVI 任务（植被指数计算）
# =============================================================================

class NdviTask(BaseTask):
    """
    NDVI / 植被指数计算任务

    示例：
        NL: "计算北京市的NDVI植被指数"
        Task: NdviTask(task="ndvi", input_file="landsat.tif", sensor="landsat8", output_file="workspace/ndvi.tif")
    """
    task: Literal["ndvi"] = "ndvi"
    input_file: str = Field(description="输入遥感影像文件路径（GeoTIFF）")
    sensor: Literal["sentinel2", "landsat8", "landsat9", "auto"] = Field(
        default="auto",
        description="传感器类型：sentinel2（Sentinel-2）/ landsat8（Landsat 8）/ landsat9（Landsat 9）/ auto（自动检测）"
    )
    output_file: Optional[str] = Field(default=None, description="输出NDVI文件路径")
    band_math_expr: Optional[str] = Field(
        default=None,
        description="自定义波段表达式（不填则使用标准公式）"
    )

    @field_validator("input_file")
    @classmethod
    def validate_input_file(cls, v: str) -> str:
        if not v:
            raise ValueError("输入文件路径不能为空")
        return v.strip()


# =============================================================================
# Hotspot 任务（热点分析）
# =============================================================================

class HotspotNeighborStrategy(str, Enum):
    QUEEN = "queen"
    ROOK = "rook"
    KNN = "knn"


class HotspotTask(BaseTask):
    """
    空间热点分析任务（Getis-Ord Gi* / Moran's I）

    示例：
        NL: "分析深圳各区的房价热点"
        Task: HotspotTask(task="hotspot", input_file="districts.shp", value_field="price", analysis_type="gstar")
    """
    task: Literal["hotspot"] = "hotspot"
    input_file: str = Field(description="输入矢量面文件路径")
    value_field: str = Field(description="分析字段名（数值型）")
    analysis_type: Literal["auto", "gstar", "moran"] = Field(
        default="auto",
        description="分析类型：auto（自动选择）/ gstar（Getis-Ord Gi*热点分析）/ moran（Moran's I全局自相关）"
    )
    neighbor_strategy: Literal["queen", "rook", "knn"] = Field(
        default="queen",
        description="邻域策略：queen（后后邻域，共享顶点或边）/ rook（车相邻域，仅共享边）/ knn（K近邻）"
    )
    k_neighbors: int = Field(default=8, ge=4, le=30, description="K近邻数量（knn策略时使用）")
    distance_band: Optional[float] = Field(default=None, gt=0, description="空间距离阈值（米）")
    output_file: Optional[str] = Field(default=None, description="输出结果文件路径")

    @field_validator("value_field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        if not v or len(v.strip()) < 1:
            raise ValueError("字段名不能为空")
        return v.strip()


# =============================================================================
# Visualization 任务（可视化）
# =============================================================================

class VisualizationType(str, Enum):
    INTERACTIVE_MAP = "interactive_map"
    STATIC_PLOT = "static_plot"
    RASTER_PLOT = "raster_plot"
    MULTI_LAYER = "multi_layer"
    HEATMAP = "heatmap"


class VisualizationTask(BaseTask):
    """
    地图可视化任务

    示例：
        NL: "生成芜湖市建筑物3D可视化地图"
        Task: VisualizationTask(task="visualization", viz_type="interactive_map", input_files=["buildings.shp"], output_file="workspace/3d_map.html")
    """
    task: Literal["visualization"] = "visualization"
    viz_type: Literal["interactive_map", "static_plot", "raster_plot", "multi_layer", "heatmap"] = Field(
        description="可视化类型：interactive_map（交互式地图）/ static_plot（静态专题图）/ raster_plot（栅格渲染）/ multi_layer（多图层）/ heatmap（热力图）"
    )
    input_files: list[str] = Field(description="输入文件路径列表")
    output_file: Optional[str] = Field(default=None, description="输出文件路径")
    height_column: Optional[str] = Field(default=None, description="3D高度字段名")
    color_column: Optional[str] = Field(default=None, description="着色字段名")
    map_style: Literal["dark", "light", "road", "satellite"] = Field(
        default="road",
        description="底图样式：dark（深色）/ light（浅色）/ road（街道）/ satellite（卫星）"
    )
    layer_type: Literal["column", "hexagon", "heatmap", "scatterplot"] = Field(
        default="column",
        description="3D图层类型：column（3D柱）/ hexagon（六边形）/ heatmap（热力）/ scatterplot（散点）"
    )

    @field_validator("input_files")
    @classmethod
    def validate_input_files(cls, v: list[str]) -> list[str]:
        if not v or len(v) < 1:
            raise ValueError("至少需要一个输入文件")
        return [f.strip() for f in v if f.strip()]


# =============================================================================
# General 任务（通用任务）
# =============================================================================

class GeneralTask(BaseTask):
    """
    通用 GIS 任务（无法归类时使用）

    用于处理复杂、多步骤或非结构化的 GIS 分析需求。
    LLM 将生成详细的任务描述供后端解析。
    """
    task: Literal["general"] = "general"
    description: str = Field(description="任务描述")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="参数字典")

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if not v or len(v.strip()) < 5:
            raise ValueError("任务描述至少需要5个字符")
        return v.strip()


# =============================================================================
# Accessibility 任务（可达性分析）
# =============================================================================

class AccessibilityMode(str, Enum):
    WALKING = "walking"
    DRIVING = "driving"
    CYCLING = "cycling"


class AccessibilityTask(BaseTask):
    """
    可达性分析任务（等时圈/服务范围）

    示例：
        NL: "分析芜湖南站步行15分钟的可达范围"
        Task: AccessibilityTask(task="accessibility", location="芜湖南站", mode="walking", time_threshold=15)
    """
    task: Literal["accessibility"] = "accessibility"
    location: str = Field(description="中心位置（地址或坐标）")
    mode: Literal["walking", "driving", "cycling"] = Field(
        default="walking",
        description="交通方式：walking（步行）/ driving（驾车）/ cycling（骑行）"
    )
    time_threshold: float = Field(
        default=15,
        gt=0, le=120,
        description="时间阈值（分钟）"
    )
    grid_resolution: float = Field(
        default=50,
        gt=0, le=500,
        description="网格分辨率（米），用于可达性栅格计算"
    )
    output_file: Optional[str] = Field(default=None, description="输出文件路径")

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("位置不能为空且长度至少2个字符")
        return v.strip()


# =============================================================================
# Suitability 任务（选址分析）
# =============================================================================

class SuitabilityMethod(str, Enum):
    WEIGHTED_SUM = "weighted_sum"
    MCDM = "mcdm"
    FUZZY = "fuzzy"


class SuitabilityTask(BaseTask):
    """
    选址/适宜性分析任务（多准则决策分析）

    示例：
        NL: "在学校周边500米内找适合开便利店的位置"
        Task: SuitabilityTask(task="suitability", criteria_layers=["school_buffer.shp", "road.shp"], area="study_area.shp")
    """
    task: Literal["suitability"] = "suitability"
    criteria_layers: List[str] = Field(
        description="参与分析的图层路径列表"
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="各图层的权重，键为图层名或字段名，值为权重（0-1，总和为1）"
    )
    area: str = Field(description="分析区域边界图层路径")
    method: Literal["weighted_sum", "mcdm", "fuzzy"] = Field(
        default="weighted_sum",
        description="分析方法：weighted_sum（加权求和）/ mcdm（多准则决策）/ fuzzy（模糊综合评价）"
    )
    output_file: Optional[str] = Field(default=None, description="输出文件路径")
    top_n: int = Field(
        default=5,
        ge=1, le=100,
        description="返回最适宜的前 N 个位置"
    )

    @field_validator("criteria_layers")
    @classmethod
    def validate_criteria_layers(cls, v: List[str]) -> List[str]:
        if not v or len(v) < 2:
            raise ValueError("选址分析至少需要2个参与图层")
        return [layer.strip() for layer in v if layer.strip()]


# =============================================================================
# Viewshed 任务（视域分析）
# =============================================================================

class ViewshedTask(BaseTask):
    """
    视域分析任务（可见性/通视分析）

    示例：
        NL: "分析从这个观景点能看到哪些区域"
        Task: ViewshedTask(task="viewshed", location="118.45,31.35", observer_height=50, dem_file="dem.tif")
    """
    task: Literal["viewshed"] = "viewshed"
    location: str = Field(description="观察点位置（坐标或地址）")
    observer_height: float = Field(
        default=1.7,
        gt=0,
        description="观察点高度（米），默认1.7米（人眼高度）"
    )
    target_height: float = Field(
        default=0,
        ge=0,
        description="目标物高度（米），默认0表示地面"
    )
    max_distance: float = Field(
        default=5000,
        gt=0, le=50000,
        description="最大可视距离（米）"
    )
    dem_file: str = Field(description="DEM 高程数据文件路径")
    output_file: Optional[str] = Field(default=None, description="输出文件路径")

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("位置不能为空")
        return v.strip()

    @field_validator("dem_file")
    @classmethod
    def validate_dem_file(cls, v: str) -> str:
        if not v or len(v.strip()) < 1:
            raise ValueError("DEM 文件路径不能为空")
        return v.strip()


# =============================================================================
# 联合模型（任务解析入口）
# =============================================================================

# 所有任务模型的联合类型
TaskModel = Union[
    RouteTask,
    BufferTask,
    OverlayTask,
    InterpolationTask,
    ShadowTask,
    NdviTask,
    HotspotTask,
    VisualizationTask,
    AccessibilityTask,
    SuitabilityTask,
    ViewshedTask,
    GeneralTask,
]

# 任务模型映射表（用于动态路由）
TASK_MODEL_MAP: Dict[str, type[BaseModel]] = {
    "route": RouteTask,
    "buffer": BufferTask,
    "overlay": OverlayTask,
    "interpolation": InterpolationTask,
    "shadow_analysis": ShadowTask,
    "ndvi": NdviTask,
    "hotspot": HotspotTask,
    "visualization": VisualizationTask,
    "accessibility": AccessibilityTask,
    "suitability": SuitabilityTask,
    "viewshed": ViewshedTask,
    "general": GeneralTask,
}


# =============================================================================
# Schema 导出（供 LLM 调用）
# =============================================================================

def get_task_schema_json(intent: str) -> Dict[str, Any]:
    """获取指定意图的 JSON Schema（用于 LLM function calling）"""
    model_cls = TASK_MODEL_MAP.get(intent)
    if model_cls is None:
        return {}
    return model_cls.model_json_schema()


def get_all_task_schemas() -> Dict[str, Dict[str, Any]]:
    """获取所有任务类型的 Schema"""
    return {
        intent: model_cls.model_json_schema()
        for intent, model_cls in TASK_MODEL_MAP.items()
    }


def get_task_description(intent: str) -> str:
    """获取任务类型的中文描述"""
    descriptions = {
        "route": "路径规划：计算两点之间的最短路径（步行/驾车/公交）",
        "buffer": "缓冲区分析：创建点/线/面的指定距离缓冲区",
        "overlay": "空间叠置：执行交集/并集/差集等叠加分析",
        "interpolation": "空间插值：基于离散点生成连续表面（IDW/Kriging）",
        "shadow_analysis": "阴影分析：计算建筑物在特定时间的阴影投射",
        "ndvi": "植被指数：计算NDVI/NDWI等遥感指数",
        "hotspot": "热点分析：Getis-Ord Gi* / Moran's I 空间自相关",
        "visualization": "地图可视化：生成交互式/静态/3D地图",
        "accessibility": "可达性分析：计算等时圈/服务范围/步行可达性",
        "suitability": "选址分析：多准则决策分析（MCDA）确定最佳位置",
        "viewshed": "视域分析：计算观察点的可见范围/通视分析",
        "general": "通用任务：复杂或多步骤的 GIS 分析",
    }
    return descriptions.get(intent, "未知任务类型")


# =============================================================================
# 任务解析入口
# =============================================================================

def parse_task_from_dict(data: Dict[str, Any]) -> BaseTask:
    """
    从字典解析任务模型

    Args:
        data: 包含 task 字段的字典

    Returns:
        对应的任务模型实例

    Raises:
        ValueError: 解析或校验失败
    """
    if not isinstance(data, dict):
        raise ValueError("输入必须是字典类型")

    task_type = data.get("task", "")
    if not task_type:
        raise ValueError("缺少 'task' 字段")

    model_cls = TASK_MODEL_MAP.get(task_type)
    if model_cls is None:
        raise ValueError(f"不支持的任务类型: {task_type}")

    try:
        return model_cls(**data)
    except Exception as e:
        raise ValueError(f"任务解析失败 [{task_type}]: {str(e)}")


def parse_task_from_json(json_str: str, intent_hint: Optional[str] = None) -> BaseTask:
    """
    从 JSON 字符串解析任务模型

    Args:
        json_str: JSON 字符串
        intent_hint: 可选的任务类型提示

    Returns:
        对应的任务模型实例
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {str(e)}")

    # 如果有 intent_hint，先验证 task 字段
    if intent_hint and isinstance(data, dict):
        if data.get("task") != intent_hint:
            data["task"] = intent_hint

    return parse_task_from_dict(data)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 枚举
    "TaskType",
    "RouteMode",
    "BufferUnit",
    "OverlayOperation",
    "InterpolationMethod",
    "HotspotNeighborStrategy",
    "VisualizationType",
    # 基础模型
    "BaseTask",
    # 任务模型
    "RouteTask",
    "BufferTask",
    "OverlayTask",
    "InterpolationTask",
    "ShadowTask",
    "NdviTask",
    "HotspotTask",
    "VisualizationTask",
    "GeneralTask",
    # 联合类型
    "TaskModel",
    # 工具函数
    "TASK_MODEL_MAP",
    "get_task_schema_json",
    "get_all_task_schemas",
    "get_task_description",
    "parse_task_from_dict",
    "parse_task_from_json",
]
