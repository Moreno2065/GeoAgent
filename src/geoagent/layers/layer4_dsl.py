"""
第4层：任务编译层（Task DSL Builder）
======================================
核心职责：
1. 把自然语言变成结构化 GeoDSL
2. Schema 校验
3. 确保参数合法
4. 标准化任务描述

设计原则：
- 不是 Python
- 不是 ArcGIS tool 名称
- 是你的标准协议

NL→DSL 模式（可选）：
- 默认（推荐）：确定性 DSL 构建，不用 LLM
- Reasoner 模式：NL → GeoDSL（用 DeepSeek Reasoner 模型）
  仅在复杂多步骤任务时启用，如 "先做500m缓冲区再找里面的餐厅"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator

from geoagent.layers.architecture import Scenario


# =============================================================================
# GeoDSL 核心模型
# =============================================================================

class OutputSpec(BaseModel):
    """输出规格定义"""
    map: bool = Field(default=True, description="是否生成地图")
    summary: bool = Field(default=True, description="是否生成文字摘要")
    files: List[str] = Field(default_factory=list, description="输出文件路径列表")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="数值指标")
    explanation: Optional[str] = Field(default=None, description="解释卡片文本")


# =============================================================================
# GeoDSL 可视化扩展模型（v2.0 升级）
# =============================================================================

class VisualizationSpec(BaseModel):
    """
    视觉编码规范（Visual Encoding）

    定义 GeoJSON 数据的渲染样式，实现数据和视觉的解耦。

    设计原则：
    - 数据（geometry）和样式（style）必须分离
    - 支持按字段值映射视觉变量（颜色/大小/透明度）
    - 支持固定值和字段映射两种模式

    示例：
        # 固定样式
        VisualizationSpec(color="blue", opacity=0.4, fill_color="#3388ff")

        # 字段映射样式（按 category 字段分类着色）
        VisualizationSpec(color={"field": "category", "scheme": "category10"})

        # 数值映射样式（按 population 字段值大小决定点半径）
        VisualizationSpec(size={"field": "population", "range": [5, 30]})
    """
    # ── 颜色配置 ─────────────────────────────────────────────────
    # 固定颜色（如 "blue", "#3388ff", "rgb(50,100,200)"）
    # 或字段映射（如 {"field": "category", "scheme": "category10"}）
    color: Optional[Any] = Field(
        default=None,
        description="主色调：固定值字符串 或 按字段映射字典（scheme 支持: category10, viridis, plasma）"
    )
    fill_color: Optional[Any] = Field(
        default=None,
        description="填充色：固定值 或 按字段映射"
    )
    stroke_color: Optional[Any] = Field(
        default=None,
        description="边框/线颜色"
    )

    # ── 大小配置 ─────────────────────────────────────────────────
    # 固定数值（点半径、线宽等）
    # 或按字段映射（如 {"field": "population", "range": [5, 30]}）
    size: Optional[Any] = Field(
        default=None,
        description="大小：固定数值(px) 或 按数值字段映射字典"
    )
    stroke_width: float = Field(default=1.0, description="线宽/边框宽度(px)")

    # ── 透明度配置 ───────────────────────────────────────────────
    opacity: float = Field(default=0.7, ge=0.0, le=1.0, description="整体透明度")
    fill_opacity: float = Field(default=0.6, ge=0.0, le=1.0, description="填充透明度")

    # ── 标签配置 ─────────────────────────────────────────────────
    label: Optional[str] = Field(
        default=None,
        description="显示的标签字段名（如 'name', 'population'）"
    )
    label_size: int = Field(default=12, description="标签字体大小(px)")
    label_color: str = Field(default="#333333", description="标签颜色")

    # ── 符号配置 ─────────────────────────────────────────────────
    # point 图层的符号类型：circle / square / triangle / star
    symbol: str = Field(default="circle", description="点符号类型")

    # ── 专题地图类型 ─────────────────────────────────────────────
    # 是否渲染为热力图
    heatmap: bool = Field(default=False, description="是否渲染为热力图（HeatMap）")
    # 热力图专用：权重字段
    heatmap_weight_field: Optional[str] = Field(
        default=None,
        description="热力图权重字段（数值型，用于热力强度）"
    )
    # 是否渲染为分级设色图
    choropleth: Optional[str] = Field(
        default=None,
        description="分级设色字段名（填字段名则启用 Choropleth，按该字段值分类着色）"
    )
    # 分级设色参数
    choropleth_scheme: str = Field(
        default="quantiles",
        description="分级方法：quantiles / equal /jenks"
    )
    choropleth_classes: int = Field(default=5, description="分级数量（默认5级）")

    # ── 字段类型推断辅助 ─────────────────────────────────────────
    model_config = ConfigDict(use_enum_values=True)


class ViewSpec(BaseModel):
    """
    视图控制规范（View Control）

    控制地图的初始视图状态：中心点、缩放级别、旋转、倾斜。

    示例：
        ViewSpec(fit_bounds=True)  # 自动适应所有图层边界
        ViewSpec(center=[31.3, 118.3], zoom=12, pitch=45, bearing=0)  # 手动指定
    """
    # ── 边界适应 ─────────────────────────────────────────────────
    fit_bounds: bool = Field(
        default=True,
        description="自动适应所有图层边界（推荐开启，会覆盖 center/zoom）"
    )
    # fit_bounds 为 True 时可额外指定 padding
    bounds_padding: Tuple[float, float, float, float] = Field(
        default=(50, 50, 50, 50),
        description="边界适应的内边距 (top, right, bottom, left)，单位px"
    )

    # ── 手动指定视图 ─────────────────────────────────────────────
    center: Optional[Tuple[float, float]] = Field(
        default=None,
        description="地图中心点 [lat, lon]"
    )
    zoom: int = Field(default=12, ge=1, le=20, description="缩放级别")
    # ── 3D 视图 ─────────────────────────────────────────────────
    pitch: int = Field(default=0, ge=0, le=60, description="俯仰角（3D模式），0=垂直向下")
    bearing: int = Field(default=0, ge=0, le=360, description="方位角（旋转角度）")

    # ── 地图限制 ─────────────────────────────────────────────────
    min_zoom: int = Field(default=3, description="最小缩放级别")
    max_zoom: int = Field(default=18, description="最大缩放级别")


class LayerSpec(BaseModel):
    """
    图层规范（Layer Specification）

    描述单个图层的完整配置：ID、类型、数据源、样式。

    设计原则：
    - 每个图层都有唯一 layer_id（用于交互控制：点击/隐藏/切换）
    - style 字段和 data 字段完全分离
    - 所有输出必须是 GeoJSON（系统内部中间语言）

    示例：
        LayerSpec(
            layer_id="buffer_1",
            layer_type="buffer",
            source="tmp_road_buf",  # 引用中间变量 或 文件路径
            style=VisualizationSpec(color="blue", opacity=0.4),
            visible=True,
            interactive=True,
        )
    """
    layer_id: str = Field(description="唯一图层ID，用于图层控制和交互")
    layer_type: str = Field(
        description="图层类型：buffer / poi / heatmap / choropleth / raw / route / overlay"
    )
    # 数据源：文件路径 或 tmp_xxx 中间变量引用
    source: str = Field(
        description="数据源（文件路径 或 WorkflowStep 的 output_id）"
    )
    # ── 可视化样式 ───────────────────────────────────────────────
    style: Optional[VisualizationSpec] = Field(
        default=None,
        description="图层视觉样式（可选，不提供则使用默认样式）"
    )
    # ── 显示控制 ─────────────────────────────────────────────────
    visible: bool = Field(default=True, description="是否显示")
    # ── 交互控制 ─────────────────────────────────────────────────
    interactive: bool = Field(
        default=True,
        description="是否可点击（True=弹出信息框，False=仅显示）"
    )
    # ── 图层名称 ─────────────────────────────────────────────────
    name: Optional[str] = Field(
        default=None,
        description="图层显示名称（用于图层控制面板）"
    )

    model_config = ConfigDict(use_enum_values=True)


class WorkflowStep(BaseModel):
    """
    工作流中的单一步骤

    用于描述多步骤链式任务中的一个步骤，支持：
    - 中间变量引用（引用前序步骤的输出）
    - 显式依赖声明
    - 任务参数传递
    - 条件执行（可选）
    - 循环执行（可选）

    示例：
        {
            "step_id": "step_1",
            "task": "buffer",
            "description": "对道路图层做100米缓冲",
            "inputs": {"layer": "道路.shp", "distance": 100, "unit": "meters"},
            "output_id": "tmp_buffer_1",
            "depends_on": [],
        }
        {
            "step_id": "step_2",
            "task": "overlay",
            "description": "用河流缓冲擦除道路缓冲，得到适宜区域",
            "inputs": {"layer1": "tmp_buffer_1", "layer2": "tmp_river_buf", "operation": "erase"},
            "output_id": "tmp_suitable",
            "depends_on": ["step_1", "step_3"],
            "condition": "density > 0.5",  # 可选：条件满足才执行
        }
    """
    step_id: str = Field(description="唯一步骤ID，如 'step_1', 'buffer_step'")
    task: str = Field(description="任务类型：buffer/overlay/select/io_proj/stats/proximity")
    description: str = Field(default="", description="步骤的自然语言描述（用于日志）")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="输入参数，支持两种来源：文件路径 或 引用其他步骤的输出（tmp_xxx）"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="分析参数（如 distance, operation, unit 等）"
    )
    output_id: str = Field(
        description="本步骤输出的中间变量名，如 'tmp_buffer_1'。后续步骤通过此名称引用。"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="依赖的步骤ID列表，用于拓扑排序"
    )
    output_file: Optional[str] = Field(
        default=None,
        description="输出文件路径（如果需要持久化到磁盘）"
    )
    # ── 条件执行（Phase 4.1 扩展）────────────────────────────────
    condition: Optional[str] = Field(
        default=None,
        description="条件表达式，满足条件才执行。格式：'field > 0.5' 或 'count >= 10'"
    )
    # ── 循环执行（Phase 4.1 扩展）────────────────────────────────
    for_each: Optional[Dict[str, Any]] = Field(
        default=None,
        description="循环执行配置。格式：{'var': 'layer', 'in': 'tmp_layers'}"
    )

    model_config = ConfigDict(use_enum_values=True)


class GeoDSL(BaseModel):
    """
    GeoDSL 统一任务描述语言

    用途：
    1. 第3层编排器将自然语言翻译为标准 DSL
    2. Validator 校验参数完整性
    3. Router 根据 scenario 分发到对应 Executor
    4. Executor 返回标准化结果

    示例（单步模式）：
        NL: "芜湖南站到方特欢乐世界的步行路径"
        GeoDSL:
            version: "1.1"
            scenario: "route"
            task: "walking_route"
            inputs:
                start: "芜湖南站"
                end: "方特欢乐世界"
            parameters:
                mode: "walking"
            outputs:
                map: true
                summary: true

    示例（工作流模式）：
        NL: "选出距离道路100m内且避开河流50m的区域"
        GeoDSL:
            version: "2.0"
            scenario: "overlay"
            task: "workflow"
            is_workflow: true
            steps:
                - step_id: "step_1"
                  task: "buffer"
                  inputs: {"layer": "道路.shp", "distance": 100, "unit": "meters"}
                  output_id: "tmp_road_buf"
                - step_id: "step_2"
                  task: "buffer"
                  inputs: {"layer": "河流.shp", "distance": 50, "unit": "meters"}
                  output_id: "tmp_river_buf"
                - step_id: "step_3"
                  task: "overlay"
                  inputs: {"layer1": "tmp_road_buf", "layer2": "tmp_river_buf", "operation": "erase"}
                  output_id: "final_result"
                  depends_on: ["step_1", "step_2"]
            final_output: "final_result"

    示例（可视化扩展 v2.1）：
        NL: "查询芜湖市的餐厅并显示成热力图"
        GeoDSL:
            version: "2.1"
            scenario: "poi_search"
            visualization:
                heatmap: true
                heatmap_weight_field: "count"
            view:
                fit_bounds: true
    """
    version: str = Field(default="1.0", description="DSL 协议版本（1.0=单步, 2.0=工作流, 2.1=可视化扩展）")
    scenario: Scenario = Field(description="场景类型")
    task: str = Field(description="具体任务类型（与 scenario 一致或更细化），工作流模式下为 'workflow'")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="输入参数（数据源、位置等），工作流模式下可为字面量或 tmp_xxx 引用"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="分析参数（方法、距离、阈值等）"
    )
    outputs: OutputSpec = Field(
        default_factory=OutputSpec,
        description="输出规格"
    )
    clarification: Optional[List[str]] = Field(
        default=None,
        description="需要追问的参数列表"
    )
    clarification_answers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="追问的答案"
    )
    # ── 工作流扩展（v2.0）─────────────────────────────────────────────
    is_workflow: bool = Field(
        default=False,
        description="是否为工作流模式（多步骤链式任务）"
    )
    steps: List[WorkflowStep] = Field(
        default_factory=list,
        description="工作流步骤列表，仅 is_workflow=True 时有效"
    )
    final_output: str = Field(
        default="final_result",
        description="最终输出变量名"
    )
    # ── 可视化扩展（v2.1）─────────────────────────────────────────────
    # 全局视觉编码配置（可被单图层 style 覆盖）
    visualization: Optional[VisualizationSpec] = Field(
        default=None,
        description="全局视觉编码配置（颜色/大小/透明度/标签等）"
    )
    # 视图控制配置（自动缩放/定位/旋转/倾斜）
    view: Optional[ViewSpec] = Field(
        default=None,
        description="视图控制配置（中心点、缩放级别、边界适应）"
    )
    # 多图层配置（覆盖 visualization 全局配置）
    layers: Optional[List[LayerSpec]] = Field(
        default=None,
        description="多图层配置列表（支持多图层叠加渲染）"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# Schema 定义（用于校验）
# =============================================================================

# 每个 Scenario 必需的参数
SCHEMA_REQUIRED_PARAMS: Dict[Scenario, Dict[str, Any]] = {
    Scenario.ROUTE: {
        "start": {"type": "string", "required": True},
        "end": {"type": "string", "required": True},
        "mode": {"type": "string", "default": "walking", "options": ["walking", "driving", "cycling", "transit"]},
    },
    Scenario.BUFFER: {
        "input_layer": {"type": "string", "required": True},
        "distance": {"type": "number", "required": True, "min": 0.1, "max": 100000},
        "unit": {"type": "string", "default": "meters", "options": ["meters", "kilometers", "degrees"]},
    },
    Scenario.OVERLAY: {
        "layer1": {"type": "string", "required": True},
        "layer2": {"type": "string", "required": True},
        "operation": {"type": "string", "default": "intersect", "options": ["intersect", "union", "clip", "difference"]},
    },
    Scenario.INTERPOLATION: {
        "input_points": {"type": "string", "required": True},
        "value_field": {"type": "string", "required": True},
        "method": {"type": "string", "default": "idw", "options": ["idw", "kriging", "nearest"]},
    },
    Scenario.VIEWSHED: {
        "location": {"type": "string", "required": True},
        "dem_file": {"type": "string", "required": True},
        "observer_height": {"type": "number", "default": 1.7, "min": 0},
    },
    Scenario.STATISTICS: {
        "input_file": {"type": "string", "required": True},
        "value_field": {"type": "string", "required": True},
    },
    Scenario.RASTER: {
        "input_file": {"type": "string", "required": True},
        "index_type": {"type": "string", "default": "ndvi", "options": ["ndvi", "ndwi", "evi"]},
    },
    # 适宜性分析/选址 (MCDA) - 多准则决策分析
    Scenario.SUITABILITY: {
        # === 分析区域 ===
        "study_area": {
            "type": "string",
            "required": True,
            "description": "研究区范围（图层文件名或区域名）",
            "examples": ["土地利用.shp", "研究区.shp", "市区范围.shp"],
        },

        # === 约束条件图层列表 ===
        # 支持多种格式：字符串(用;分隔) 或 列表
        "constraint_layers": {
            "type": "list",
            "required": True,
            "description": "参与选址分析的约束/影响因素图层",
            "examples": ["土地利用.shp;道路.shp;河流.shp;住宅小区.shp"],
        },

        # === 约束条件定义 ===
        # 支持多种条件类型：
        # - 土地利用过滤：landuse=value（如 landuse=unallocated）
        # - 距离缓冲：distance<=X 或 distance>=X
        # - 属性过滤：field=value
        "constraint_conditions": {
            "type": "dict",
            "required": True,
            "description": "约束条件字典，格式：图层名:条件",
            "examples": {
                "土地利用": "landuse=unallocated",
                "道路": "distance<=100",
                "河流": "distance>=150",
                "住宅小区": "distance>=800",
            },
        },

        # === 权重设置（可选，用于加权叠加）===
        "factor_weights": {
            "type": "dict",
            "required": False,
            "description": "各图层权重（用于加权叠加分析）",
            "default": {},
        },

        # === 设施类型 ===
        "facility_type": {
            "type": "string",
            "required": False,
            "default": "general",
            "options": ["general", "garbage", "school", "hospital", "factory", "warehouse", "park", "parking"],
            "description": "设施类型，影响默认距离阈值",
        },

        # === 设施名称 ===
        "facility_name": {
            "type": "string",
            "required": False,
            "description": "设施名称（用于结果描述和影响范围统计）",
            "examples": ["垃圾场", "学校", "医院"],
        },

        # === 影响范围统计 ===
        "impact_radius": {
            "type": "number",
            "required": False,
            "default": 2000,
            "description": "影响半径（米），用于统计受影响对象数量",
        },

        # === 输出选项 ===
        "output_file": {"type": "string", "required": False},
        "output_count": {
            "type": "integer",
            "required": False,
            "default": 2,
            "description": "输出选址结果数量（TOP N）",
        },
        "visualize": {"type": "bool", "default": True},

        # === 坐标系 ===
        "target_crs": {
            "type": "string",
            "required": False,
            "default": "EPSG:4548",  # Beijing 1954 / 3-degree GK CM 120E
            "description": "目标坐标系",
        },
    },

    # ── 🟣 代码沙盒（受限代码执行）───────────────────────────────────────
    Scenario.CODE_SANDBOX: {
        "code": {
            "type": "string",
            "required": True,
            "description": "LLM 生成的 Python 代码",
        },
        "description": {
            "type": "string",
            "required": False,
            "description": "任务描述（用于代码生成时的上下文）",
        },
        "context_data": {
            "type": "dict",
            "required": False,
            "description": "传入的数据上下文（如 workspace 文件路径、变量等）",
        },
        "language": {
            "type": "string",
            "required": False,
            "default": "python",
            "description": "编程语言（目前仅支持 python）",
        },
        "timeout_seconds": {
            "type": "number",
            "required": False,
            "default": 60.0,
            "min": 1.0,
            "max": 300.0,
            "description": "执行超时时间（秒）",
        },
        "mode": {
            "type": "string",
            "required": False,
            "default": "exec",
            "options": ["exec", "eval"],
            "description": "执行模式：exec=执行语句块，eval=计算表达式",
        },
    },

    # ── 🟣 代码沙盒（受限代码执行）────────────────────────────────
    Scenario.CODE_SANDBOX: {
        "instruction": {
            "type": "string",
            "required": True,
            "description": "用户原始指令（透传给 LLM 生成代码）",
            "examples": ["计算两个圆的交集面积", "生成随机点并计算距离"],
        },
        "timeout_seconds": {
            "type": "number",
            "required": False,
            "default": 60.0,
            "min": 1.0,
            "max": 300.0,
            "description": "执行超时时间（秒）",
        },
        "mode": {
            "type": "string",
            "required": False,
            "default": "exec",
            "options": ["exec", "interactive", "notebook"],
            "description": "执行模式",
        },
    },

    # ── 🟣 多条件综合搜索 ───────────────────────────────────────────
    Scenario.MULTI_CRITERIA_SEARCH: {
        "user_input": {
            "type": "string",
            "required": True,
            "description": "用户原始输入（用于解析多条件）",
            "examples": ["广州体育中心附近找距离星巴克小于200米且地铁站大于500米的地方"],
        },
        "center_point": {
            "type": "string",
            "required": True,
            "description": "搜索中心位置（地址或地标名）",
            "examples": ["广州体育中心", "天河城", "珠江新城"],
        },
        "search_radius": {
            "type": "number",
            "required": False,
            "default": 3000,
            "min": 500,
            "max": 10000,
            "description": "搜索半径（米）",
        },
        "city": {
            "type": "string",
            "required": False,
            "description": "城市名称（用于辅助定位同名地点）",
            "examples": ["广州", "深圳"],
        },
        "visualize": {
            "type": "boolean",
            "required": False,
            "default": True,
            "description": "是否生成可视化地图",
        },
    },

    # ── POI 周边搜索（"XX周围有多少个XX"模式）──────────────────────────────
    Scenario.POI_SEARCH: {
        "center_point": {
            "type": "string",
            "required": True,
            "description": "中心地点（地址或地标名，用于 Geocode 转经纬度）",
            "examples": ["上海静安寺", "北京天安门", "浦东陆家嘴"],
        },
        "keywords": {
            "type": "string",
            "required": True,
            "description": "POI 搜索关键词（如商家名、品牌名、设施类型）",
            "examples": ["星巴克", "餐厅", "银行", "医院", "超市", "地铁站"],
        },
        "radius": {
            "type": "number",
            "required": False,
            "default": 3000,
            "min": 100,
            "max": 5000,
            "description": "周边搜索半径（米），最大 5000 米",
        },
        "city": {
            "type": "string",
            "required": False,
            "description": "城市名称（用于辅助定位同名地点）",
            "examples": ["上海", "北京", "深圳", "杭州"],
        },
    },

    # ── 🟣 OSM 地图下载 ────────────────────────────────────────────
    Scenario.FETCH_OSM: {
        "center_point": {
            "type": "string",
            "required": True,
            "description": "中心地点（地址或地标名，用于 Geocode 转经纬度）",
            "examples": ["武汉黄鹤楼", "北京天安门", "上海外滩"],
        },
        "radius": {
            "type": "number",
            "required": False,
            "default": 1000,
            "min": 100,
            "max": 50000,
            "description": "下载半径（米）",
        },
        "data_type": {
            "type": "string",
            "required": False,
            "default": "all",
            "options": ["all", "network", "building"],
            "description": "数据类型：all=路网+建筑，network=仅路网，building=仅建筑",
        },
        "network_type": {
            "type": "string",
            "required": False,
            "default": "walk",
            "options": ["walk", "drive", "bike", "all"],
            "description": "路网类型：walk=步行网络，drive=车行网络，bike=骑行网络，all=所有道路",
        },
    },
}


# =============================================================================
# Schema 校验器
# =============================================================================

class SchemaValidationError(Exception):
    """Schema 校验错误"""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"[{field}] {message}")


class SchemaValidator:
    """
    Schema 校验器

    核心职责：
    1. 检查必填字段是否存在
    2. 检查字段类型是否正确
    3. 检查值是否在允许范围内
    4. 不允许"差不多也行"
    """

    def validate(self, scenario: Scenario, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        校验 DSL 参数

        Args:
            scenario: 场景类型
            inputs: 输入参数
            parameters: 分析参数

        Returns:
            校验通过后的完整参数字典

        Raises:
            SchemaValidationError: 校验失败
        """
        schema = SCHEMA_REQUIRED_PARAMS.get(scenario, {})
        errors: List[str] = []

        # 合并 inputs 和 parameters
        all_params = {**inputs, **parameters}

        # 检查必填字段
        for field, spec in schema.items():
            if spec.get("required", False):
                if field not in all_params or all_params[field] is None or all_params[field] == "":
                    errors.append(f"缺少必填参数: {field}")

        # 检查类型和范围
        for field, value in all_params.items():
            if field not in schema:
                continue

            spec = schema[field]
            param_type = spec.get("type", "string")

            # 类型检查
            if param_type == "number" and not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(f"[{field}] 必须是数字类型，当前值: {value}")

            # 范围检查
            if param_type == "number" and isinstance(value, (int, float)):
                if "min" in spec and value < spec["min"]:
                    errors.append(f"[{field}] 值 {value} 小于最小值 {spec['min']}")
                if "max" in spec and value > spec["max"]:
                    errors.append(f"[{field}] 值 {value} 大于最大值 {spec['max']}")

            # 选项检查
            if "options" in spec and value not in spec["options"]:
                errors.append(f"[{field}] 值 '{value}' 不在允许的选项中: {spec['options']}")

        if errors:
            raise SchemaValidationError(
                field="schema",
                message="; ".join(errors)
            )

        return all_params

    def fill_defaults(self, scenario: Scenario, params: Dict[str, Any]) -> Dict[str, Any]:
        """填充默认值"""
        schema = SCHEMA_REQUIRED_PARAMS.get(scenario, {})
        result = params.copy()

        for field, spec in schema.items():
            if field not in result or result[field] is None:
                default = spec.get("default")
                if default is not None:
                    result[field] = default

        return result


# =============================================================================
# DSL 构建器
# =============================================================================

class DSLBuilder:
    """
    DSL 构建器

    核心职责：
    1. 根据 scenario 和 extracted_params 构建 GeoDSL
    2. 填充默认值
    3. 标准化参数
    """

    def __init__(self):
        self.validator = SchemaValidator()

    def build(
        self,
        scenario: Scenario,
        extracted_params: Dict[str, Any],
        task_suffix: str = "",
    ) -> GeoDSL:
        """
        构建 GeoDSL

        Args:
            scenario: 场景类型
            extracted_params: 从自然语言提取的参数
            task_suffix: 任务后缀（如 walking_route, point_buffer 等）

        Returns:
            GeoDSL 对象

        Raises:
            SchemaValidationError: 校验失败
        """
        # 强制将 scenario 转换为枚举，确保后续 .value 和字典查找绝对安全
        if not hasattr(scenario, "value"):
            try:
                scenario = Scenario(str(scenario))
            except Exception:
                scenario = Scenario.ROUTE

        # ── POI_SEARCH 特殊处理：normalize `keyword` → `keywords` ────────────
        # LLM 提取器返回 `keyword`，Schema/PoiSearchTask 用 `keywords`
        extracted_params = dict(extracted_params)
        scenario_val = scenario.value if hasattr(scenario, "value") else str(scenario)
        if scenario_val == "poi_search":
            if extracted_params.get("keyword") and not extracted_params.get("keywords"):
                extracted_params["keywords"] = extracted_params.pop("keyword")

        # 分离 inputs 和 parameters
        inputs, parameters = self._separate_params(scenario, extracted_params)

        # 填充默认值
        parameters = self.validator.fill_defaults(scenario, parameters)

        # 校验
        self.validator.validate(scenario, inputs, parameters)

        # 构建 task 名称
        task = scenario.value
        if task_suffix:
            task = f"{scenario.value}_{task_suffix}"

        return GeoDSL(
            version="1.0",
            scenario=scenario,
            task=task,
            inputs=inputs,
            parameters=parameters,
            outputs=OutputSpec(map=True, summary=True),
        )

    def _separate_params(self, scenario: Scenario, params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """分离 inputs 和 parameters"""
        inputs_keys = {
            Scenario.ROUTE: ["start", "end"],
            Scenario.BUFFER: ["input_layer"],
            Scenario.OVERLAY: ["layer1", "layer2"],
            Scenario.INTERPOLATION: ["input_points"],
            Scenario.VIEWSHED: ["location", "dem_file"],
            Scenario.STATISTICS: ["input_file"],
            Scenario.RASTER: ["input_file"],
            Scenario.SUITABILITY: ["study_area"],
            Scenario.POI_SEARCH: ["center_point"],
            Scenario.CODE_SANDBOX: ["instruction"],  # 用户原始指令
            Scenario.FETCH_OSM: ["center_point"],  # OSM 地图下载的中心点
        }

        keys = inputs_keys.get(scenario, [])
        inputs = {k: v for k, v in params.items() if k in keys}
        parameters = {k: v for k, v in params.items() if k not in keys}

        return inputs, parameters

    def build_from_template(
        self,
        template_id: str,
        extracted_params: Dict[str, Any],
    ) -> GeoDSL:
        """
        从工作流模板构建 GeoDSL。

        Args:
            template_id: 模板 ID（如 "poi_distribution", "buffer_analysis"）
            extracted_params: 从用户输入提取的参数

        Returns:
            GeoDSL 对象

        Raises:
            ValueError: 模板不存在
        """
        from geoagent.workflow_templates import WORKFLOW_TEMPLATES, WorkflowTemplateEngine

        if template_id not in WORKFLOW_TEMPLATES:
            raise ValueError(f"模板 '{template_id}' 不存在。可用模板：{list(WORKFLOW_TEMPLATES.keys())}")

        template_data = WORKFLOW_TEMPLATES[template_id]
        scenario_str = template_data.get("scenario", "general")
        try:
            scenario = Scenario(scenario_str)
        except (ValueError, TypeError):
            scenario = Scenario.ROUTE

        # 使用模板引擎填充
        engine = WorkflowTemplateEngine()
        match = engine.match_best(extracted_params.get("user_input", ""))

        # 构建填充后的模板
        filled = engine.fill_template(
            TemplateMatch(
                template_id=template_id,
                template_name=template_data.get("name", template_id),
                confidence=1.0,
                matched_keywords=[],
                scenario=scenario,
                steps=template_data.get("steps", []),
                visualization=template_data.get("visualization", {}),
                view=template_data.get("view", {}),
            ),
            extracted_params,
        )

        # 构建 WorkflowSteps
        steps = [WorkflowStep(**s) for s in filled.get("steps", [])]

        # 构建可视化配置
        vis_data = filled.get("visualization")
        visualization = None
        if vis_data:
            from geoagent.layers.layer4_dsl import VisualizationSpec
            visualization = VisualizationSpec(**vis_data)

        view_data = filled.get("view")
        view = None
        if view_data:
            from geoagent.layers.layer4_dsl import ViewSpec
            view = ViewSpec(**view_data)

        return GeoDSL(
            version="2.1",
            scenario=scenario,
            task="workflow",
            inputs={"user_input": extracted_params.get("user_input", "")},
            is_workflow=True,
            steps=steps,
            final_output=steps[-1].output_id if steps else "final_result",
            outputs=OutputSpec(map=True, summary=True),
            visualization=visualization,
            view=view,
        )

    def build_from_orchestration(
        self,
        orchestration_result: Any,
        use_reasoner: bool = False,
        reasoner_factory: Optional[Callable[[], Any]] = None,
    ) -> GeoDSL:
        """
        从编排结果构建 DSL

        Args:
            orchestration_result: OrchestrationResult 对象
            use_reasoner: 是否使用 Reasoner 模式
            reasoner_factory: Reasoner 实例工厂

        Returns:
            GeoDSL 对象
        """
        scenario = orchestration_result.scenario
        extracted_params = orchestration_result.extracted_params
        return build_dsl(
            scenario,
            extracted_params,
            use_reasoner=use_reasoner,
            reasoner_factory=reasoner_factory,
        )


# =============================================================================
# 便捷函数
# =============================================================================

_validator: Optional[SchemaValidator] = None
_builder: Optional[DSLBuilder] = None


def get_validator() -> SchemaValidator:
    """获取校验器单例"""
    global _validator
    if _validator is None:
        _validator = SchemaValidator()
    return _validator


def get_builder() -> DSLBuilder:
    """获取 DSL 构建器单例"""
    global _builder
    if _builder is None:
        _builder = DSLBuilder()
    return _builder


def validate_dsl(scenario: Scenario, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    便捷函数：校验 DSL 参数

    Returns:
        校验通过后的完整参数字典
    """
    validator = get_validator()
    return validator.validate(scenario, inputs, parameters)


def build_dsl(
    scenario: Scenario,
    extracted_params: Dict[str, Any],
    task_suffix: str = "",
    use_reasoner: bool = False,
    reasoner_factory: Optional[Callable[[], Any]] = None,
) -> GeoDSL:
    """
    便捷函数：构建 GeoDSL

    Args:
        scenario: 场景类型
        extracted_params: 从 Layer 3 提取的参数（包含 user_input 自然语言原文）
        task_suffix: 任务后缀（如 walking_route, point_buffer 等）
        use_reasoner: 是否使用 Reasoner 模式（NL → DSL）
        reasoner_factory: Reasoner 实例工厂（用于注入 mock）

    Returns:
        GeoDSL 对象

    这是第4层的标准出口函数。
    """
    # ── Reasoner 模式 ──────────────────────────────────────────────────────
    if use_reasoner:
        reasoner = reasoner_factory() if reasoner_factory else None
        if reasoner is None:
            raise ValueError(
                "use_reasoner=True 但未提供 reasoner_factory。"
                "请传入 lambda: lambda: get_reasoner()"
            )
        user_input = extracted_params.get("user_input", "")

        # 判断是否为复杂多步骤任务
        # 如果包含 "先...再..."、"然后"、"接着"、"并且" 等多步骤关键词，使用工作流翻译
        multi_step_keywords = ["先", "再", "然后", "接着", "并且", "而且", "此外", "缓冲", "擦除", "叠加", "intersect", "erase", "overlay"]
        is_complex = any(kw in user_input for kw in multi_step_keywords)

        if is_complex:
            # 工作流翻译模式（多步骤）
            raw_workflow = reasoner.translate_workflow(user_input, scenario=scenario)

            # 构建工作流 GeoDSL
            steps_data = raw_workflow.get("steps", [])
            return build_workflow_dsl(
                steps_data=steps_data,
                scenario=scenario,
                user_input=user_input,
                final_output=raw_workflow.get("final_output", "final_result"),
            )
        else:
            # 简单翻译模式（向后兼容）
            raw_dsl = reasoner.translate(user_input, scenario=scenario)
            return _build_from_reasoner_output(raw_dsl, scenario, extracted_params)

    # ── 默认：确定性 DSL 构建 ───────────────────────────────────────────────
    builder = get_builder()
    return builder.build(scenario, extracted_params, task_suffix)


def _build_from_reasoner_output(
    raw_dsl: Dict[str, Any],
    fallback_scenario: Scenario,
    extracted_params: Dict[str, Any],
) -> GeoDSL:
    """
    将 Reasoner 输出的原始 DSL 字典转换为 GeoDSL，并进行 Schema 校验。
    """
    # 补充 scenario（Reasoner 可能输出字符串）
    scenario_str = raw_dsl.get("scenario", fallback_scenario.value)
    try:
        scenario = Scenario(scenario_str)
    except ValueError:
        scenario = fallback_scenario

    # 合并 extracted_params（用于补充 Reasoner 漏掉的字段）
    merged = {**raw_dsl}

    # 补充 inputs/parameters 中的缺失字段
    inputs = merged.get("inputs", {})
    parameters = merged.get("parameters", {})
    for k, v in extracted_params.items():
        if k == "user_input":
            continue
        if k in inputs or k in parameters:
            continue
        # 根据字段名决定放哪
        if k in ("start", "end", "input_layer", "layer1", "layer2",
                 "input_points", "input_file", "location", "dem_file"):
            if k not in inputs:
                inputs[k] = v
        else:
            if k not in parameters:
                parameters[k] = v

    # Schema 校验（会填充默认值、报错缺失必填字段）
    validator = get_validator()
    try:
        validator.validate(scenario, inputs, parameters)
    except SchemaValidationError:
        # Reasoner 模式：允许校验失败时尝试自动修正
        inputs, parameters = _fix_missing_fields(
            scenario, inputs, parameters
        )

    merged["scenario"] = scenario
    merged["inputs"] = inputs
    merged["parameters"] = parameters
    merged.setdefault("version", "1.0")
    
    # 使用安全的方式获取 scenario 的 value
    task_val = scenario.value if hasattr(scenario, "value") else str(scenario)
    merged.setdefault("task", task_val)
    merged.setdefault("outputs", {"map": True, "summary": True})

    return GeoDSL(**merged)


def _fix_missing_fields(
    scenario: Scenario,
    inputs: Dict[str, Any],
    parameters: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """尝试用默认值填充缺失的必填字段（Reasoner 模式保底）"""
    from geoagent.layers.layer4_dsl import SCHEMA_REQUIRED_PARAMS
    schema = SCHEMA_REQUIRED_PARAMS.get(scenario, {})
    fixed_inputs = dict(inputs)
    fixed_params = dict(parameters)

    for field_name, spec in schema.items():
        required = spec.get("required", False)
        default = spec.get("default")
        target = fixed_params if field_name not in ("start", "end", "input_layer",
                                                      "layer1", "layer2", "input_points",
                                                      "input_file", "location", "dem_file") else fixed_inputs

        if required:
            if field_name not in target or target[field_name] is None:
                if default is not None:
                    target[field_name] = default
                else:
                    target[field_name] = f"[MISSING:{field_name}]"

    return fixed_inputs, fixed_params


def build_dsl_with_reasoner(
    user_input: str,
    scenario: Scenario,
    extracted_params: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = "deepseek-r1",
) -> GeoDSL:
    """
    使用 Reasoner 模式构建 DSL（便捷函数，直接传入 API Key）。

    适用于：复杂 NL（多步骤、复合条件）。
    不适用于：简单 NL（如 "从 A 到 B"），请直接用 build_dsl()。
    """
    from geoagent.layers.reasoner import get_reasoner

    reasoner = get_reasoner(api_key=api_key, model=model)
    raw_dsl = reasoner.translate(user_input, scenario=scenario)
    return _build_from_reasoner_output(raw_dsl, scenario, extracted_params)


def build_workflow_dsl(
    steps_data: List[Dict[str, Any]],
    scenario: Scenario,
    user_input: str = "",
    final_output: str = "final_result",
) -> GeoDSL:
    """
    从工作流步骤数据构建 GeoDSL（工作流模式便捷函数）。

    Args:
        steps_data: 工作流步骤列表，每项为字典
        scenario: 场景类型
        user_input: 原始用户输入
        final_output: 最终输出变量名

    Returns:
        is_workflow=True 的 GeoDSL 对象

    示例：
        steps = [
            {"step_id": "step_1", "task": "buffer", "inputs": {...}, "output_id": "tmp_1"},
            {"step_id": "step_2", "task": "overlay", "inputs": {...}, "output_id": "final"},
        ]
        dsl = build_workflow_dsl(steps, Scenario.OVERLAY, "选址分析")
    """
    steps = [WorkflowStep(**s) for s in steps_data]
    return GeoDSL(
        version="2.0",
        scenario=scenario,
        task="workflow",
        inputs={"user_input": user_input},
        is_workflow=True,
        steps=steps,
        final_output=final_output,
        outputs=OutputSpec(map=True, summary=True),
    )


def validate_workflow_steps(steps: List[WorkflowStep]) -> List[str]:
    """
    校验工作流步骤的合法性。

    Args:
        steps: 工作流步骤列表

    Returns:
        错误信息列表（空列表表示校验通过）
    """
    errors: List[str] = []
    output_ids: set = set()

    for i, step in enumerate(steps):
        # 检查 step_id 唯一性
        if step.output_id in output_ids:
            errors.append(f"步骤 '{step.step_id}' 的 output_id '{step.output_id}' 与其他步骤重复")
        output_ids.add(step.output_id)

        # 检查 depends_on 引用有效性
        step_ids = {s.step_id for s in steps}
        for dep in step.depends_on:
            if dep not in step_ids:
                errors.append(f"步骤 '{step.step_id}' 依赖的 '{dep}' 不存在")

        # 检查自身循环依赖
        if step.step_id in step.depends_on:
            errors.append(f"步骤 '{step.step_id}' 存在自身循环依赖")

        # 检查输入引用有效性
        for key, value in step.inputs.items():
            if isinstance(value, str) and value.startswith("tmp_"):
                if value not in output_ids:
                    errors.append(f"步骤 '{step.step_id}' 引用了尚未生成的中间变量 '{value}'")

    return errors


__all__ = [
    "OutputSpec",
    "WorkflowStep",
    "GeoDSL",
    "VisualizationSpec",
    "ViewSpec",
    "LayerSpec",
    "SCHEMA_REQUIRED_PARAMS",
    "SchemaValidationError",
    "SchemaValidator",
    "DSLBuilder",
    "get_validator",
    "get_builder",
    "validate_dsl",
    "build_dsl",
    "build_dsl_with_reasoner",
    "build_workflow_dsl",
    "validate_workflow_steps",
    "_build_from_reasoner_output",
]
