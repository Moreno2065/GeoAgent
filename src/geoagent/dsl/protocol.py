"""
GeoDSL 协议定义
================
GeoAgent 统一任务描述语言（GeoDSL）的核心协议定义。

设计原则：
1. LLM 只做翻译（自然语言 → 结构化 DSL），不参与决策
2. 所有任务必须符合本协议定义的格式
3. 执行层完全确定性，不依赖 LLM 的后续判断
"""

from __future__ import annotations

from typing import Literal, Optional, Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum


# =============================================================================
# 场景类型枚举（与 TaskType 保持一致）
# =============================================================================

class ScenarioType(str, Enum):
    """GeoDSL 支持的场景类型枚举"""
    ROUTE = "route"                          # 路径/可达性分析
    BUFFER = "buffer"                        # 缓冲/邻近分析
    OVERLAY = "overlay"                       # 叠置/裁剪分析
    INTERPOLATION = "interpolation"          # 插值/表面分析
    SHADOW_ANALYSIS = "shadow_analysis"      # 阴影分析
    VIEWSHED = "viewshed"                   # 视域/可见性分析
    NDVI = "ndvi"                            # 植被指数分析
    HOTSPOT = "hotspot"                      # 热点/统计聚合分析
    VISUALIZATION = "visualization"          # 可视化
    ACCESSIBILITY = "accessibility"          # 可达性分析
    SUITABILITY = "suitability"             # 选址分析
    CODE_SANDBOX = "code_sandbox"           # 受限代码执行
    MULTI_CRITERIA_SEARCH = "multi_criteria_search"  # 多条件综合搜索
    GENERAL = "general"                      # 通用任务


# =============================================================================
# 输出规格定义
# =============================================================================

class OutputSpec(BaseModel):
    """输出规格定义"""
    map: bool = Field(default=True, description="是否生成地图")
    summary: bool = Field(default=True, description="是否生成文字摘要")
    files: List[str] = Field(default_factory=list, description="输出文件路径列表")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="数值指标")
    explanation: Optional[str] = Field(default=None, description="解释卡片文本")


# =============================================================================
# GeoDSL 核心模型
# =============================================================================

class GeoDSL(BaseModel):
    """
    GeoDSL 统一任务描述语言

    用途：
    1. LLM 将自然语言翻译为标准 DSL
    2. Validator 校验参数完整性
    3. Router 根据 scenario 分发到对应 Executor
    4. Executor 返回标准化结果

    示例：
        NL: "芜湖南站到方特欢乐世界的步行路径"
        GeoDSL:
            version: "1.0"
            scenario: "route"
            task: "route"
            inputs:
                start: "芜湖南站"
                end: "方特欢乐世界"
            parameters:
                mode: "walking"
            outputs:
                map: true
                summary: true
    """
    version: str = Field(default="1.0", description="DSL 协议版本")
    scenario: ScenarioType = Field(description="场景类型")
    task: str = Field(description="具体任务类型（与 scenario 一致或更细化）")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="输入参数（数据源、位置等）"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="分析参数（方法、距离、阈值等）"
    )
    outputs: OutputSpec = Field(
        default_factory=OutputSpec,
        description="输出规格"
    )
    # 待追问的字段（由 ClarificationEngine 填充）
    clarification: Optional[List[str]] = Field(
        default=None,
        description="需要追问的参数列表"
    )
    # 追问后的答案（由前端填充）
    clarification_answers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="追问的答案"
    )

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# 澄清问题模型
# =============================================================================

class ClarificationQuestion(BaseModel):
    """澄清问题定义"""
    field: str = Field(description="需要澄清的字段名")
    question: str = Field(description="追问话术")
    options: Optional[List[str]] = Field(default=None, description="可选答案列表")
    required: bool = Field(default=True, description="是否必填")


class ClarificationResult(BaseModel):
    """澄清结果"""
    needs_clarification: bool = Field(description="是否需要追问")
    questions: List[ClarificationQuestion] = Field(default_factory=list)
    auto_filled: Dict[str, Any] = Field(default_factory=dict, description="自动填充的参数")


# =============================================================================
# 执行结果标准化模型
# =============================================================================

class ExecutorResult(BaseModel):
    """
    Executor 返回的标准化结果

    所有 GIS Executor 必须返回此格式的结果，
    确保前端可以统一展示。
    """
    success: bool = Field(description="执行是否成功")
    scenario: str = Field(description="场景类型")
    task: str = Field(description="任务类型")
    summary: str = Field(description="业务结论摘要")
    detail: Optional[str] = Field(default=None, description="技术详情")
    map_file: Optional[str] = Field(default=None, description="地图文件路径")
    output_files: List[str] = Field(default_factory=list, description="输出文件列表")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="数值指标")
    error: Optional[str] = Field(default=None, description="错误信息（失败时）")
    explanation: str = Field(
        default="",
        description="解释卡片：做什么、为什么、结果含义"
    )

    def to_display_dict(self) -> Dict[str, Any]:
        """转换为前端展示用的字典"""
        return {
            "success": self.success,
            "summary": self.summary,
            "detail": self.detail,
            "map_file": self.map_file,
            "files": self.output_files,
            "metrics": self.metrics,
            "explanation": self.explanation,
            "error": self.error,
        }

    @classmethod
    def from_executor_result(cls, executor_result: Any) -> "ExecutorResult":
        """
        从执行层 ExecutorResult 转换为本协议的展示层 ExecutorResult

        Args:
            executor_result: executors/base.py 中的 ExecutorResult 对象或字典

        Returns:
            展示层 ExecutorResult
        """
        # 获取数据字典
        if isinstance(executor_result, dict):
            data = executor_result.get("data", executor_result)
            task_type = executor_result.get("task_type", "")
            success = executor_result.get("success", True)
            error = executor_result.get("error")
        elif hasattr(executor_result, "data"):
            data = executor_result.data or {}
            task_type = getattr(executor_result, "task_type", "")
            success = executor_result.success
            error = executor_result.error
        else:
            data = {}
            task_type = ""
            success = True
            error = None

        # 提取文件
        output_files = []
        if data.get("output_file"):
            output_files.append(data["output_file"])
        if data.get("map_file"):
            output_files.append(data["map_file"])
        if data.get("files"):
            files = data["files"]
            output_files.extend(files if isinstance(files, list) else [files])
        output_files = list(set(output_files))

        # 提取指标
        metric_keys = [
            "distance", "duration", "feature_count", "result_count",
            "visible_area_km2", "coverage_area", "mean_ndvi",
            "hotspot_count", "coldspot_count", "building_count",
            "total_area", "valid_pixels", "resolution",
        ]
        metrics = {k: v for k, v in data.items() if k in metric_keys}

        # 生成摘要和解释
        summary = _protocol_summary(task_type, data)
        explanation = _protocol_explanation(task_type, data)

        return cls(
            success=success,
            scenario=task_type,
            task=task_type,
            summary=summary,
            detail=data.get("detail"),
            map_file=data.get("map_file") or data.get("output_file"),
            output_files=output_files,
            metrics=metrics,
            error=error,
            explanation=explanation,
        )


def _protocol_summary(scenario: str, result: Dict[str, Any]) -> str:
    """生成业务结论摘要"""
    summaries = {
        "route": f"路径规划完成：总距离 {result.get('distance', '?')} 米，预计 {result.get('duration', '?')} 分钟",
        "buffer": f"缓冲区分析完成：生成 {result.get('feature_count', '?')} 个缓冲多边形",
        "overlay": f"叠加分析完成：得到 {result.get('result_count', '?')} 个交集要素",
        "interpolation": f"插值分析完成：生成 {result.get('resolution', '?')} 米分辨率的栅格",
        "accessibility": f"可达性分析完成：以 {result.get('location', '?')} 为中心的可达范围",
        "suitability": f"选址分析完成：找到 {result.get('result_count', result.get('top_n', '?'))} 个候选位置",
        "viewshed": f"视域分析完成：可视范围约 {result.get('visible_area_km2', '?')} 平方公里",
        "shadow_analysis": f"阴影分析完成：计算了 {result.get('building_count', '?')} 个建筑物的阴影",
        "ndvi": f"NDVI 分析完成：平均植被指数 {result.get('mean_ndvi', '?')}",
        "hotspot": f"热点分析完成：识别 {result.get('hotspot_count', '?')} 个热点区域",
        "visualization": f"可视化完成：生成地图 {result.get('output_file', '?')}",
    }
    return summaries.get(scenario, result.get("summary", "分析完成"))


def _protocol_explanation(scenario: str, result: Dict[str, Any]) -> str:
    """生成解释卡片"""
    explanations = {
        "route": (
            f"**做了什么：** 计算了从起点到终点的最优路径。\n\n"
            f"**为什么这么做：** 使用最短路径算法找到时间或距离最优的路线。\n\n"
            f"**结果含义：** "
            f"总距离 {result.get('distance', '?')} 米，"
            f"预计耗时 {result.get('duration', '?')} 分钟。"
        ),
        "buffer": (
            f"**做了什么：** 以指定的距离对输入要素创建缓冲区。\n\n"
            f"**为什么这么做：** 缓冲区用于确定某要素的影响范围或邻近区域。\n\n"
            f"**结果含义：** "
            f"共生成 {result.get('feature_count', '?')} 个缓冲多边形，"
            f"总面积 {result.get('total_area', '?')} 平方米。"
        ),
        "overlay": (
            f"**做了什么：** 对两个图层执行空间叠加分析。\n\n"
            f"**为什么这么做：** 叠加分析用于确定空间要素的交集、合并或差集。\n\n"
            f"**结果含义：** 得到 {result.get('result_count', '?')} 个结果要素。"
        ),
        "interpolation": (
            f"**做了什么：** 使用空间插值算法基于离散点生成连续表面。\n\n"
            f"**为什么这么做：** 插值可以将稀疏的点数据转换为连续的栅格表面。\n\n"
            f"**结果含义：** 生成的栅格分辨率为 {result.get('resolution', '?')} 米。"
        ),
        "accessibility": (
            f"**做了什么：** 计算了从中心点在特定时间内可达的范围。\n\n"
            f"**为什么这么做：** 等时圈分析用于评估地点的可达性和服务范围。\n\n"
            f"**结果含义：** 以 {result.get('location', '?')} 为中心，"
            f"{result.get('mode', '步行')} 可达约 {result.get('coverage_area', '?')} 平方公里。"
        ),
        "suitability": (
            f"**做了什么：** 使用多准则决策分析（MCDA）对候选位置进行适宜性评价。\n\n"
            f"**为什么这么做：** MCDA 综合考虑多个因素来确定最佳选址。\n\n"
            f"**结果含义：** 根据 {result.get('criteria_count', '?')} 个评价因素，"
            f"筛选出 {result.get('result_count', '?')} 个最适宜的候选位置。"
        ),
        "viewshed": (
            f"**做了什么：** 基于 DEM 计算了观察点的可视范围。\n\n"
            f"**为什么这么做：** 视域分析用于确定从某点能看到哪些区域。\n\n"
            f"**结果含义：** 可视范围约 {result.get('visible_area_km2', '?')} 平方公里。"
        ),
        "shadow_analysis": (
            f"**做了什么：** 计算了建筑物在指定时间的阴影投射。\n\n"
            f"**为什么这么做：** 阴影分析用于评估日照条件、采光分析等。\n\n"
            f"**结果含义：** 分析了 {result.get('building_count', '?')} 个建筑物，"
            f"总阴影面积 {result.get('shadow_area', '?')} 平方米。"
        ),
        "ndvi": (
            f"**做了什么：** 计算了遥感影像的归一化植被指数（NDVI）。\n\n"
            f"**为什么这么做：** NDVI 是评估植被覆盖和生长状态的重要指标。\n\n"
            f"**结果含义：** NDVI 范围 [{result.get('min_ndvi', '?')}, {result.get('max_ndvi', '?')}]，"
            f"平均值 {result.get('mean_ndvi', '?')}。"
        ),
        "hotspot": (
            f"**做了什么：** 执行了空间自相关分析（Getis-Ord Gi*）识别热点和冷点。\n\n"
            f"**为什么这么做：** 热点分析用于识别空间聚集的显著性区域。\n\n"
            f"**结果含义：** 识别出 {result.get('hotspot_count', '?')} 个高值热点区域，"
            f"和 {result.get('coldspot_count', '?')} 个低值冷点区域。"
        ),
        "visualization": (
            f"**做了什么：** 生成了交互式/静态地图可视化。\n\n"
            f"**为什么这么做：** 可视化帮助理解空间数据的分布和模式。\n\n"
            f"**结果含义：** 地图文件：{result.get('output_file', '?')}。"
        ),
    }
    return explanations.get(scenario, "分析已完成。")


# =============================================================================
# 编排状态枚举
# =============================================================================

class OrchestrationStatus(str, Enum):
    """编排状态枚举"""
    READY = "ready"                           # 准备就绪，可执行
    CLARIFICATION_NEEDED = "clarification_needed"  # 需要追问
    INVALID = "invalid"                       # 参数无效
    EXECUTING = "executing"                   # 执行中
    COMPLETED = "completed"                   # 执行完成
    FAILED = "failed"                         # 执行失败


# =============================================================================
# 任务 DSL 生成器（LLM 调用模板）
# =============================================================================

DSL_GENERATION_PROMPT = """
你是一个 GIS 任务翻译专家，负责将用户的自然语言请求翻译成标准的 GeoDSL 格式。

## GeoDSL 协议

任务 DSL 必须符合以下 JSON Schema：
{
  "version": "1.0",
  "scenario": "route|buffer|overlay|interpolation|shadow_analysis|ndvi|hotspot|accessibility|suitability|general",
  "task": "具体任务标识",
  "inputs": { ... },
  "parameters": { ... },
  "outputs": {
    "map": true,
    "summary": true,
    "files": [],
    "metrics": {}
  }
}

## 场景类型说明

1. **route** - 路径/可达性分析
   inputs: start, end, mode
   parameters: mode(walking/driving/transit), provider(amap/osm)

2. **buffer** - 缓冲/邻近分析
   inputs: input_layer, distance, unit
   parameters: dissolve, cap_style

3. **overlay** - 叠置/裁剪分析
   inputs: layer1, layer2, operation
   parameters: (operation: intersect/union/clip/difference)

4. **interpolation** - 插值/表面分析
   inputs: input_points, value_field
   parameters: method(IDW/kriging/nearest_neighbor), power, resolution

5. **accessibility** - 可达性分析（等时圈、服务范围）
   inputs: location, mode
   parameters: time_threshold, grid_resolution

6. **suitability** - 选址/适宜性分析
   inputs: criteria_layers, weights
   parameters: method(wsum/mcdm), thresholds

## 输出要求

1. 只输出 JSON，不要解释
2. 确保所有必填字段都有值
3. 使用中文描述
4. 如有缺失参数，在 clarification 字段标注

请将以下用户请求翻译为 GeoDSL：

{user_input}
"""


# =============================================================================
# 工具函数
# =============================================================================

def validate_geo_dsl(data: Dict[str, Any]) -> GeoDSL:
    """
    校验并解析 GeoDSL 数据

    Args:
        data: 待校验的字典数据

    Returns:
        GeoDSL: 校验通过的任务对象

    Raises:
        ValidationError: 校验失败时抛出
    """
    return GeoDSL(**data)


def _dsl_to_task_type(scenario: ScenarioType) -> str:
    """将场景类型映射为任务类型"""
    return scenario.value


# =============================================================================
# 场景到执行器的映射表（延迟导入，避免循环依赖）
# 此映射已迁移到 geoagent.executors.router.SCENARIO_EXECUTOR_KEY
# 此处保留用于 dsl 层内部使用，通过函数延迟获取
# =============================================================================

def get_scenario_executor_map() -> dict[str, str]:
    """
    获取场景到执行器的映射表

    通过延迟导入避免循环依赖。

    Returns:
        dict: scenario → executor key 映射表
    """
    try:
        from geoagent.executors.router import SCENARIO_EXECUTOR_KEY
        return dict(SCENARIO_EXECUTOR_KEY)
    except ImportError:
        # Fallback：硬编码值
        return {
            "route":            "route",
            "buffer":           "buffer",
            "overlay":          "overlay",
            "interpolation":     "interpolation",
            "shadow_analysis":   "shadow_analysis",
            "ndvi":             "ndvi",
            "hotspot":          "hotspot",
            "visualization":     "visualization",
            "accessibility":     "route",
            "suitability":      "general",
            "general":          "general",
        }


# 向后兼容别名
SCENARIO_EXECUTOR_MAP = get_scenario_executor_map()
