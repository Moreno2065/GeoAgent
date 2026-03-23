"""
Result Renderer - 结果渲染器
===========================
将执行结果转换为标准化的前端展示格式。

核心职责：
1. 统一结果格式
2. 生成解释卡片
3. 生成业务结论摘要
4. 关联地图文件
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime


# =============================================================================
# 场景结果渲染器映射
# =============================================================================

_RESULT_RENDERERS: Dict[str, callable] = {}


def _register_renderer(scenario: str):
    """装饰器：注册结果渲染器"""
    def decorator(func):
        _RESULT_RENDERERS[scenario] = func
        return func
    return decorator


# =============================================================================
# 基础渲染器
# =============================================================================

def render_basic_result(
    scenario: str,
    result_data: Dict[str, Any],
    explanation: str = "",
) -> Dict[str, Any]:
    """
    渲染基础结果

    Args:
        scenario: 场景类型
        result_data: 执行结果数据
        explanation: 解释卡片文本

    Returns:
        渲染后的结果字典
    """
    success = result_data.get("success", True)

    return {
        "success": success,
        "scenario": scenario,
        "summary": _generate_summary(scenario, result_data),
        "detail": result_data,
        "map_file": result_data.get("map_file") or result_data.get("output_file"),
        "output_files": _extract_output_files(result_data),
        "metrics": _extract_metrics(result_data),
        "explanation": explanation or _generate_explanation(scenario, result_data),
        "timestamp": datetime.now().isoformat(),
        "error": result_data.get("error") if not success else None,
    }


def _generate_summary(scenario: str, result: Dict[str, Any]) -> str:
    """生成业务结论摘要"""
    summaries = {
        "route": f"路径规划完成：总距离 {result.get('distance', '?')} 米，预计 {result.get('duration', '?')} 分钟",
        "buffer": f"缓冲区分析完成：生成 {result.get('feature_count', '?')} 个缓冲多边形",
        "overlay": f"叠加分析完成：得到 {result.get('result_count', '?')} 个交集要素",
        "interpolation": f"插值分析完成：生成 {result.get('resolution', '?')} 米分辨率的栅格",
        "accessibility": f"可达性分析完成：以 {result.get('location', '?')} 为中心 {result.get('time_threshold', '?')} 分钟可达范围",
        "suitability": f"选址分析完成：找到 {result.get('result_count', result.get('top_n', '?'))} 个候选位置",
        "viewshed": f"视域分析完成：可视范围约 {result.get('visible_area_km2', '?')} 平方公里",
        "shadow_analysis": f"阴影分析完成：计算了 {result.get('building_count', '?')} 个建筑物的阴影",
        "ndvi": f"NDVI 分析完成：平均植被指数 {result.get('mean_ndvi', '?')}，覆盖 {result.get('valid_pixels', '?')} 个有效像元",
        "hotspot": f"热点分析完成：识别 {result.get('hotspot_count', '?')} 个热点区域，{result.get('coldspot_count', '?')} 个冷点区域",
        "visualization": f"可视化完成：生成地图 {result.get('output_file', '?')}",
        "general": f"通用任务执行完成：{result.get('description', result.get('task_description', '任务完成'))}",
        "unknown": f"任务执行完成",
    }
    return summaries.get(scenario, result.get("summary", "分析完成"))


def _generate_explanation(scenario: str, result: Dict[str, Any]) -> str:
    """生成解释卡片"""
    explanations = {
        "route": (
            "**做了什么：** 计算了从起点到终点的最优路径。\n\n"
            "**为什么这么做：** 使用最短路径算法找到时间或距离最优的路线。\n\n"
            "**结果含义：** "
            f"总距离 {result.get('distance', '?')} 米，"
            f"预计耗时 {result.get('duration', '?')} 分钟。"
        ),
        "buffer": (
            "**做了什么：** 以指定的距离对输入要素创建缓冲区。\n\n"
            "**为什么这么做：** 缓冲区用于确定某要素的影响范围或邻近区域。\n\n"
            "**结果含义：** "
            f"共生成 {result.get('feature_count', '?')} 个缓冲多边形，"
            f"总面积 {result.get('total_area', '?')} 平方米。"
        ),
        "overlay": (
            "**做了什么：** 对两个图层执行空间叠加分析。\n\n"
            "**为什么这么做：** 叠加分析用于确定空间要素的交集、合并或差集。\n\n"
            "**结果含义：** "
            f"得到 {result.get('result_count', '?')} 个结果要素。"
        ),
        "interpolation": (
            "**做了什么：** 使用空间插值算法基于离散点生成连续表面。\n\n"
            "**为什么这么做：** 插值可以将稀疏的点数据转换为连续的栅格表面。\n\n"
            "**结果含义：** "
            f"生成的栅格分辨率为 {result.get('resolution', '?')} 米，"
            f"有效范围 {result.get('valid_pixels', '?')} 平方公里。"
        ),
        "accessibility": (
            "**做了什么：** 计算了从中心点在特定时间内可达的范围。\n\n"
            "**为什么这么做：** 等时圈分析用于评估地点的可达性和服务范围。\n\n"
            "**结果含义：** "
            f"以 {result.get('location', '?')} 为中心，"
            f"{result.get('mode', '步行')} {result.get('time_threshold', '?')} 分钟"
            f"可覆盖约 {result.get('coverage_area', '?')} 平方公里。"
        ),
        "suitability": (
            "**做了什么：** 使用多准则决策分析（MCDA）对候选位置进行适宜性评价。\n\n"
            "**为什么这么做：** MCDA 综合考虑多个因素来确定最佳选址。\n\n"
            "**结果含义：** "
            f"根据 {result.get('criteria_count', '?')} 个评价因素，"
            f"筛选出 {result.get('result_count', '?')} 个最适宜的候选位置。"
        ),
        "viewshed": (
            "**做了什么：** 基于数字高程模型（DEM）计算了观察点的可视范围。\n\n"
            "**为什么这么做：** 视域分析用于确定从某点能看到哪些区域。\n\n"
            "**结果含义：** "
            f"观察点 {result.get('observer_location', '?')} 的可视范围约 "
            f"{result.get('visible_area_km2', '?')} 平方公里，"
            f"占观察范围的比例为 {result.get('visibility_ratio', '?')}%。"
        ),
        "shadow_analysis": (
            "**做了什么：** 计算了建筑物在指定时间的阴影投射。\n\n"
            "**为什么这么做：** 阴影分析用于评估日照条件、采光分析等。\n\n"
            "**结果含义：** "
            f"分析了 {result.get('building_count', '?')} 个建筑物，"
            f"总阴影面积 {result.get('shadow_area', '?')} 平方米。"
        ),
        "ndvi": (
            "**做了什么：** 计算了遥感影像的归一化植被指数（NDVI）。\n\n"
            "**为什么这么做：** NDVI 是评估植被覆盖和生长状态的重要指标。\n\n"
            "**结果含义：** "
            f"NDVI 范围 [{result.get('min_ndvi', '?')}, {result.get('max_ndvi', '?')}]，"
            f"平均值 {result.get('mean_ndvi', '?')}。"
            f"正值表示有植被覆盖，接近 1 表示植被茂密。"
        ),
        "hotspot": (
            "**做了什么：** 执行了空间自相关分析（Getis-Ord Gi*）识别热点和冷点。\n\n"
            "**为什么这么做：** 热点分析用于识别空间聚集的显著性区域。\n\n"
            "**结果含义：** "
            f"识别出 {result.get('hotspot_count', '?')} 个高值热点区域（置信度>95%），"
            f"和 {result.get('coldspot_count', '?')} 个低值冷点区域。"
        ),
        "visualization": (
            "**做了什么：** 生成了交互式/静态地图可视化。\n\n"
            "**为什么这么做：** 可视化帮助理解空间数据的分布和模式。\n\n"
            "**结果含义：** "
            f"地图文件：{result.get('output_file', '?')}。"
        ),
        "general": (
            f"**做了什么：** 执行了通用 GIS 分析任务。\n\n"
            f"**为什么这么做：** 系统无法将任务归类到标准场景，执行通用处理。\n\n"
            f"**结果含义：** "
            f"{result.get('description', result.get('task_description', '任务已完成'))}"
        ),
        "unknown": "**做了什么：** 系统无法识别任务类型，执行了默认处理。",
    }
    return explanations.get(scenario, "分析已完成。")


def _extract_output_files(result: Dict[str, Any]) -> List[str]:
    """从结果中提取输出文件列表"""
    files = []

    # 直接字段
    if result.get("output_file"):
        files.append(result["output_file"])
    if result.get("map_file"):
        files.append(result["map_file"])

    # 间接字段
    if result.get("files"):
        if isinstance(result["files"], list):
            files.extend(result["files"])
        else:
            files.append(result["files"])

    # 去重
    return list(set(files))


def _extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """从结果中提取关键指标"""
    metrics = {}

    # 常见指标字段
    metric_keys = [
        "distance", "duration", "feature_count", "result_count",
        "visible_area_km2", "coverage_area", "mean_ndvi",
        "hotspot_count", "coldspot_count", "building_count",
        "total_area", "valid_pixels", "resolution",
    ]

    for key in metric_keys:
        if key in result:
            metrics[key] = result[key]

    return metrics


# =============================================================================
# 场景特定渲染器
# =============================================================================

@_register_renderer("route")
def _render_route(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染路径规划结果"""
    base = render_basic_result("route", result_data)

    # 添加路径详情
    if result_data.get("steps"):
        base["route_details"] = {
            "distance": result_data.get("distance"),
            "duration": result_data.get("duration"),
            "steps": result_data.get("steps", [])[:5],  # 只显示前5步
            "waypoints": result_data.get("waypoints", []),
        }

    return base


@_register_renderer("buffer")
def _render_buffer(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染缓冲区分析结果"""
    base = render_basic_result("buffer", result_data)

    # 添加缓冲区详情
    base["buffer_details"] = {
        "distance": result_data.get("distance"),
        "feature_count": result_data.get("feature_count"),
        "total_area": result_data.get("total_area"),
        "dissolved": result_data.get("dissolved", False),
    }

    return base


@_register_renderer("accessibility")
def _render_accessibility(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染可达性分析结果"""
    base = render_basic_result("accessibility", result_data)

    # 添加可达性详情
    base["accessibility_details"] = {
        "center": result_data.get("location"),
        "mode": result_data.get("mode"),
        "time_threshold": result_data.get("time_threshold"),
        "coverage_area": result_data.get("coverage_area"),
        "reachable_points": result_data.get("reachable_points"),
    }

    return base


@_register_renderer("suitability")
def _render_suitability(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染选址分析结果"""
    base = render_basic_result("suitability", result_data)

    # 添加选址详情
    base["suitability_details"] = {
        "criteria_count": result_data.get("criteria_count"),
        "method": result_data.get("method"),
        "top_locations": result_data.get("top_locations", [])[:10],
        "score_distribution": result_data.get("score_distribution"),
    }

    return base


@_register_renderer("ndvi")
def _render_ndvi(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染 NDVI 分析结果"""
    base = render_basic_result("ndvi", result_data)

    # 添加 NDVI 详情
    mean_ndvi = result_data.get("mean_ndvi", 0)
    vegetation_cover = "无植被" if mean_ndvi < 0.1 else "稀疏植被" if mean_ndvi < 0.3 else "中等植被" if mean_ndvi < 0.6 else "茂密植被"

    base["ndvi_details"] = {
        "mean": mean_ndvi,
        "min": result_data.get("min_ndvi"),
        "max": result_data.get("max_ndvi"),
        "std": result_data.get("std_ndvi"),
        "vegetation_cover": vegetation_cover,
        "valid_pixels": result_data.get("valid_pixels"),
    }

    return base


@_register_renderer("hotspot")
def _render_hotspot(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染热点分析结果"""
    base = render_basic_result("hotspot", result_data)

    # 添加热点详情
    base["hotspot_details"] = {
        "hotspot_count": result_data.get("hotspot_count"),
        "coldspot_count": result_data.get("coldspot_count"),
        "global_moran_i": result_data.get("moran_i"),
        "p_value": result_data.get("p_value"),
        "significance": "显著" if result_data.get("p_value", 1) < 0.05 else "不显著",
    }

    return base


@_register_renderer("overlay")
def _render_overlay(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染叠加分析结果"""
    base = render_basic_result("overlay", result_data)

    base["overlay_details"] = {
        "operation": result_data.get("operation"),
        "result_count": result_data.get("result_count"),
        "total_area": result_data.get("total_area"),
        "layer1_features": result_data.get("layer1_features"),
        "layer2_features": result_data.get("layer2_features"),
    }

    return base


@_register_renderer("interpolation")
def _render_interpolation(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染插值分析结果"""
    base = render_basic_result("interpolation", result_data)

    base["interpolation_details"] = {
        "method": result_data.get("method"),
        "resolution": result_data.get("resolution"),
        "valid_area_km2": result_data.get("valid_area_km2"),
        "min_value": result_data.get("min_value"),
        "max_value": result_data.get("max_value"),
        "mean_value": result_data.get("mean_value"),
    }

    return base


@_register_renderer("viewshed")
def _render_viewshed(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染视域分析结果"""
    base = render_basic_result("viewshed", result_data)

    visible_area = result_data.get("visible_area_km2", 0)
    total_area = result_data.get("total_area_km2", 1)
    visibility_ratio = round(visible_area / total_area * 100, 1) if total_area > 0 else 0

    base["viewshed_details"] = {
        "observer_location": result_data.get("observer_location"),
        "observer_height": result_data.get("observer_height"),
        "max_distance": result_data.get("max_distance"),
        "visible_area_km2": visible_area,
        "total_area_km2": total_area,
        "visibility_ratio": visibility_ratio,
        "visible_pixels": result_data.get("visible_pixels"),
    }

    return base


@_register_renderer("shadow_analysis")
def _render_shadow_analysis(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染阴影分析结果"""
    base = render_basic_result("shadow_analysis", result_data)

    base["shadow_details"] = {
        "building_count": result_data.get("building_count"),
        "total_shadow_area": result_data.get("shadow_area"),
        "analysis_time": result_data.get("time"),
        "avg_shadow_length": result_data.get("avg_shadow_length"),
    }

    return base


@_register_renderer("visualization")
def _render_visualization(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染可视化结果"""
    base = render_basic_result("visualization", result_data)

    base["visualization_details"] = {
        "viz_type": result_data.get("viz_type"),
        "layer_count": result_data.get("layer_count"),
        "feature_count": result_data.get("feature_count"),
    }

    return base


@_register_renderer("general")
def _render_general(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染通用任务结果"""
    base = render_basic_result("general", result_data)

    # 从 result_data 中提取有用信息
    description = result_data.get("description", result_data.get("task_description", ""))
    parameters = result_data.get("parameters", {})

    base["general_details"] = {
        "description": description,
        "parameters": parameters,
        "message": result_data.get("message", result_data.get("raw_result", "")),
    }

    # 为 general 任务生成更完整的解释
    if description:
        base["explanation"] = (
            f"**做了什么：** {description}\n\n"
            f"**参数：** {json.dumps(parameters, ensure_ascii=False) if parameters else '无'}\n\n"
            f"**状态：** {'执行成功' if result_data.get('success', False) else '执行失败'}"
        )

    return base


@_register_renderer("unknown")
def _render_unknown(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """渲染未知/未识别任务结果"""
    return render_basic_result("unknown", result_data)


# =============================================================================
# 主渲染器
# =============================================================================

class ResultRenderer:
    """
    结果渲染器

    统一将执行结果转换为前端展示格式。

    使用方式：
        renderer = ResultRenderer()
        display = renderer.render("route", {"distance": 1234, "duration": 600})
        print(display["summary"])  # 业务结论
        print(display["map_file"]) # 地图文件
    """

    def __init__(self):
        self.renderers = _RESULT_RENDERERS

    def render(
        self,
        scenario: str,
        result_data: Dict[str, Any],
        explanation: str = "",
    ) -> Dict[str, Any]:
        """
        渲染执行结果

        Args:
            scenario: 场景类型
            result_data: 执行结果数据
            explanation: 自定义解释卡片文本

        Returns:
            渲染后的结果字典
        """
        # 检查是否有场景特定渲染器
        if scenario in self.renderers:
            # 场景特定渲染器签名：_render_xxx(result_data)，忽略 scenario
            return self.renderers[scenario](result_data)
        # 否则使用通用渲染器
        return render_basic_result(scenario, result_data, explanation)

    def render_from_json(
        self,
        json_str: str,
        scenario: str = "",
        explanation: str = "",
    ) -> Dict[str, Any]:
        """
        从 JSON 字符串渲染结果

        Args:
            json_str: JSON 格式的执行结果
            scenario: 场景类型（如果 JSON 中没有）
            explanation: 自定义解释卡片文本

        Returns:
            渲染后的结果字典
        """
        try:
            data = json.loads(json_str)
            if not scenario and "task" in data:
                scenario = data["task"]
            return self.render(scenario, data, explanation)
        except json.JSONDecodeError:
            return render_basic_result(
                scenario or "unknown",
                {"success": False, "error": "无法解析结果 JSON"},
                explanation,
            )

    def generate_report(
        self,
        result: Dict[str, Any],
        include_details: bool = True,
    ) -> str:
        """
        生成文字报告

        Args:
            result: 渲染后的结果
            include_details: 是否包含详细信息

        Returns:
            Markdown 格式的报告
        """
        lines = []

        # 标题
        scenario = result.get("scenario", "分析")
        lines.append(f"# {scenario.upper()} 分析报告")
        lines.append("")

        # 状态
        if result.get("success"):
            lines.append("**状态：** ✅ 执行成功")
        else:
            lines.append(f"**状态：** ❌ 执行失败 - {result.get('error', '未知错误')}")
            return "\n".join(lines)

        lines.append("")

        # 业务结论
        lines.append("## 📋 分析结论")
        lines.append(result.get("summary", "分析完成"))
        lines.append("")

        # 关键指标
        metrics = result.get("metrics", {})
        if metrics:
            lines.append("## 📊 关键指标")
            for key, value in metrics.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        # 解释卡片
        explanation = result.get("explanation", "")
        if explanation:
            lines.append("## 💡 结果解释")
            lines.append(explanation)
            lines.append("")

        # 详细信息
        if include_details:
            details = {k: v for k, v in result.items()
                       if k not in ("success", "scenario", "summary", "explanation", "metrics", "timestamp")}
            if details:
                lines.append("## 📁 详细信息")
                for key, value in details.items():
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value, ensure_ascii=False, indent=2)
                    lines.append(f"- **{key}**: {value}")

        # 输出文件
        output_files = result.get("output_files", [])
        if output_files:
            lines.append("")
            lines.append("## 📄 输出文件")
            for f in output_files:
                lines.append(f"- `{f}`")

        # 时间戳
        if result.get("timestamp"):
            lines.append("")
            lines.append(f"_报告生成时间: {result['timestamp']}_")

        return "\n".join(lines)


# =============================================================================
# 便捷函数
# =============================================================================

_renderer: Optional[ResultRenderer] = None


def get_renderer() -> ResultRenderer:
    """获取渲染器单例"""
    global _renderer
    if _renderer is None:
        _renderer = ResultRenderer()
    return _renderer


def render_result(
    scenario: str,
    result_data: Dict[str, Any],
    explanation: str = "",
) -> Dict[str, Any]:
    """
    便捷函数：渲染结果

    Args:
        scenario: 场景类型
        result_data: 执行结果数据
        explanation: 自定义解释卡片文本

    Returns:
        渲染后的结果字典
    """
    renderer = get_renderer()
    return renderer.render(scenario, result_data, explanation)


def generate_report(
    result: Dict[str, Any],
    include_details: bool = True,
) -> str:
    """
    便捷函数：生成文字报告

    Args:
        result: 渲染后的结果
        include_details: 是否包含详细信息

    Returns:
        Markdown 格式的报告
    """
    renderer = get_renderer()
    return renderer.generate_report(result, include_details)


__all__ = [
    "ResultRenderer",
    "get_renderer",
    "render_result",
    "generate_report",
    "render_basic_result",
]
