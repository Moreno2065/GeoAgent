"""
第6层：结果呈现层（Result Renderer）
=====================================
核心职责：
1. 将 ExecutorResult 转换为业务结论
2. 生成解释卡片
3. 标准化输出格式
4. 支持地图、图层、表格、报告等多种输出

设计原则：
- 用户不想看到"执行了 buffer + overlay + intersect"
- 他想看到的是：哪些地方更合适、为什么合适、结果图在哪、结论是什么
- 这就是产品价值
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

from geoagent.layers.layer5_executor import ExecutorResult


# =============================================================================
# 解释卡片
# =============================================================================

@dataclass
class ExplanationCard:
    """
    解释卡片

    帮助用户理解：
    1. 我做了什么
    2. 用了什么数据
    3. 为什么这么做
    4. 结果是什么意思
    """
    title: str
    what_i_did: str
    why: str
    what_it_means: str
    data_used: Optional[str] = None
    caveats: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "what_i_did": self.what_i_did,
            "why": self.why,
            "what_it_means": self.what_it_means,
            "data_used": self.data_used,
            "caveats": self.caveats,
        }


# =============================================================================
# 业务结论
# =============================================================================

@dataclass
class BusinessConclusion:
    """
    业务结论

    面向"业务结论"的输出，不是只给技术结果。
    """
    summary: str  # 一句话总结
    key_findings: List[str] = field(default_factory=list)  # 关键发现
    recommendations: List[str] = field(default_factory=list)  # 建议
    data_quality: str = "unknown"  # 数据质量
    confidence: str = "medium"  # 置信度: high/medium/low

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 渲染结果
# =============================================================================

@dataclass
class RenderResult:
    """
    标准化渲染结果

    前端可以统一展示这个格式。
    """
    success: bool
    summary: str  # 业务结论摘要
    conclusion: Optional[BusinessConclusion] = None
    explanation: Optional[ExplanationCard] = None
    map_file: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    raw_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "summary": self.summary,
            "conclusion": {
                "summary": self.conclusion.summary if self.conclusion else "",
                "key_findings": self.conclusion.key_findings if self.conclusion else [],
                "recommendations": self.conclusion.recommendations if self.conclusion else [],
                "data_quality": self.conclusion.data_quality if self.conclusion else "unknown",
                "confidence": self.conclusion.confidence if self.conclusion else "medium",
            } if self.conclusion else None,
            "explanation": {
                "title": self.explanation.title if self.explanation else "",
                "what_i_did": self.explanation.what_i_did if self.explanation else "",
                "why": self.explanation.why if self.explanation else "",
                "what_it_means": self.explanation.what_it_means if self.explanation else "",
                "data_used": self.explanation.data_used if self.explanation else None,
                "caveats": self.explanation.caveats if self.explanation else None,
            } if self.explanation else None,
            "map_file": self.map_file,
            "output_files": self.output_files,
            "metrics": self.metrics,
            "error": self.error,
        }

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_user_friendly_text(self) -> str:
        """转换为用户友好的文本"""
        lines = []

        # 结论摘要
        lines.append(f"📊 {self.summary}")

        # 关键发现
        if self.conclusion and self.conclusion.key_findings:
            lines.append("\n🔍 关键发现：")
            for finding in self.conclusion.key_findings:
                lines.append(f"  • {finding}")

        # 解释
        if self.explanation:
            lines.append(f"\n💡 {self.explanation.title}")
            lines.append(f"\n**做了什么：** {self.explanation.what_i_did}")
            lines.append(f"**为什么：** {self.explanation.why}")
            lines.append(f"**结果含义：** {self.explanation.what_it_means}")

        # 指标
        if self.metrics:
            lines.append("\n📈 关键指标：")
            for key, value in self.metrics.items():
                lines.append(f"  • {key}: {value}")

        # 输出文件
        if self.output_files:
            lines.append("\n📁 输出文件：")
            for f in self.output_files:
                lines.append(f"  • {f}")

        # 建议
        if self.conclusion and self.conclusion.recommendations:
            lines.append("\n💡 建议：")
            for rec in self.conclusion.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


# =============================================================================
# 结果渲染器
# =============================================================================

class ResultRenderer:
    """
    结果渲染器

    核心职责：
    1. 将 ExecutorResult 转换为业务结论
    2. 生成解释卡片
    3. 标准化输出格式
    """

    # 场景 → 渲染器方法映射
    _RENDER_METHODS: Dict[str, str] = {
        "route": "_render_route",
        "buffer": "_render_buffer",
        "overlay": "_render_overlay",
        "interpolation": "_render_interpolation",
        "viewshed": "_render_viewshed",
        "statistics": "_render_statistics",
        "raster": "_render_raster",
    }

    def render(self, result: ExecutorResult) -> RenderResult:
        """
        渲染执行结果

        Args:
            result: Executor 返回的结果

        Returns:
            RenderResult 标准化渲染结果
        """
        # 如果执行失败
        if not result.success:
            return self._render_error(result)

        # 根据场景渲染
        render_method = self._RENDER_METHODS.get(result.task_type, "_render_generic")
        method = getattr(self, render_method, self._render_generic)
        return method(result)

    def _render_error(self, result: ExecutorResult) -> RenderResult:
        """渲染错误结果"""
        return RenderResult(
            success=False,
            summary=f"分析失败：{result.error}",
            error=result.error,
            raw_result=result.to_dict(),
        )

    def _render_generic(self, result: ExecutorResult) -> RenderResult:
        """通用渲染"""
        data = result.data or {}
        output_files = [data.get("output_file", "")] if data.get("output_file") else []
        output_files.extend(data.get("files", []) if isinstance(data.get("files"), list) else [])

        task_type = result.task_type
        return RenderResult(
            success=True,
            summary=f"{task_type} 分析完成",
            conclusion=BusinessConclusion(
                summary=f"{task_type} 分析完成",
                key_findings=[f"使用了 {result.engine} 引擎"],
            ),
            explanation=ExplanationCard(
                title=f"{task_type} 分析结果",
                what_i_did=f"执行了 {task_type} 任务",
                why="用户请求",
                what_it_means="分析已完成",
                data_used=data.get("data_source"),
            ),
            map_file=data.get("map_file"),
            output_files=[f for f in output_files if f],
            metrics=self._extract_metrics(result),
            raw_result=result.to_dict(),
        )

    def _render_route(self, result: ExecutorResult) -> RenderResult:
        """渲染路径分析结果"""
        data = result.data or {}
        
        # 从 route_data 中提取路径信息
        route_data = data.get("route_data", {})
        distance_raw = route_data.get("distance") or data.get("distance") or "?"
        duration_raw = route_data.get("duration") or data.get("duration") or "?"
        
        # 转换距离（米）转为公里显示
        try:
            distance_m = int(distance_raw) if distance_raw != "?" else "?"
            distance_km = round(distance_m / 1000, 1) if distance_m != "?" else "?"
            distance_display = distance_km if distance_km != "?" else distance_raw
        except (ValueError, TypeError):
            distance_m = distance_raw
            distance_km = distance_raw
        
        # 转换时长（秒）转为分钟显示
        try:
            duration_s = int(duration_raw) if duration_raw != "?" else "?"
            duration_min = round(duration_s / 60, 0) if duration_s != "?" else "?"
            duration_display = int(duration_min) if duration_min != "?" else duration_raw
        except (ValueError, TypeError):
            duration_s = duration_raw
            duration_min = duration_raw
            duration_display = duration_raw
        
        mode = data.get("mode", "walking")
        start = data.get("start", route_data.get("origin_name", "?"))
        end = data.get("end", route_data.get("destination_name", "?"))

        mode_text = {
            "walking": "步行",
            "driving": "驾车",
            "cycling": "骑行",
            "transit": "公交",
        }.get(mode, mode)

        summary = f"已计算从 {start} 到 {end} 的{mode_text}路径"

        conclusion = BusinessConclusion(
            summary=summary,
            key_findings=[
                f"总距离约 {distance_display} {'公里' if distance_km != '?' else '米'}",
                f"预计 {duration_display} 分钟" if duration_display != "?" else f"预计耗时：{duration_raw} 秒",
                f"使用 {result.engine} 引擎计算",
            ],
            recommendations=[
                f"建议使用 {mode_text} 方式到达目的地" if mode == "walking" else "可选择驾车或公交前往",
            ],
            data_quality="good",
            confidence="high",
        )

        explanation = ExplanationCard(
            title=f"{mode_text}路径规划结果",
            what_i_did=f"计算了从 {start} 到 {end} 的最优 {mode_text} 路径",
            why="路径规划是 GIS 分析的基础功能，用于确定两点之间的最优路线",
            what_it_means=f"路线总长约 {distance_display} {'公里' if distance_km != '?' else '米'}，{mode_text}约 {duration_display} 分钟",
            data_used=f"基于 {result.engine} 路网数据",
        )

        output_files = []
        if data.get("output_file"):
            output_files.append(data["output_file"])
        if data.get("map_file"):
            output_files.append(data["map_file"])

        return RenderResult(
            success=True,
            summary=summary,
            conclusion=conclusion,
            explanation=explanation,
            map_file=data.get("map_file"),
            output_files=output_files,
            metrics={
                "distance_m": distance_m,
                "distance_km": distance_display,
                "duration_min": duration_display,
                "duration_s": duration_s,
                "mode": mode,
                "engine": result.engine,
                "start": start,
                "end": end,
            },
            raw_result=result.to_dict(),
        )

    def _render_buffer(self, result: ExecutorResult) -> RenderResult:
        """渲染缓冲区分析结果"""
        data = result.data or {}
        distance = data.get("distance", "?")
        unit = data.get("unit", "meters")
        feature_count = data.get("feature_count", "?")
        input_layer = data.get("input_layer", "?")

        summary = f"已完成 {input_layer} 的 {distance}{unit} 缓冲区分析"

        conclusion = BusinessConclusion(
            summary=summary,
            key_findings=[
                f"生成 {feature_count} 个缓冲多边形" if isinstance(feature_count, int) else f"缓冲要素数：{feature_count}",
                f"缓冲半径 {distance} {unit}",
                f"使用 {result.engine} 引擎计算",
            ],
            recommendations=[
                "可用于分析设施服务范围",
                "可进一步与人口数据进行叠加分析",
            ],
            data_quality="good",
            confidence="high",
        )

        explanation = ExplanationCard(
            title="缓冲区分析结果",
            what_i_did=f"以 {input_layer} 为中心，生成了半径 {distance} {unit} 的缓冲区",
            why="缓冲区分析用于确定某要素的影响范围或邻近区域，是选址分析的常用方法",
            what_it_means=f"生成了 {feature_count} 个缓冲多边形，可用于后续的空间分析",
            data_used=f"输入图层：{input_layer}",
        )

        output_files = []
        if data.get("output_file"):
            output_files.append(data["output_file"])
        if data.get("output_path"):
            output_files.append(data["output_path"])

        return RenderResult(
            success=True,
            summary=summary,
            conclusion=conclusion,
            explanation=explanation,
            map_file=data.get("map_file"),
            output_files=output_files,
            metrics={
                "distance": distance,
                "unit": unit,
                "feature_count": feature_count,
                "engine": result.engine,
            },
            raw_result=result.to_dict(),
        )

    def _render_overlay(self, result: ExecutorResult) -> RenderResult:
        """渲染叠置分析结果"""
        data = result.data or {}
        operation = data.get("operation", "intersect")
        layer1 = data.get("layer1", "?")
        layer2 = data.get("layer2", "?")
        feature_count = data.get("feature_count", "?")

        operation_text = {
            "intersect": "交集",
            "union": "并集",
            "clip": "裁剪",
            "difference": "差集",
        }.get(operation, operation)

        summary = f"已完成 {layer1} 与 {layer2} 的{operation_text}分析"

        conclusion = BusinessConclusion(
            summary=summary,
            key_findings=[
                f"得到 {feature_count} 个结果要素" if isinstance(feature_count, int) else f"结果要素数：{feature_count}",
                f"操作类型：{operation_text}",
                f"使用 {result.engine} 引擎计算",
            ],
            recommendations=[
                "可用于确定空间要素的交集区域",
                "可进一步用于选址分析",
            ],
            data_quality="good",
            confidence="high",
        )

        explanation = ExplanationCard(
            title="叠置分析结果",
            what_i_did=f"对 {layer1} 和 {layer2} 执行了 {operation_text}（{operation}）操作",
            why="叠置分析用于确定空间要素的交集、合并或差集，是空间查询的核心方法",
            what_it_means=f"得到了 {feature_count} 个{operation_text}结果要素",
            data_used=f"输入图层：{layer1}, {layer2}",
        )

        output_files = []
        if data.get("output_file"):
            output_files.append(data["output_file"])
        if data.get("output_path"):
            output_files.append(data["output_path"])

        return RenderResult(
            success=True,
            summary=summary,
            conclusion=conclusion,
            explanation=explanation,
            map_file=data.get("map_file"),
            output_files=output_files,
            metrics={
                "operation": operation,
                "layer1": layer1,
                "layer2": layer2,
                "feature_count": feature_count,
                "engine": result.engine,
            },
            raw_result=result.to_dict(),
        )

    def _render_interpolation(self, result: ExecutorResult) -> RenderResult:
        """渲染插值分析结果"""
        data = result.data or {}
        method = data.get("method", "IDW")
        resolution = data.get("resolution", "?")
        feature_count = data.get("feature_count", "?")

        summary = f"已完成基于 {method} 方法的空间插值分析"

        conclusion = BusinessConclusion(
            summary=summary,
            key_findings=[
                f"插值方法：{method}",
                f"栅格分辨率：{resolution} 米" if isinstance(resolution, (int, float)) else f"分辨率：{resolution}",
                f"使用 {result.engine} 引擎计算",
            ],
            recommendations=[
                "可用于预测未采样位置的数值",
                "建议验证插值结果的准确性",
            ],
            data_quality="medium",
            confidence="medium",
        )

        explanation = ExplanationCard(
            title="空间插值分析结果",
            what_i_did=f"使用 {method} 方法基于离散采样点生成了连续空间表面",
            why="空间插值可以将稀疏的点数据转换为连续的栅格表面，用于预测和分析",
            what_it_means=f"生成了分辨率为 {resolution} 米的连续表面",
            caveats="插值结果受采样点分布和密度影响，建议验证",
        )

        output_files = []
        if data.get("output_file"):
            output_files.append(data["output_file"])

        return RenderResult(
            success=True,
            summary=summary,
            conclusion=conclusion,
            explanation=explanation,
            map_file=data.get("map_file"),
            output_files=output_files,
            metrics={
                "method": method,
                "resolution": resolution,
                "engine": result.engine,
            },
            raw_result=result.to_dict(),
        )

    def _render_viewshed(self, result: ExecutorResult) -> RenderResult:
        """渲染视域分析结果"""
        data = result.data or {}
        visible_area = data.get("visible_area_km2", "?")
        location = data.get("location", "?")

        summary = f"已完成观察点 {location} 的视域分析"

        conclusion = BusinessConclusion(
            summary=summary,
            key_findings=[
                f"可视范围约 {visible_area} 平方公里" if isinstance(visible_area, (int, float)) else f"可视范围：{visible_area}",
                f"使用 {result.engine} 引擎计算",
            ],
            recommendations=[
                "可用于评估观景点视野",
                "可用于信号基站选址",
            ],
            data_quality="good",
            confidence="high",
        )

        explanation = ExplanationCard(
            title="视域分析结果",
            what_i_did=f"计算了观察点 {location} 的可视范围",
            why="视域分析用于确定从某点能看到哪些区域，常用于观景点评估、监控选址等",
            what_it_means=f"可视范围约 {visible_area} 平方公里",
        )

        output_files = []
        if data.get("output_file"):
            output_files.append(data["output_file"])

        return RenderResult(
            success=True,
            summary=summary,
            conclusion=conclusion,
            explanation=explanation,
            map_file=data.get("map_file"),
            output_files=output_files,
            metrics={
                "visible_area_km2": visible_area,
                "location": location,
                "engine": result.engine,
            },
            raw_result=result.to_dict(),
        )

    def _render_statistics(self, result: ExecutorResult) -> RenderResult:
        """渲染统计分析结果"""
        data = result.data or {}
        hotspot_count = data.get("hotspot_count", "?")
        coldspot_count = data.get("coldspot_count", "?")

        summary = f"已完成热点冷点分析"

        conclusion = BusinessConclusion(
            summary=summary,
            key_findings=[
                f"识别出 {hotspot_count} 个热点区域" if isinstance(hotspot_count, int) else f"热点数：{hotspot_count}",
                f"识别出 {coldspot_count} 个冷点区域" if isinstance(coldspot_count, int) else f"冷点数：{coldspot_count}",
                f"使用 {result.engine} 引擎计算",
            ],
            recommendations=[
                "热点区域可能存在聚集效应",
                "建议进一步分析聚集原因",
            ],
            data_quality="good",
            confidence="medium",
        )

        explanation = ExplanationCard(
            title="热点分析结果",
            what_i_did="执行了 Getis-Ord Gi* 空间自相关分析",
            why="热点分析用于识别空间聚集的显著性区域，帮助发现数据中的空间模式",
            what_it_means=f"识别出 {hotspot_count} 个高值热点和 {coldspot_count} 个低值冷点",
        )

        output_files = []
        if data.get("output_file"):
            output_files.append(data["output_file"])

        return RenderResult(
            success=True,
            summary=summary,
            conclusion=conclusion,
            explanation=explanation,
            map_file=data.get("map_file"),
            output_files=output_files,
            metrics={
                "hotspot_count": hotspot_count,
                "coldspot_count": coldspot_count,
                "engine": result.engine,
            },
            raw_result=result.to_dict(),
        )

    def _render_raster(self, result: ExecutorResult) -> RenderResult:
        """渲染栅格分析结果"""
        data = result.data or {}
        mean_ndvi = data.get("mean_ndvi", "?")
        index_type = data.get("index_type", "NDVI")

        summary = f"已完成 {index_type} 指数计算"

        conclusion = BusinessConclusion(
            summary=summary,
            key_findings=[
                f"平均 {index_type} 值：{mean_ndvi}" if isinstance(mean_ndvi, (int, float)) else f"{index_type}：{mean_ndvi}",
                f"使用 {result.engine} 引擎计算",
            ],
            recommendations=[
                "NDVI > 0.3 表示有植被覆盖",
                "NDVI > 0.6 表示植被茂密",
            ],
            data_quality="good",
            confidence="high",
        )

        explanation = ExplanationCard(
            title=f"{index_type} 分析结果",
            what_i_did=f"计算了遥感影像的 {index_type}（归一化植被/水体指数）",
            why=f"{index_type} 是评估植被覆盖或水体的重要遥感指标",
            what_it_means=f"平均值为 {mean_ndvi}",
        )

        output_files = []
        if data.get("output_file"):
            output_files.append(data["output_file"])

        return RenderResult(
            success=True,
            summary=summary,
            conclusion=conclusion,
            explanation=explanation,
            map_file=data.get("map_file"),
            output_files=output_files,
            metrics={
                "index_type": index_type,
                "mean_value": mean_ndvi,
                "engine": result.engine,
            },
            raw_result=result.to_dict(),
        )

    def _extract_metrics(self, result: ExecutorResult) -> Dict[str, Any]:
        """从结果中提取关键指标"""
        data = result.data or {}

        # 常见的指标字段
        metric_keys = [
            "distance", "duration", "feature_count", "result_count",
            "visible_area_km2", "coverage_area", "mean_ndvi",
            "hotspot_count", "coldspot_count", "building_count",
            "total_area", "valid_pixels", "resolution",
        ]

        return {k: v for k, v in data.items() if k in metric_keys}


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


def render_result(result: ExecutorResult) -> RenderResult:
    """
    便捷函数：渲染执行结果

    这是第6层的标准出口函数。
    """
    renderer = get_renderer()
    return renderer.render(result)


__all__ = [
    "ExplanationCard",
    "BusinessConclusion",
    "RenderResult",
    "ResultRenderer",
    "get_renderer",
    "render_result",
]
