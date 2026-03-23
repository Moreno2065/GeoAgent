"""
Layer 6 - Result Renderer
确定性渲染层
【防幻觉铁律】
本层绝对不调用 LLM 进行"二次润色"或"总结"。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExplanationCard:
    title: str = ""
    what_it_does: str = ""
    how_to_read: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    usage_tips: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        parts = [f"### {self.title}"]
        if self.what_it_does:
            parts.append(f"\n**这是什么？** {self.what_it_does}")
        if self.how_to_read:
            parts.append("\n**如何阅读结果：**")
            for tip in self.how_to_read:
                parts.append(f"- {tip}")
        if self.limitations:
            parts.append("\n**局限性：**")
            for lim in self.limitations:
                parts.append(f"- {lim}")
        if self.usage_tips:
            parts.append("\n**使用建议：**")
            for tip in self.usage_tips:
                parts.append(f"- {tip}")
        return "\n".join(parts)


@dataclass
class BusinessConclusion:
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    data_quality: str = "unknown"
    confidence: str = "medium"

    def to_user_friendly_text(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"📋 **总结**: {self.summary}")
        if self.key_findings:
            parts.append("\n🔍 **关键发现**:")
            for finding in self.key_findings:
                parts.append(f"  - {finding}")
        if self.recommendations:
            parts.append("\n💡 **建议**:")
            for rec in self.recommendations:
                parts.append(f"  - {rec}")
        return "\n".join(parts)


@dataclass
class RenderResult:
    success: bool = True
    summary: str = ""
    conclusion: Optional[BusinessConclusion] = None
    explanation: Optional[ExplanationCard] = None
    map_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    raw_result: Optional[Dict[str, Any]] = None

    @property
    def map_file(self) -> Optional[str]:
        """兼容 pipeline 中的 map_file（单数）访问"""
        return self.map_files[0] if self.map_files else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "conclusion": (
                {
                    "summary": self.conclusion.summary,
                    "key_findings": self.conclusion.key_findings,
                    "recommendations": self.conclusion.recommendations,
                    "data_quality": self.conclusion.data_quality,
                    "confidence": self.conclusion.confidence,
                }
                if self.conclusion
                else None
            ),
            "map_files": self.map_files,
            "output_files": self.output_files,
            "metrics": self.metrics,
            "error": self.error,
        }

    def to_user_text(self) -> str:
        parts = []
        if not self.success:
            parts.append(f"❌ 执行失败: {self.error or '未知错误'}")
            return "\n".join(parts)
        if self.conclusion:
            parts.append(self.conclusion.to_user_friendly_text())
        if self.metrics:
            parts.append("\n📊 **指标**:")
            for k, v in self.metrics.items():
                parts.append(f"  - {k}: {v}")
        if self.map_files:
            parts.append(f"\n🗺️ **地图**: {', '.join(self.map_files)}")
        if self.output_files:
            parts.append(f"\n📁 **输出**: {', '.join(self.output_files)}")
        return "\n".join(parts)


class ResultRenderer:
    SCENARIO_EXPLANATIONS: Dict[str, ExplanationCard] = {
        "route": ExplanationCard(title="路径规划", what_it_does="在路网中找到两个地点之间的最优路线", how_to_read=["红线表示推荐路线", "红色标记是起点和终点", "里程和时间是实际计算结果"], limitations=["仅基于 OSM 路网数据", "不考虑实时交通"], usage_tips=["点击地图可以查看路线详情"]),
        "buffer": ExplanationCard(title="缓冲区分析", what_it_does="在选定要素周围生成指定距离的区域", how_to_read=["蓝色/绿色区域是缓冲范围", "原要素以轮廓显示", "可以叠加多个缓冲区"], limitations=["不考虑地形障碍", "是欧氏距离，非实际行走距离"], usage_tips=["调整缓冲半径可改变分析范围"]),
        "overlay": ExplanationCard(title="叠置分析", what_it_does="将多个图层叠加，提取同时满足所有条件的区域", how_to_read=["暖色（红/橙）表示高适宜性区域", "冷色（蓝/绿）表示低适宜性区域", "颜色越深，叠加条件越多"], limitations=["依赖输入数据的质量和精度", "权重设置会影响结果"], usage_tips=["检查各地物的权重是否合理"]),
        "interpolation": ExplanationCard(title="空间插值", what_it_does="根据已知点的数值，推测整个区域的数值分布", how_to_read=["颜色渐变表示数值从低到高", "等值线连接相同数值的点", "数据点以圆点标记"], limitations=["假设空间渐变是平滑的", "边界外推可能不准确"], usage_tips=["数据点越多，插值结果越可靠"]),
        "viewshed": ExplanationCard(title="视域分析", what_it_does="从某一点能看到哪些区域（考虑地形起伏）", how_to_read=["绿色区域是可见区域", "灰色区域被遮挡", "颜色深浅表示可见程度"], limitations=["假设观察者高度为 1.7m", "不考虑植被和建筑物"], usage_tips=["提高观察高度可扩大可视范围"]),
        "statistics": ExplanationCard(title="统计分析", what_it_does="分析空间数据的分布规律和统计特征", how_to_read=["热点（红色）表示高值聚集区", "冷点（蓝色）表示低值聚集区", "图表显示数值分布直方图"], limitations=["需要足够的样本点", "统计显著性需要一定数量"], usage_tips=["检查是否有异常值影响结果"]),
        "raster": ExplanationCard(title="栅格分析", what_it_does="对栅格数据进行计算和处理", how_to_read=["像元值表示实际物理量", "颜色映射表示数值大小", "分辨率决定精度"], limitations=["受限于栅格分辨率", "投影变形会影响面积计算"], usage_tips=["选择合适的像元大小"]),
        "code_sandbox": ExplanationCard(title="代码执行", what_it_does="在沙盒环境中执行 Python 代码进行分析", how_to_read=["输出区域显示执行结果", "错误信息会高亮显示", "支持查看中间变量"], limitations=["执行有资源限制", "某些库可能不可用"], usage_tips=["分步执行便于调试"]),
        "poi_search": ExplanationCard(title="周边搜索", what_it_does="搜索指定地点周边的兴趣点", how_to_read=["标记显示各 POI 位置", "颜色区分不同类别", "可查看详细信息"], limitations=["依赖 OSM 数据覆盖", "数据可能不完整"], usage_tips=["扩大搜索范围可获得更多结果"]),
        "fetch_osm": ExplanationCard(title="OSM 在线下载", what_it_does="从 OpenStreetMap 下载地理数据", how_to_read=["数据下载后保存在工作区", "可查看属性表了解详情", "支持多种要素类型"], limitations=["受限于 OSM 数据质量", "某些区域数据可能缺失"], usage_tips=["检查下载范围是否包含所需区域"]),
    }

    def __init__(self):
        self.scenario = "unknown"
        self.metrics: Dict[str, Any] = {}
        self.map_files: List[str] = []
        self.output_files: List[str] = []
        self.conclusion: Optional[BusinessConclusion] = None
        self.explanation: Optional[ExplanationCard] = None
        self._raw_result: Optional[Dict[str, Any]] = None

    def render(self, result: Any) -> RenderResult:
        self._parse_result(result)
        if not getattr(result, "success", True):
            return self._render_error(result)
        self.scenario = getattr(result, "scenario", "unknown") or "unknown"
        if self.scenario in self.SCENARIO_EXPLANATIONS:
            self.explanation = self.SCENARIO_EXPLANATIONS[self.scenario]
        if hasattr(result, "metrics") and result.metrics:
            self.metrics = result.metrics
        if hasattr(result, "conclusion") and result.conclusion:
            self.conclusion = result.conclusion
        if hasattr(result, "map_files") and result.map_files:
            self.map_files = result.map_files
        if hasattr(result, "output_files") and result.output_files:
            self.output_files = result.output_files
        self._raw_result = result.to_dict() if hasattr(result, "to_dict") else {"raw": str(result)}
        return RenderResult(
            success=True,
            summary=getattr(result, "summary", "") or self._get_default_summary(),
            conclusion=self.conclusion,
            explanation=self.explanation,
            map_files=self.map_files,
            output_files=self.output_files,
            metrics=self.metrics,
            raw_result=self._raw_result,
        )

    def _render_error(self, result: Any) -> RenderResult:
        error_msg = getattr(result, "error", None)
        if error_msg is None and hasattr(result, "message"):
            error_msg = result.message
        error_template = (
            "抱歉，分析过程中遇到问题无法完成计算。\n\n"
            f"**错误信息**: {error_msg or '未知错误'}\n\n"
            "**可能的原因**:\n"
            "- 输入数据格式不正确或缺少必要字段\n"
            "- 分析参数超出有效范围\n"
            "- 网络请求失败（涉及外部服务时）\n"
            "- 系统资源不足\n\n"
            "**建议操作**:\n"
            "- 请检查输入数据是否正确\n"
            "- 尝试简化分析范围或减少数据量\n"
            "- 如问题持续，请查看详细日志"
        )
        return RenderResult(
            success=False,
            summary="分析失败",
            error=error_msg,
            conclusion=BusinessConclusion(
                summary=error_template,
                key_findings=[],
                recommendations=["请检查输入数据是否正确", "尝试简化分析范围", "如问题持续，请查看详细日志"],
                data_quality="failed",
                confidence="low"
            ),
            raw_result=self._raw_result,
        )

    def _parse_result(self, result: Any):
        if hasattr(result, "metrics"):
            self.metrics = result.metrics or {}
        if hasattr(result, "map_files"):
            self.map_files = result.map_files or []
        if hasattr(result, "output_files"):
            self.output_files = result.output_files or []

    def _get_default_summary(self) -> str:
        summaries = {
            "route": "路径规划完成",
            "buffer": "缓冲区分析完成",
            "overlay": "叠置分析完成",
            "interpolation": "空间插值完成",
            "viewshed": "视域分析完成",
            "statistics": "统计分析完成",
            "raster": "栅格分析完成",
            "code_sandbox": "代码执行完成",
            "poi_search": "周边搜索完成",
            "fetch_osm": "OSM 数据下载完成",
            "unknown": "分析完成",
        }
        return summaries.get(self.scenario, f"{self.scenario} 分析完成")

    @classmethod
    def for_scenario(
        cls,
        scenario: str,
        metrics: Optional[Dict[str, Any]] = None,
        conclusion: Optional[BusinessConclusion] = None,
    ) -> "ResultRenderer":
        renderer = cls()
        renderer.scenario = scenario
        if metrics:
            renderer.metrics = metrics
        if conclusion:
            renderer.conclusion = conclusion
        if scenario in cls.SCENARIO_EXPLANATIONS:
            renderer.explanation = cls.SCENARIO_EXPLANATIONS[scenario]
        return renderer

    def render_with_view(
        self,
        result: Any,
        view: Optional[Dict[str, Any]] = None,
        visualization: Optional[Dict[str, Any]] = None,
    ) -> RenderResult:
        """
        渲染结果（带视图配置）

        Args:
            result: 执行器结果
            view: 视图配置字典
            visualization: 可视化配置字典

        Returns:
            RenderResult
        """
        # 先调用基本渲染
        render_result = self.render(result)
        # 可以进一步处理 view 和 visualization 配置
        return render_result
