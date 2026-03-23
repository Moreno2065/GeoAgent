"""
WorkflowTemplateEngine - 工作流模板引擎
======================================
核心职责：
1. 预置常见 GIS 工作流模板（POI分布图、缓冲区分析、路径规划等）
2. 根据用户自然语言自动匹配最合适的模板
3. 填充模板参数生成 GeoDSL

使用示例：
    from geoagent.workflow_templates import WorkflowTemplateEngine, WORKFLOW_TEMPLATES

    engine = WorkflowTemplateEngine()
    dsl = engine.match_and_build(
        user_input="查询芜湖市的餐厅并显示成热力图",
        extracted_params={"location": "芜湖", "query": "餐厅"}
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geoagent.layers.architecture import Scenario


# =============================================================================
# 工作流模板定义
# =============================================================================

WORKFLOW_TEMPLATES: Dict[str, Dict[str, Any]] = {

    # ── 模板1：POI 分布图 ──────────────────────────────────────────
    "poi_distribution": {
        "name": "POI 分布图",
        "description": "查询指定区域的 POI 并展示在地图上",
        "keywords": ["poi", "分布", "餐厅", "酒店", "银行", "超市", "学校", "医院",
                     "搜索", "查找附近", "周边", "查询", "找", "附近有什么",
                     "restaurant", "hotel", "bank", "supermarket", "school", "hospital"],
        "scenario": "poi_search",
        "steps": [
            {
                "step_id": "poi_query",
                "task": "poi_search",
                "description": "查询 POI 数据",
                "inputs": {},
                "parameters": {},
                "output_id": "tmp_poi",
                "depends_on": [],
            },
        ],
        "visualization": {
            "opacity": 0.8,
            "fill_opacity": 0.7,
        },
        "view": {
            "fit_bounds": True,
        },
    },

    # ── 模板2：POI 热力图 ─────────────────────────────────────────
    "poi_heatmap": {
        "name": "POI 热力图",
        "description": "将 POI 数据渲染为热力图展示密度分布",
        "keywords": ["热力图", "heatmap", "密度", "密度图", "heat map",
                     "热图", "分布密度", "密度分布"],
        "scenario": "heatmap",
        "steps": [
            {
                "step_id": "poi_query",
                "task": "poi_search",
                "description": "查询 POI 数据用于热力图",
                "inputs": {},
                "parameters": {},
                "output_id": "tmp_poi",
                "depends_on": [],
            },
            {
                "step_id": "heatmap_render",
                "task": "visualization",
                "description": "渲染热力图",
                "inputs": {"source": "tmp_poi"},
                "parameters": {"viz_type": "heatmap"},
                "output_id": "tmp_heatmap",
                "depends_on": ["poi_query"],
            },
        ],
        "visualization": {
            "heatmap": True,
        },
        "view": {
            "fit_bounds": True,
        },
    },

    # ── 模板3：缓冲区分析 ────────────────────────────────────────
    "buffer_analysis": {
        "name": "缓冲区分析",
        "description": "对输入要素做缓冲区并叠加显示",
        "keywords": ["缓冲", "buffer", "方圆", "周边", "半径", "范围",
                     "周围", "附近", "radius", "proximity"],
        "scenario": "buffer",
        "steps": [
            {
                "step_id": "buffer",
                "task": "buffer",
                "description": "生成缓冲区",
                "inputs": {},
                "parameters": {},
                "output_id": "tmp_buffer",
                "depends_on": [],
            },
        ],
        "visualization": {
            "color": "blue",
            "fill_color": "#3388ff",
            "opacity": 0.4,
            "fill_opacity": 0.3,
        },
        "view": {
            "fit_bounds": True,
        },
    },

    # ── 模板4：路径规划 ──────────────────────────────────────────
    "route_planning": {
        "name": "路径规划",
        "description": "起点到终点的路线规划与展示",
        "keywords": ["路径", "路线", "route", "步行", "导航", "从", "到",
                     "最短路径", "drive", "walk", "骑行", "驾车",
                     "从哪到哪", "起点", "终点"],
        "scenario": "route",
        "steps": [
            {
                "step_id": "route",
                "task": "route",
                "description": "计算最优路径",
                "inputs": {},
                "parameters": {},
                "output_id": "tmp_route",
                "depends_on": [],
            },
        ],
        "visualization": {
            "color": "#e74c3c",
            "stroke_width": 4,
            "opacity": 0.9,
        },
        "view": {
            "fit_bounds": True,
        },
    },

    # ── 模板5：叠置分析 ───────────────────────────────────────────
    "overlay_analysis": {
        "name": "叠置分析",
        "description": "多层叠加做交集/并集/裁剪",
        "keywords": ["叠加", "overlay", "相交", "intersect", "合并", "union",
                     "裁剪", "clip", "擦除", "difference", "叠图",
                     "叠合", "空间分析", "选址"],
        "scenario": "overlay",
        "steps": [
            {
                "step_id": "buffer_layer1",
                "task": "buffer",
                "description": "对第一图层做缓冲区",
                "inputs": {},
                "parameters": {},
                "output_id": "tmp_buf1",
                "depends_on": [],
            },
            {
                "step_id": "buffer_layer2",
                "task": "buffer",
                "description": "对第二图层做缓冲区",
                "inputs": {},
                "parameters": {},
                "output_id": "tmp_buf2",
                "depends_on": [],
            },
            {
                "step_id": "overlay",
                "task": "overlay",
                "description": "执行叠置操作",
                "inputs": {},
                "parameters": {},
                "output_id": "tmp_overlay",
                "depends_on": ["buffer_layer1", "buffer_layer2"],
            },
        ],
        "visualization": {
            "color": "#2ecc71",
            "fill_color": "#27ae60",
            "opacity": 0.5,
            "fill_opacity": 0.4,
        },
        "view": {
            "fit_bounds": True,
        },
    },

    # ── 模板6：分类地图 ───────────────────────────────────────────
    "choropleth_map": {
        "name": "分类地图",
        "description": "按数值字段做分级设色",
        "keywords": ["分级", "分类", "choropleth", "分类图", "分级图",
                     "按字段", "按数值", "按人口", "choropleth map",
                     "thematic", "专题地图", "专题图"],
        "scenario": "choropleth",
        "steps": [
            {
                "step_id": "data_load",
                "task": "data_source",
                "description": "加载数据源",
                "inputs": {},
                "parameters": {},
                "output_id": "tmp_data",
                "depends_on": [],
            },
            {
                "step_id": "choropleth",
                "task": "visualization",
                "description": "渲染分级设色图",
                "inputs": {"source": "tmp_data"},
                "parameters": {},
                "output_id": "tmp_choropleth",
                "depends_on": ["data_load"],
            },
        ],
        "visualization": {
            "choropleth": None,  # 字段名由用户指定
            "choropleth_scheme": "quantiles",
            "choropleth_classes": 5,
            "opacity": 0.8,
            "fill_opacity": 0.6,
        },
        "view": {
            "fit_bounds": True,
        },
    },

}


# =============================================================================
# 模板匹配结果
# =============================================================================

@dataclass
class TemplateMatch:
    """模板匹配结果"""
    template_id: str
    template_name: str
    confidence: float  # 0-1 匹配置信度
    matched_keywords: List[str]
    scenario: Scenario
    steps: List[Dict[str, Any]]
    visualization: Dict[str, Any]
    view: Dict[str, Any]


# =============================================================================
# 工作流模板引擎
# =============================================================================

class WorkflowTemplateEngine:
    """
    工作流模板引擎

    核心职责：
    1. 根据用户输入匹配最佳模板
    2. 用 extracted_params 填充模板
    3. 生成完整的 GeoDSL

    使用示例：
        engine = WorkflowTemplateEngine()
        dsl = engine.match_and_build(
            user_input="查询芜湖的餐厅并显示成热力图",
            extracted_params={"location": "芜湖", "query": "餐厅"}
        )
    """

    def __init__(self):
        self.templates = WORKFLOW_TEMPLATES
        self._build_keyword_index()

    def _build_keyword_index(self) -> None:
        """构建关键词 → 模板ID 的倒排索引"""
        self._keyword_to_templates: Dict[str, List[str]] = {}

        for template_id, template in self.templates.items():
            for keyword in template.get("keywords", []):
                kw_lower = keyword.lower()
                if kw_lower not in self._keyword_to_templates:
                    self._keyword_to_templates[kw_lower] = []
                self._keyword_to_templates[kw_lower].append(template_id)

    def match(self, user_input: str) -> List[TemplateMatch]:
        """
        根据用户输入匹配所有可能的模板。

        Args:
            user_input: 原始用户输入

        Returns:
            按置信度排序的匹配结果列表
        """
        user_lower = user_input.lower()
        matched_templates: Dict[str, List[str]] = {}

        # 精确匹配
        for kw, template_ids in self._keyword_to_templates.items():
            if kw in user_lower:
                for tid in template_ids:
                    if tid not in matched_templates:
                        matched_templates[tid] = []
                    matched_templates[tid].append(kw)

        # 分词匹配（兜底）
        if not matched_templates:
            words = re.split(r'[\s,，、。.;:/\\]+', user_lower)
            for word in words:
                word = word.strip()
                if len(word) < 2:
                    continue
                for kw, template_ids in self._keyword_to_templates.items():
                    if len(kw) >= 2 and (kw in word or word in kw):
                        for tid in template_ids:
                            if tid not in matched_templates:
                                matched_templates[tid] = []
                            matched_templates[tid].append(kw)

        # 计算置信度
        results = []
        for template_id, keywords in matched_templates.items():
            template = self.templates.get(template_id, {})
            template_kws = template.get("keywords", [])
            matched_chars = sum(len(kw) for kw in keywords)

            # 关键词长度加权（长关键词权重更高）
            total_weight = sum(len(kw) for kw in template_kws)
            confidence = matched_chars / max(total_weight, 1) if matched_chars > 0 else 0.0

            # 匹配关键词数量奖励（核心关键词数量）
            count_bonus = len(keywords) / max(len(template_kws), 1) * 0.2
            confidence = min(confidence + count_bonus, 1.0)

            # 精确语义匹配奖励（包含实体名词的关键词）
            semantic_bonus = 0.0
            for kw in keywords:
                if any(entity in kw for entity in ["餐厅", "酒店", "银行", "超市", "医院", "学校", "公园", "景区"]):
                    semantic_bonus += 0.1
            confidence = min(confidence + semantic_bonus, 1.0)

            try:
                scenario = Scenario(template.get("scenario", "general"))
            except (ValueError, TypeError):
                scenario = Scenario.ROUTE

            results.append(TemplateMatch(
                template_id=template_id,
                template_name=template.get("name", template_id),
                confidence=confidence,
                matched_keywords=keywords,
                scenario=scenario,
                steps=template.get("steps", []),
                visualization=template.get("visualization", {}),
                view=template.get("view", {}),
            ))

        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def match_best(self, user_input: str) -> Optional[TemplateMatch]:
        """
        获取最佳匹配的模板。

        Args:
            user_input: 原始用户输入

        Returns:
            最佳匹配的 TemplateMatch 或 None
        """
        matches = self.match(user_input)
        if matches and matches[0].confidence >= 0.05:
            return matches[0]
        return None

    def fill_template(
        self,
        template: TemplateMatch,
        extracted_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        用 extracted_params 填充模板。

        这个方法的核心逻辑：将 extracted_params 中的具体值
        填入模板的 steps/visualization/view 字段。

        Args:
            template: 模板匹配结果
            extracted_params: 从用户输入提取的参数

        Returns:
            填充后的完整模板字典
        """
        filled_steps = []
        for step in template.steps:
            filled_step = dict(step)
            # 合并 parameters 和 extracted_params
            params = dict(step.get("parameters", {}))
            inputs = dict(step.get("inputs", {}))

            # 从 extracted_params 填充 inputs
            task = step.get("task", "")
            if task == "poi_search":
                inputs.setdefault("location", extracted_params.get("location"))
                inputs.setdefault("query", extracted_params.get("query"))
            elif task == "buffer":
                inputs.setdefault("input_layer", extracted_params.get("input_layer"))
                params.setdefault("distance", extracted_params.get("distance"))
                params.setdefault("unit", extracted_params.get("unit", "meters"))
            elif task == "route":
                inputs.setdefault("start", extracted_params.get("start"))
                inputs.setdefault("end", extracted_params.get("end"))
                params.setdefault("mode", extracted_params.get("mode", "walking"))
            elif task == "overlay":
                inputs.setdefault("layer1", extracted_params.get("layer1"))
                inputs.setdefault("layer2", extracted_params.get("layer2"))
                params.setdefault("operation", extracted_params.get("operation", "intersect"))
            elif task == "visualization":
                inputs.setdefault("source", extracted_params.get("source"))

            filled_step["parameters"] = params
            filled_step["inputs"] = inputs
            filled_steps.append(filled_step)

        return {
            "steps": filled_steps,
            "visualization": dict(template.visualization),
            "view": dict(template.view),
        }

    def build_dsl_from_template(
        self,
        template: TemplateMatch,
        extracted_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        从模板构建 GeoDSL 字典。

        返回可直接传给 GeoDSL.from_dict() 的字典。

        Args:
            template: 模板匹配结果
            extracted_params: 从用户输入提取的参数

        Returns:
            GeoDSL 字典
        """
        filled = self.fill_template(template, extracted_params)

        return {
            "version": "2.1",
            "scenario": template.scenario.value if hasattr(template.scenario, 'value') else str(template.scenario),
            "task": "workflow",
            "inputs": {"user_input": extracted_params.get("user_input", "")},
            "is_workflow": True,
            "steps": filled["steps"],
            "final_output": filled["steps"][-1]["output_id"] if filled["steps"] else "final_result",
            "visualization": filled.get("visualization"),
            "view": filled.get("view"),
            "outputs": {
                "map": True,
                "summary": True,
            },
        }


# =============================================================================
# 便捷函数
# =============================================================================

_engine: Optional[WorkflowTemplateEngine] = None


def get_template_engine() -> WorkflowTemplateEngine:
    """获取模板引擎单例"""
    global _engine
    if _engine is None:
        _engine = WorkflowTemplateEngine()
    return _engine


def match_template(user_input: str) -> Optional[TemplateMatch]:
    """便捷函数：匹配最佳模板"""
    engine = get_template_engine()
    return engine.match_best(user_input)


__all__ = [
    "WORKFLOW_TEMPLATES",
    "TemplateMatch",
    "WorkflowTemplateEngine",
    "get_template_engine",
    "match_template",
]
