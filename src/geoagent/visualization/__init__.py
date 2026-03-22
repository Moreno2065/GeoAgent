"""
geoagent.visualization - 可视化引擎包
=================================

核心模块：
- engine.py: VisualizationEngine（核心渲染引擎）

导出：
    from geoagent.visualization import VisualizationEngine

使用示例：
    from geoagent.visualization import VisualizationEngine
    engine = VisualizationEngine()
    m = engine.render_multi_layer(layers=[...], view={"fit_bounds": True})
"""

from __future__ import annotations

from geoagent.visualization.engine import (
    VisualizationEngine,
    CATEGORY_COLORS,
    GRADIENT_COLORS,
    _get_color_for_category,
    _get_gradient_color,
)

__all__ = [
    "VisualizationEngine",
    "CATEGORY_COLORS",
    "GRADIENT_COLORS",
    "_get_color_for_category",
    "_get_gradient_color",
]
