"""
GeoAgent Renderer Package
========================
结果渲染器模块：统一结果格式、生成解释卡片、生成报告
"""

from geoagent.renderer.result_renderer import (
    ResultRenderer,
    get_renderer,
    render_result,
    generate_report,
    render_basic_result,
)

__all__ = [
    "ResultRenderer",
    "get_renderer",
    "render_result",
    "generate_report",
    "render_basic_result",
]
