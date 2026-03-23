"""
ChoroplethExecutor - 分级设色图执行器
====================================
基于属性字段对矢量数据进行分类渲染，生成 Choropleth 分级设色图。

职责：
  - 按属性字段值对矢量数据进行分类
  - 支持多种分类方法（等距、分位数、自然断点等）
  - 生成 Folium Choropleth 分级设色图层
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


class ChoroplethExecutor(BaseExecutor):
    """
    分级设色图执行器

    基于属性字段对矢量数据进行分类渲染。

    使用方式：
        executor = ChoroplethExecutor()
        result = executor.run({
            "input_layer": "regions.shp",
            "output_file": "choropleth.html",
            "value_field": "population",
            "classification": "quantile",   # 等距/分位数/自然断点
            "num_classes": 5,
            "fill_color": "YlOrRd",
            "legend_name": "人口数量",
        })
    """

    task_type = "choropleth"
    supported_engines: Set[str] = {"folium"}

    # Folium 内置配色方案
    COLOR_SCHEMES = [
        "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys",
        "Oranges", "OrRd", "PuBu", "PuBuGn", "PuRd", "Purples",
        "RdPu", "Reds", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd",
    ]

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行分级设色图生成任务

        Args:
            task: 任务参数字典
                - input_layer: 输入图层文件路径
                - output_file: 输出 HTML 文件路径
                - value_field: 用于分类的数值字段名
                - classification: 分类方法 (equal, quantile, natural_breaks)
                - num_classes: 分类数量（默认 5）
                - fill_color: 配色方案（默认 YlOrRd）
                - legend_name: 图例名称
                - style: 自定义样式配置

        Returns:
            ExecutorResult 统一结果格式
        """
        input_layer = task.get("input_layer") or task.get("file_path")
        if not input_layer:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未提供 input_layer 参数",
                engine="folium",
            )

        value_field = task.get("value_field")
        if not value_field:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未提供 value_field 参数（用于分类的字段名）",
                engine="folium",
            )

        output_file = task.get("output_file", "choropleth.html")
        classification = task.get("classification", "quantile")
        num_classes = task.get("num_classes", 5)
        fill_color = task.get("fill_color", "YlOrRd")
        legend_name = task.get("legend_name", value_field)

        try:
            import geopandas as gpd
            import folium
        except ImportError as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"缺少依赖，请安装: pip install geopandas folium",
                engine="folium",
            )

        try:
            resolved = self._resolve_path(input_layer)
            gdf = gpd.read_file(resolved)

            # 验证字段存在
            if value_field not in gdf.columns:
                available = [c for c in gdf.columns if c != "geometry"]
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"字段 '{value_field}' 不存在，可用字段: {available}",
                    engine="folium",
                )

            # 提取数值列并去除 NaN
            gdf = gdf.dropna(subset=[value_field])
            if len(gdf) == 0:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"字段 '{value_field}' 没有有效数值",
                    engine="folium",
                )

            # 计算分类断点
            try:
                values = gdf[value_field].astype(float)
            except (ValueError, TypeError):
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"字段 '{value_field}' 包含非数值数据",
                    engine="folium",
                )

            if classification == "equal":
                breaks = self._equal_breaks(values, num_classes)
            elif classification == "natural_breaks":
                breaks = self._natural_breaks(values, num_classes)
            else:
                breaks = self._quantile_breaks(values, num_classes)

            # 计算边界
            bounds = gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            # 创建地图
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles="OpenStreetMap",
            )

            # GeoJSON
            geojson_data = gdf.to_json()

            # 确保有唯一 ID 字段
            id_key = gdf.index.name or gdf.columns[0]
            if id_key == "geometry" or id_key not in gdf.columns:
                gdf = gdf.reset_index(drop=True)
                id_key = gdf.index.name or "id"

            folium.GeoJson(
                geojson_data,
                name="choropleth",
                style_function=lambda feature: self._style_function(
                    feature, value_field, breaks, fill_color
                ),
                tooltip=folium.GeoJsonTooltip(
                    fields=[value_field],
                    aliases=[legend_name],
                    localize=True,
                ),
                highlight_function=lambda x: {"weight": 3, "color": "#333", "fillOpacity": 0.9},
            ).add_to(m)

            # 添加图例
            self._add_legend(m, breaks, fill_color, legend_name)

            # 保存
            output_path = self._resolve_output_path(output_file, "choropleth.html")
            m.save(output_path)

            return ExecutorResult.ok(
                task_type=self.task_type,
                engine="folium",
                data={
                    "output_file": output_path,
                    "feature_count": len(gdf),
                    "value_field": value_field,
                    "classification": classification,
                    "num_classes": num_classes,
                    "breaks": [float(b) for b in breaks],
                    "fill_color": fill_color,
                },
                meta={
                    "format": "html",
                },
            )

        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=str(e),
                error_detail=traceback.format_exc(),
                engine="folium",
            )

    def _equal_breaks(self, values, num_classes):
        """等距分类"""
        import numpy as np
        min_val, max_val = values.min(), values.max()
        return np.linspace(min_val, max_val, num_classes + 1)

    def _quantile_breaks(self, values, num_classes):
        """分位数分类"""
        import numpy as np
        quantiles = np.linspace(0, 1, num_classes + 1)
        return np.quantile(values, quantiles)

    def _natural_breaks(self, values, num_classes):
        """自然断点分类（使用 Jenks 算法近似）"""
        import numpy as np
        try:
            from jenkspy import jenks_breaks
            return jenks_breaks(values, num_classes)
        except ImportError:
            return self._quantile_breaks(values, num_classes)

    def _style_function(self, feature, value_field, breaks, fill_color):
        """GeoJSON 样式函数"""
        import branca.colormap as cm
        try:
            from branca.colormap import LinearColormap
            colormap = LinearColormap(
                colors=cm.linear.get_scheme(fill_color).colors,
                vmin=breaks[0],
                vmax=breaks[-1],
            )
            value = feature["properties"].get(value_field)
            if value is not None:
                try:
                    fill_color_out = colormap(float(value))
                except (ValueError, TypeError):
                    fill_color_out = "#808080"
            else:
                fill_color_out = "#808080"
        except Exception:
            fill_color_out = "#808080"

        return {
            "fillColor": fill_color_out,
            "color": "#333",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    def _add_legend(self, m, breaks, fill_color, legend_name):
        """添加图例"""
        import branca.colormap as cm
        try:
            from branca.colormap import LinearColormap
            colormap = LinearColormap(
                colors=cm.linear.get_scheme(fill_color).colors,
                vmin=breaks[0],
                vmax=breaks[-1],
                caption=legend_name,
            )
            colormap.add_to(m)
        except Exception:
            pass


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "ChoroplethExecutor",
]
