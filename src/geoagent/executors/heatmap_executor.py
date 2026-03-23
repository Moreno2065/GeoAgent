"""
HeatmapExecutor - 热力图执行器
=============================
基于地理数据生成热力图可视化。

职责：
  - 接收矢量点/线/面数据
  - 生成 Folium 热力图层（HeatMap）
  - 支持热力图参数配置（半径、透明度、颜色等）
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


class HeatmapExecutor(BaseExecutor):
    """
    热力图执行器

    基于地理数据生成热力图可视化，支持 Folium 地图引擎。

    使用方式：
        executor = HeatmapExecutor()
        result = executor.run({
            "input_layer": "poi.shp",
            "output_file": "heatmap.html",
            # 热力图参数
            "radius": 15,
            "blur": 10,
            "max_zoom": 18,
            "gradient": {0.4: 'blue', 0.65: 'lime', 1.0: 'red'},
        })
    """

    task_type = "heatmap"
    supported_engines: Set[str] = {"folium"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行热力图生成任务

        Args:
            task: 任务参数字典
                - input_layer: 输入图层文件路径
                - output_file: 输出 HTML 文件路径
                - radius: 热力点半径（默认 15）
                - blur: 模糊程度（默认 10）
                - max_zoom: 最大缩放级别（默认 18）
                - gradient: 颜色渐变配置
                - weight_field: 权重字段名（可选）

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

        output_file = task.get("output_file", "heatmap.html")
        radius = task.get("radius", 15)
        blur = task.get("blur", 10)
        max_zoom = task.get("max_zoom", 18)
        gradient = task.get("gradient")
        weight_field = task.get("weight_field")

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
            # 读取数据
            resolved = self._resolve_path(input_layer)
            gdf = gpd.read_file(resolved)

            if len(gdf) == 0:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error="输入图层没有要素",
                    engine="folium",
                )

            # 提取坐标
            points = []
            weights = []

            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == "Point":
                    lon, lat = geom.x, geom.y
                elif geom.geom_type in ("MultiPoint", "LineString"):
                    centroid = geom.centroid
                    lon, lat = centroid.x, centroid.y
                elif geom.geom_type in ("Polygon", "MultiPolygon"):
                    centroid = geom.centroid
                    lon, lat = centroid.x, centroid.y
                else:
                    continue

                points.append([lat, lon])

                if weight_field and weight_field in row:
                    try:
                        weights.append(float(row[weight_field]))
                    except (ValueError, TypeError):
                        weights.append(1.0)
                else:
                    weights.append(1.0)

            if not points:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error="无法从输入数据中提取坐标点",
                    engine="folium",
                )

            # 计算中心点
            lats = [p[0] for p in points]
            lons = [p[1] for p in points]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            # 创建地图
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles="OpenStreetMap",
            )

            # 添加热力图层
            from folium.plugins import HeatMap

            heat_data = [[p[0], p[1], w] for p, w in zip(points, weights)]

            heat_config = {
                "heatMap": heat_data,
                "radius": radius,
                "blur": blur,
                "max_zoom": max_zoom,
            }
            if gradient:
                heat_config["gradient"] = gradient

            HeatMap(**heat_config).add_to(m)

            # 保存
            output_path = self._resolve_output_path(output_file, "heatmap.html")
            m.save(output_path)

            return ExecutorResult.ok(
                task_type=self.task_type,
                engine="folium",
                data={
                    "output_file": output_path,
                    "point_count": len(points),
                    "center": [center_lat, center_lon],
                    "radius": radius,
                    "blur": blur,
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


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "HeatmapExecutor",
]
