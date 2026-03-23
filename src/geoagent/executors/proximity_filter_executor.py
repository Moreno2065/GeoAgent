"""
ProximityFilterExecutor - 空间邻近筛选执行器
===========================================
基于空间邻近关系（缓冲区）筛选矢量要素。

职责：
  - 计算参考要素的缓冲区
  - 筛选在缓冲区范围内（或外）的要素
  - 支持"距离XX米内"等自然语言查询
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Set

from geoagent.executors.base import BaseExecutor, ExecutorResult


class ProximityFilterExecutor(BaseExecutor):
    """
    空间邻近筛选执行器

    基于缓冲区关系筛选矢量要素，支持"筛选在XX范围内"或"筛选不在XX范围内"。

    使用方式：
        executor = ProximityFilterExecutor()
        result = executor.run({
            "input_layer": "schools.shp",
            "reference_layer": "pollution_sources.shp",
            "buffer_distance": 500,
            "unit": "meters",
            "mode": "within",        # within: 在缓冲区内, outside: 在缓冲区外
            "output_file": "filtered_schools.shp",
        })
    """

    task_type = "proximity_filter"
    supported_engines: Set[str] = {"geopandas"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        """
        执行空间邻近筛选任务

        Args:
            task: 任务参数字典
                - input_layer: 待筛选的输入图层
                - reference_layer: 参考图层（用于生成缓冲区）
                - buffer_distance: 缓冲距离
                - unit: 距离单位（meters/degrees，默认 meters）
                - mode: 筛选模式（within 在缓冲区内, outside 在缓冲区外）
                - output_file: 输出文件路径

        Returns:
            ExecutorResult 统一结果格式
        """
        input_layer = task.get("input_layer")
        reference_layer = task.get("reference_layer")
        buffer_distance = task.get("buffer_distance")
        unit = task.get("unit", "meters")
        mode = task.get("mode", "within")
        output_file = task.get("output_file", "proximity_filtered.zip")

        # 参数校验
        if not input_layer:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未提供 input_layer 参数",
                engine="geopandas",
            )
        if not reference_layer:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未提供 reference_layer 参数",
                engine="geopandas",
            )
        if buffer_distance is None:
            return ExecutorResult.err(
                task_type=self.task_type,
                error="未提供 buffer_distance 参数",
                engine="geopandas",
            )

        try:
            import geopandas as gpd
            from shapely.ops import unary_union
        except ImportError as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=f"缺少依赖，请安装: pip install geopandas",
                engine="geopandas",
            )

        try:
            # 读取数据
            input_gdf = gpd.read_file(self._resolve_path(input_layer))
            reference_gdf = gpd.read_file(self._resolve_path(reference_layer))

            if len(input_gdf) == 0:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error="输入图层为空",
                    engine="geopandas",
                )
            if len(reference_gdf) == 0:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error="参考图层为空",
                    engine="geopandas",
                )

            # 统一 CRS 到 EPSG:3857（米）
            target_crs = "EPSG:3857"
            input_gdf = input_gdf.to_crs(target_crs)
            reference_gdf = reference_gdf.to_crs(target_crs)

            # 构建缓冲区
            buffer_dist = float(buffer_distance)
            if unit == "meters":
                buffer_geom = reference_gdf.geometry.buffer(buffer_dist)
            elif unit == "degrees":
                buffer_geom = reference_gdf.geometry.buffer(buffer_dist)
            else:
                buffer_geom = reference_gdf.geometry.buffer(buffer_dist)

            # 合并所有缓冲区
            merged_buffer = unary_union(buffer_geom.tolist())

            # 空间筛选
            if mode == "within":
                mask = input_gdf.geometry.within(merged_buffer)
            elif mode == "outside":
                mask = ~input_gdf.geometry.within(merged_buffer)
            else:
                return ExecutorResult.err(
                    task_type=self.task_type,
                    error=f"不支持的筛选模式: {mode}（仅支持 within / outside）",
                    engine="geopandas",
                )

            filtered_gdf = input_gdf[mask].copy()

            # 恢复原始 CRS
            original_crs = gpd.read_file(self._resolve_path(input_layer)).crs
            if original_crs:
                filtered_gdf = filtered_gdf.to_crs(original_crs)

            # 保存结果
            output_path = self._resolve_output_path(output_file, "proximity_filtered.zip")
            saved_path, driver = self.save_geodataframe(filtered_gdf, output_path)

            return ExecutorResult.ok(
                task_type=self.task_type,
                engine="geopandas",
                data={
                    "output_file": saved_path,
                    "input_count": len(input_gdf),
                    "filtered_count": len(filtered_gdf),
                    "filtered_out_count": len(input_gdf) - len(filtered_gdf),
                    "buffer_distance": buffer_distance,
                    "unit": unit,
                    "mode": mode,
                    "reference_file": reference_layer,
                },
                meta={
                    "format": driver,
                },
            )

        except Exception as e:
            return ExecutorResult.err(
                task_type=self.task_type,
                error=str(e),
                error_detail=traceback.format_exc(),
                engine="geopandas",
            )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "ProximityFilterExecutor",
]
